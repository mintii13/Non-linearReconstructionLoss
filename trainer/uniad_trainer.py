import os
import time
import copy
import glob
import shutil
import datetime
import tabulate
import torch
import torch.nn.functional as F
from util.util import makedirs, log_cfg, able, log_msg, get_log_terms, update_log_term
from util.net import trans_state_dict, print_networks, get_timepc, reduce_tensor
from util.net import get_loss_scaler, get_autocast, distribute_bn
from optim.scheduler import get_scheduler
from data import get_loader
from model import get_model
from optim import get_optim
from loss import get_loss_terms
from util.metric import get_evaluator
from timm.data import Mixup
from tqdm import tqdm
import numpy as np
from torch.nn.parallel import DistributedDataParallel as NativeDDP
try:
	from apex import amp
	from apex.parallel import DistributedDataParallel as ApexDDP
	from apex.parallel import convert_syncbn_model as ApexSyncBN
except:
	from timm.layers.norm_act import convert_sync_batchnorm as ApexSyncBN
from timm.layers.norm_act import convert_sync_batchnorm as TIMMSyncBN
from timm.utils import dispatch_clip_grad

from ._base_trainer import BaseTrainer
from . import TRAINER
from util.vis import vis_rgb_gt_amp
import wandb
import setproctitle
from einops import rearrange
setproctitle.setproctitle("Minh Tri is training...")


@TRAINER.register_module
class UniADTrainer(BaseTrainer):
	def __init__(self, cfg):
		super(UniADTrainer, self).__init__(cfg)
		if self.master and hasattr(self.cfg, 'wandb') and self.cfg.wandb.enabled:
			wandb_cfg = self.cfg.wandb
			if wandb_cfg.api_key:
				os.environ["WANDB_API_KEY"] = wandb_cfg.api_key
			self.wandb_run = wandb.init(
				project=wandb_cfg.project,
				entity=wandb_cfg.entity,
				name=wandb_cfg.name,
				tags=wandb_cfg.tags,
				notes=wandb_cfg.notes,
				mode=wandb_cfg.mode,
				group=wandb_cfg.group,
				job_type=wandb_cfg.job_type,
				resume=wandb_cfg.resume,
				id=wandb_cfg.run_id
			)
		else:
			self.wandb_run = None

		# Accumulator cho diagnostic metrics (avg qua các iter trong 1 log period)
		self._diag_accum = {}
		self._diag_count = 0

	# ============================================================
	# Helper: lấy model reference (xử lý DDP wrapper)
	# ============================================================
	def _get_model_ref(self):
		if hasattr(self.net, 'module'):
			return self.net.module
		return self.net

	# ============================================================
	# calculate_k_value: tính K và lưu lower/upper bounds
	# ============================================================
	def calculate_k_value(self):
		if not self.master:
			return

		model_ref = self._get_model_ref()
			
		if not hasattr(model_ref, 'stats_config') or not model_ref.stats_config.get('enabled', False):
			return

		log_msg(self.logger, f"Started calculating K-Channel stats (CI Ratio: {model_ref.stats_config['ci_ratio']})...")
		
		self.net.eval()
		train_loader = iter(self.train_loader)
		all_features = []
		
		with torch.no_grad():
			for i in tqdm(range(len(self.train_loader)), desc="Calculating K"):
				try:
					data = next(train_loader)
				except StopIteration:
					break
				self.set_input(data)
				feats_backbone = model_ref.net_backbone(self.imgs)
				feats_merge = model_ref.net_merge(feats_backbone)
				all_features.append(feats_merge.detach().cpu())

		full_features = torch.cat(all_features, dim=0)
		N, C, H, W = full_features.shape
		feature_np = full_features.permute(1, 0, 2, 3).reshape(C, -1).numpy()
		
		ci_ratio = model_ref.stats_config['ci_ratio']
		tail = (100 - ci_ratio) / 2.0
		numerator = 8.0 if model_ref.activation_type == 'sigmoid' else 4.8
		
		k_list      = []
		lower_list  = []
		upper_list  = []

		for c in range(C):
			channel_data = feature_np[c]
			lower = np.percentile(channel_data, tail)
			upper = np.percentile(channel_data, 100 - tail)
			r = upper - lower
			k = numerator / r if r > 1e-6 else 1.0
			k_list.append(k)
			lower_list.append(lower)
			upper_list.append(upper)
			
		# Update model buffers
		k_tensor     = torch.tensor(k_list,     dtype=torch.float32).cuda()
		lower_tensor = torch.tensor(lower_list, dtype=torch.float32).cuda()
		upper_tensor = torch.tensor(upper_list, dtype=torch.float32).cuda()

		model_ref.k_value.copy_(k_tensor)
		model_ref.lower_bound.copy_(lower_tensor)
		model_ref.upper_bound.copy_(upper_tensor)

		print("\n")
		log_msg(self.logger, f"K-Values calculated. Mean K: {k_tensor.mean():.4f} | Min K: {k_tensor.min():.4f} | Max K: {k_tensor.max():.4f}")
		print(f"K-Values calculated.** Mean K: {k_tensor.mean():.4f} | Min K: {k_tensor.min():.4f} | Max K: {k_tensor.max():.4f}", flush=True)
		log_msg(self.logger, f"Normal range (mean across channels): [{lower_tensor.mean():.4f}, {upper_tensor.mean():.4f}]")
		print(f"Normal range (mean across channels): [{lower_tensor.mean():.4f}, {upper_tensor.mean():.4f}]", flush=True)

		del all_features, full_features, feature_np
		torch.cuda.empty_cache()

	@torch.no_grad()
	def _accumulate_grad_diagnostics(self, grad_tokens):
		pre_rec_map = self.output_dict.get('pre_sigmoid_rec')  # [B, C, H, W]
		if pre_rec_map is None:
			return

		model_ref = self._get_model_ref()
		lower = model_ref.lower_bound.detach()[None, :, None, None]
		upper = model_ref.upper_bound.detach()[None, :, None, None]

		normal_mask  = (pre_rec_map >= lower) & (pre_rec_map <= upper)
		outlier_mask = ~normal_mask

		H, W = pre_rec_map.shape[2], pre_rec_map.shape[3]
		L, B, C = grad_tokens.shape

		# Dùng rearrange để consistent với model forward
		grad_map = rearrange(grad_tokens, "(h w) b c -> b c h w", h=H, w=W)

		grad_magnitude = grad_map.abs()

		grad_normal_mean  = grad_magnitude[normal_mask].mean().item()  if normal_mask.any()  else 0.0
		grad_outlier_mean = grad_magnitude[outlier_mask].mean().item() if outlier_mask.any() else 0.0
		ratio = grad_normal_mean / (grad_outlier_mean + 1e-9)
		normal_ratio_rec  = normal_mask.float().mean().item()

		key_map = {
			'Gradient/grad_normal_mean':          grad_normal_mean,
			'Gradient/grad_outlier_mean':         grad_outlier_mean,
			'Gradient/grad_normal_outlier_ratio': ratio,
			'Gradient/rec_normal_ratio':          normal_ratio_rec,
		}
		for k, v in key_map.items():
			if k not in self._diag_accum:
				self._diag_accum[k] = 0.0
			self._diag_accum[k] += v

	# ============================================================
	# _compute_diagnostic_metrics: tính tất cả diagnostic metrics
	# từ output_dict của 1 forward pass
	# ============================================================
	@torch.no_grad()
	def _compute_diagnostic_metrics(self, output_dict):
		"""
		Trả về dict các scalar metrics để log lên WandB.
		Tất cả tính trong no_grad để không ảnh hưởng training.
		"""
		metrics = {}
		model_ref = self._get_model_ref()

		# ---- 1. Pre-sigmoid range validation ----
		pre_rec  = output_dict.get('pre_sigmoid_rec')   # [B, C, H, W]
		pre_orig = output_dict.get('pre_sigmoid_orig')  # [B, C, H, W]

		if pre_rec is not None and pre_orig is not None:
			# Percentile của reconstructed (dùng flatten toàn bộ)
			rec_flat  = pre_rec.flatten()
			orig_flat = pre_orig.flatten()

			metrics['PreSigmoid/rec_p5']   = torch.quantile(rec_flat,  0.05).item()
			metrics['PreSigmoid/rec_p25']  = torch.quantile(rec_flat,  0.25).item()
			metrics['PreSigmoid/rec_p50']  = torch.quantile(rec_flat,  0.50).item()
			metrics['PreSigmoid/rec_p75']  = torch.quantile(rec_flat,  0.75).item()
			metrics['PreSigmoid/rec_p95']  = torch.quantile(rec_flat,  0.95).item()

			metrics['PreSigmoid/orig_p5']  = torch.quantile(orig_flat, 0.05).item()
			metrics['PreSigmoid/orig_p50'] = torch.quantile(orig_flat, 0.50).item()
			metrics['PreSigmoid/orig_p95'] = torch.quantile(orig_flat, 0.95).item()

			# Delta (mean absolute diff) overall
			metrics['PreSigmoid/delta_mean'] = (pre_rec - pre_orig).abs().mean().item()

			# ---- 2. Delta_normal / Delta_outlier dùng per-channel CI bounds ----
			# lower_bound, upper_bound shape: [C] -> broadcast sang [1, C, 1, 1]
			lower = model_ref.lower_bound.detach()[None, :, None, None]  # [1, C, 1, 1]
			upper = model_ref.upper_bound.detach()[None, :, None, None]  # [1, C, 1, 1]

			normal_mask  = (pre_orig >= lower) & (pre_orig <= upper)  # [B, C, H, W] bool
			outlier_mask = ~normal_mask

			sq_diff = (pre_rec - pre_orig) ** 2  # [B, C, H, W]

			if normal_mask.any():
				metrics['PreSigmoid/delta_normal'] = sq_diff[normal_mask].mean().item()
			
			if outlier_mask.any():
				metrics['PreSigmoid/delta_outlier'] = sq_diff[outlier_mask].mean().item()

			# Tỷ lệ pixel nằm trong normal range (sanity check)
			metrics['PreSigmoid/normal_ratio'] = normal_mask.float().mean().item()

		# ---- 3. Memory perturbation: cosine similarity before/after ----
		pre_mem   = output_dict.get('pre_memory_tokens')   # [L, B, hidden_dim]
		post_fuse = output_dict.get('post_fusion_tokens')  # [L, B, hidden_dim]

		if pre_mem is not None and post_fuse is not None:
			# Flatten [L*B, hidden_dim] để tính cosine similarity
			pre_flat  = pre_mem.reshape(-1, pre_mem.shape[-1])
			post_flat = post_fuse.reshape(-1, post_fuse.shape[-1])
			cos_sim_overall = F.cosine_similarity(pre_flat, post_flat, dim=-1).mean().item()
			metrics['Memory/cos_sim_before_after_fuse'] = cos_sim_overall

		# ---- 4. Channel memory: cosine sim + normalized entropy ----
		channel_result = output_dict.get('channel_result')
		if channel_result is not None:
			ch_out = channel_result['output']  # [L, B, hidden_dim]
			if pre_mem is not None:
				pre_f  = pre_mem.reshape(-1, pre_mem.shape[-1])
				ch_f   = ch_out.reshape(-1, ch_out.shape[-1])
				metrics['Memory/cos_sim_before_after_channel'] = F.cosine_similarity(pre_f, ch_f, dim=-1).mean().item()

			att_w = channel_result['att_weight']  # [L*B, mem_dim]
			mem_dim_ch = att_w.shape[-1]
			entropy_ch = -(att_w * torch.log(att_w + 1e-9)).sum(dim=-1).mean()
			max_entropy = torch.log(torch.tensor(float(mem_dim_ch), device=att_w.device))
			metrics['Memory/active_slot_ratio_channel'] = (entropy_ch / max_entropy).item()

		# ---- 5. Spatial memory: cosine sim + normalized entropy ----
		spatial_result = output_dict.get('spatial_result')
		if spatial_result is not None:
			sp_out = spatial_result['output']  # [L, B, hidden_dim]
			if pre_mem is not None:
				pre_f  = pre_mem.reshape(-1, pre_mem.shape[-1])
				sp_f   = sp_out.reshape(-1, sp_out.shape[-1])
				metrics['Memory/cos_sim_before_after_spatial'] = F.cosine_similarity(pre_f, sp_f, dim=-1).mean().item()

			att_w = spatial_result['att_weight']  # [B*C, mem_dim]
			mem_dim_sp = att_w.shape[-1]
			entropy_sp = -(att_w * torch.log(att_w + 1e-9)).sum(dim=-1).mean()
			max_entropy = torch.log(torch.tensor(float(mem_dim_sp), device=att_w.device))
			metrics['Memory/active_slot_ratio_spatial'] = (entropy_sp / max_entropy).item()

		# ---- 6. Variance của post_fusion_tokens ----
		if post_fuse is not None:
			# post_fuse: [L, B, hidden_dim]
			# Tính variance theo batch dimension để xem output có diverse không
			# Nếu variance thấp → output gần constant → decoder bypass memory
			
			# Variance theo spatial+batch dimension, giữ hidden dim
			post_fuse_flat = post_fuse.reshape(-1, post_fuse.shape[-1])  # [L*B, hidden_dim]
			
			# Variance trung bình trên mỗi hidden dimension, rồi mean over dims
			var_per_dim = post_fuse_flat.var(dim=0)  # [hidden_dim]
			metrics['Memory/post_fusion_variance_mean'] = var_per_dim.mean().item()
			metrics['Memory/post_fusion_variance_min']  = var_per_dim.min().item()

		# ---- 7. Cosine similarity giữa các memory slots ----
		# Nếu slots giống nhau → memory collapse → decoder nhận cùng 1 value
		# Nếu slots đa dạng   → memory có khả năng encode nhiều pattern khác nhau
		if channel_result is not None:
			mem_slots = channel_result['memory']  # [mem_dim, feature_dim]
			mem_norm  = F.normalize(mem_slots, p=2, dim=-1)  # [mem_dim, feature_dim]
			
			# Tính pairwise cosine similarity matrix [mem_dim, mem_dim]
			cos_matrix = torch.mm(mem_norm, mem_norm.t())
			
			# Lấy upper triangle (loại bỏ diagonal = 1)
			mask_upper = torch.triu(torch.ones_like(cos_matrix, dtype=torch.bool), diagonal=1)
			pairwise_cos = cos_matrix[mask_upper]
			
			metrics['Memory/channel_slot_cos_mean'] = pairwise_cos.mean().item()
			metrics['Memory/channel_slot_cos_max']  = pairwise_cos.max().item()
			# Nếu mean gần 0, max << 1 → slots đa dạng → memory không collapse

		if spatial_result is not None:
			mem_slots = spatial_result['memory']  # [mem_dim, H, W]
			mem_dim_s = mem_slots.shape[0]
			mem_flat  = mem_slots.view(mem_dim_s, -1)  # [mem_dim, H*W]
			mem_norm  = F.normalize(mem_flat, p=2, dim=-1)
			
			cos_matrix = torch.mm(mem_norm, mem_norm.t())
			mask_upper = torch.triu(torch.ones_like(cos_matrix, dtype=torch.bool), diagonal=1)
			pairwise_cos = cos_matrix[mask_upper]
			
			metrics['Memory/spatial_slot_cos_mean'] = pairwise_cos.mean().item()
			metrics['Memory/spatial_slot_cos_max']  = pairwise_cos.max().item()

		if channel_result is not None:
			att_w = channel_result['att_weight']  # [L*B, mem_dim]
			# Variance của attention weights theo sample dimension
			# Cao → mỗi input attend vào slots khác nhau → memory đang dùng input info
			# Thấp → mọi input attend giống nhau → memory bypass input
			att_var = att_w.var(dim=0).mean().item()  # variance across samples, mean across slots
			metrics['Memory/channel_att_weight_variance'] = att_var

		if spatial_result is not None:
			att_w = spatial_result['att_weight']  # [B*C, mem_dim]
			att_var = att_w.var(dim=0).mean().item()
			metrics['Memory/spatial_att_weight_variance'] = att_var

		return metrics

	# ============================================================
	# reset, scheduler_step, set_input
	# ============================================================
	def reset(self, isTrain=True):
		self.net.train(mode=isTrain)
		self.log_terms, self.progress = get_log_terms(
			able(self.cfg.logging.log_terms_train, isTrain, self.cfg.logging.log_terms_test),
			default_prefix=('Train' if isTrain else 'Test')
		)
		# Reset diagnostic accumulator khi reset
		self._diag_accum = {}
		self._diag_count = 0
		
	def scheduler_step(self, step):
		self.scheduler.step(step)
		update_log_term(self.log_terms.get('lr'), self.optim.param_groups[0]["lr"], 1, self.master)
		
	def set_input(self, inputs):
		self.imgs      = inputs['img'].cuda()
		self.imgs_mask = inputs['img_mask'].cuda()
		self.cls_name  = inputs['cls_name']
		self.anomaly   = inputs['anomaly']
		self.img_path  = inputs['img_path']
		self.bs        = self.imgs.shape[0]
	
	def forward(self):
		# Unpack 4 values (model giờ trả về thêm output_dict)
		self.feats_t, self.feats_s, self.pred, self.output_dict = self.net(self.imgs)

	def backward_term(self, loss_term, optim):
		optim.zero_grad()
		if self.loss_scaler:
			self.loss_scaler(loss_term, optim, clip_grad=self.cfg.loss.clip_grad, parameters=self.net.parameters(), create_graph=self.cfg.loss.create_graph)
		else:
			loss_term.backward(retain_graph=self.cfg.loss.retain_graph)
			if self.cfg.loss.clip_grad is not None:
				dispatch_clip_grad(self.net.parameters(), value=self.cfg.loss.clip_grad)
			optim.step()
		
	def optimize_parameters(self):
		if self.mixup_fn is not None:
			self.imgs, _ = self.mixup_fn(self.imgs, torch.ones(self.imgs.shape[0], device=self.imgs.device))
		with self.amp_autocast():
			self.forward()
			loss_mse = self.loss_terms['pixel'](self.feats_t, self.feats_s)
		# ---- Register gradient hook TRƯỚC backward ----
		grad_store = {}
		pre_sig_tokens = self.output_dict.get('pre_sigmoid_rec_tokens_for_grad')
		if self.master and pre_sig_tokens is not None and pre_sig_tokens.requires_grad:
			scaler_scale = self.loss_scaler.state_dict().get('scale', 1.0) \
						if self.loss_scaler else 1.0
			def save_grad(grad):
				grad_store['grad'] = (grad / scaler_scale).detach()
			pre_sig_tokens.register_hook(save_grad)
		self.backward_term(loss_mse, self.optim)
		if self.master and 'grad' in grad_store:
			self._accumulate_grad_diagnostics(grad_store['grad'])
		update_log_term(self.log_terms.get('pixel'), reduce_tensor(loss_mse, self.world_size).clone().detach().item(), 1, self.master)

		# Accumulate diagnostic metrics (chỉ trên master để tiết kiệm compute)
		if self.master:
			diag = self._compute_diagnostic_metrics(self.output_dict)
			for k, v in diag.items():
				if k not in self._diag_accum:
					self._diag_accum[k] = 0.0
				self._diag_accum[k] += v
			self._diag_count += 1

	# ============================================================
	# _finish
	# ============================================================
	def _finish(self):
		log_msg(self.logger, 'finish training')
		self.writer.close() if self.master else None
		if self.master and self.wandb_run:
			self.wandb_run.finish()
		metric_list = []
		for idx, cls_name in enumerate(self.cls_names):
			for metric in self.metrics:
				metric_list.append(self.metric_recorder[f'{metric}_{cls_name}'])
				if idx == len(self.cls_names) - 1 and len(self.cls_names) > 1:
					metric_list.append(self.metric_recorder[f'{metric}_Avg'])
		f = open(f'{self.cfg.logdir}/metric.txt', 'w')
		msg = ''
		for i in range(len(metric_list[0])):
			for j in range(len(metric_list)):
				msg += '{:3.5f}\t'.format(metric_list[j][i])
			msg += '\n'
		f.write(msg)
		f.close()
	
	# ============================================================
	# train
	# ============================================================
	def train(self):
		self.reset(isTrain=True)
		self.train_loader.sampler.set_epoch(int(self.epoch)) if self.cfg.dist else None
		if self.epoch == 0 and self.iter == 0:
			self.calculate_k_value()
			if self.cfg.dist:
				model_ref = self._get_model_ref()
				torch.distributed.broadcast(model_ref.k_value,     src=0)
				torch.distributed.broadcast(model_ref.lower_bound, src=0)
				torch.distributed.broadcast(model_ref.upper_bound, src=0)
			self.net.train()

		train_length = self.cfg.data.train_size
		train_loader = iter(self.train_loader)

		while self.epoch < self.epoch_full and self.iter < self.iter_full:
			self.scheduler_step(self.iter)
			# ---------- data ----------
			t1 = get_timepc()
			self.iter += 1
			train_data = next(train_loader)
			self.set_input(train_data)
			t2 = get_timepc()
			update_log_term(self.log_terms.get('data_t'), t2 - t1, 1, self.master)
			# ---------- optimization ----------
			self.optimize_parameters()
			t3 = get_timepc()
			update_log_term(self.log_terms.get('optim_t'), t3 - t2, 1, self.master)
			update_log_term(self.log_terms.get('batch_t'), t3 - t1, 1, self.master)
			# ---------- log ----------
			if self.master:
				if self.iter % self.cfg.logging.train_log_per == 0:
					msg = able(self.progress.get_msg(self.iter, self.iter_full, self.iter / train_length, self.iter_full / train_length), self.master, None)
					log_msg(self.logger, msg)
					
					if self.wandb_run:
						log_data = {f'Train/{k}': v.val for k, v in self.log_terms.items()}
						log_data['lr']          = self.optim.param_groups[0]["lr"]
						log_data['pixel_loss']  = self.log_terms.get('pixel').val

						# Log diagnostic metrics (trung bình qua các iter từ lần reset cuối)
						if self._diag_count > 0:
							for k, v in self._diag_accum.items():
								log_data[k] = v / self._diag_count
							# Reset accumulator sau khi log
							self._diag_accum = {}
							self._diag_count = 0

						self.wandb_run.log(log_data, step=self.iter)
						
					if self.writer:
						for k, v in self.log_terms.items():
							self.writer.add_scalar(f'Train/{k}', v.val, self.iter)
						self.writer.flush()

			if self.iter % self.cfg.logging.train_reset_log_per == 0:
				self.reset(isTrain=True)
			# ---------- update train_loader ----------
			if self.iter % train_length == 0:
				self.epoch += 1
				if self.cfg.dist and self.dist_BN != '':
					distribute_bn(self.net, self.world_size, self.dist_BN)
				self.optim.sync_lookahead() if hasattr(self.optim, 'sync_lookahead') else None
				if self.epoch == self.cfg.trainer.test_start_epoch:
					self.test()
				elif self.epoch > self.cfg.trainer.test_start_epoch and self.epoch % self.cfg.trainer.test_per_epoch == 0:
					self.test()
				else:
					self.test_ghost()
				self.cfg.total_time = get_timepc() - self.cfg.task_start_time
				total_time_str = str(datetime.timedelta(seconds=int(self.cfg.total_time)))
				eta_time_str   = str(datetime.timedelta(seconds=int(self.cfg.total_time / self.epoch * (self.epoch_full - self.epoch))))
				log_msg(self.logger, f'==> Total time: {total_time_str}\t Eta: {eta_time_str} \tLogged in \'{self.cfg.logdir}\'')
				self.save_checkpoint()
				self.reset(isTrain=True)
				self.train_loader.sampler.set_epoch(int(self.epoch)) if self.cfg.dist else None
				train_loader = iter(self.train_loader)
		self._finish()

	@torch.no_grad()
	def test_ghost(self):
		for idx, cls_name in enumerate(self.cls_names):
			for metric in self.metrics:
				self.metric_recorder[f'{metric}_{cls_name}'].append(0)
				if idx == len(self.cls_names) - 1 and len(self.cls_names) > 1:
					self.metric_recorder[f'{metric}_Avg'].append(0)

	@torch.no_grad()
	def test(self):
		if self.master:
			if os.path.exists(self.tmp_dir):
				shutil.rmtree(self.tmp_dir)
			os.makedirs(self.tmp_dir, exist_ok=True)
		self.reset(isTrain=False)
		imgs_masks, anomaly_maps, cls_names, anomalys = [], [], [], []
		batch_idx   = 0
		test_length = self.cfg.data.test_size
		test_loader = iter(self.test_loader)
		while batch_idx < test_length:
			t1 = get_timepc()
			batch_idx += 1
			test_data = next(test_loader)
			self.set_input(test_data)
			self.forward()
			loss_mse = self.loss_terms['pixel'](self.feats_t, self.feats_s)
			update_log_term(self.log_terms.get('pixel'), reduce_tensor(loss_mse, self.world_size).clone().detach().item(), 1, self.master)
			anomaly_map = self.pred.cpu().numpy()
			self.imgs_mask[self.imgs_mask > 0.5], self.imgs_mask[self.imgs_mask <= 0.5] = 1, 0
			if self.cfg.vis:
				if self.cfg.vis_dir is not None:
					root_out = self.cfg.vis_dir
				else:
					root_out = self.writer.logdir
				vis_rgb_gt_amp(self.img_path, self.imgs, self.imgs_mask.cpu().numpy().astype(int), anomaly_map, self.cfg.model.name, root_out, self.cfg.data.root.split('/')[1])
			imgs_masks.append(self.imgs_mask.cpu().numpy().astype(int))
			anomaly_maps.append(anomaly_map)
			cls_names.append(np.array(self.cls_name))
			anomalys.append(self.anomaly.cpu().numpy().astype(int))
			t2 = get_timepc()
			update_log_term(self.log_terms.get('batch_t'), t2 - t1, 1, self.master)
			print(f'\r{batch_idx}/{test_length}', end='') if self.master else None
			if self.master:
				if batch_idx % self.cfg.logging.test_log_per == 0 or batch_idx == test_length:
					msg = able(self.progress.get_msg(batch_idx, test_length, 0, 0, prefix=f'Test'), self.master, None)
					log_msg(self.logger, msg)

		# merge results
		if self.cfg.dist:
			results = dict(imgs_masks=imgs_masks, anomaly_maps=anomaly_maps, cls_names=cls_names, anomalys=anomalys)
			torch.save(results, f'{self.tmp_dir}/{self.rank}.pth', _use_new_zipfile_serialization=False)
			if self.master:
				results = dict(imgs_masks=[], anomaly_maps=[], cls_names=[], anomalys=[])
				valid_results = False
				while not valid_results:
					results_files = glob.glob(f'{self.tmp_dir}/*.pth')
					if len(results_files) != self.cfg.world_size:
						time.sleep(1)
					else:
						idx_result = 0
						while idx_result < self.cfg.world_size:
							results_file = results_files[idx_result]
							try:
								result = torch.load(results_file)
								for k, v in result.items():
									results[k].extend(v)
								idx_result += 1
							except:
								time.sleep(1)
						valid_results = True
		else:
			results = dict(imgs_masks=imgs_masks, anomaly_maps=anomaly_maps, cls_names=cls_names, anomalys=anomalys)
			
		if self.master:
			results = {k: np.concatenate(v, axis=0) for k, v in results.items()}
			msg = {}
			wandb_metric_log = {}
			all_class_metrics = {metric: [] for metric in self.metrics}
			
			for idx, cls_name in enumerate(self.cls_names):
				metric_results = self.evaluator.run(results, cls_name, self.logger)
				msg['Name'] = msg.get('Name', [])
				msg['Name'].append(cls_name)
				avg_act = True if len(self.cls_names) > 1 and idx == len(self.cls_names) - 1 else False
				msg['Name'].append('Avg') if avg_act else None
				
				for metric in self.metrics:
					metric_result = metric_results[metric] * 100
					self.metric_recorder[f'{metric}_{cls_name}'].append(metric_result)
					all_class_metrics[metric].append(metric_result)
					max_metric     = max(self.metric_recorder[f'{metric}_{cls_name}'])
					max_metric_idx = self.metric_recorder[f'{metric}_{cls_name}'].index(max_metric) + 1
					msg[metric] = msg.get(metric, [])
					msg[metric].append(metric_result)
					msg[f'{metric} (Max)'] = msg.get(f'{metric} (Max)', [])
					msg[f'{metric} (Max)'].append(f'{max_metric:.3f} ({max_metric_idx:<3d} epoch)')
					
					if avg_act:
						metric_result_avg = sum(all_class_metrics[metric]) / len(all_class_metrics[metric])
						self.metric_recorder[f'{metric}_Avg'].append(metric_result_avg)
						wandb_metric_log[f'Test/Avg/{metric}'] = metric_result_avg / 100.0
						max_metric     = max(self.metric_recorder[f'{metric}_Avg'])
						max_metric_idx = self.metric_recorder[f'{metric}_Avg'].index(max_metric) + 1
						msg[metric].append(metric_result_avg)
						msg[f'{metric} (Max)'].append(f'{max_metric:.3f} ({max_metric_idx:<3d} epoch)')
			
			msg = tabulate.tabulate(msg, headers='keys', tablefmt="pipe", floatfmt='.3f', numalign="center", stralign="center")
			log_msg(self.logger, f'\n{msg}')
			epoch_msg = f"\n==================== TEST RESULTS (EPOCH {self.epoch}) ===================="
			print(epoch_msg, flush=True)
			print(f'\n{msg}', flush=True)

			if self.wandb_run:
				wandb_metric_log['epoch'] = self.epoch
				self.wandb_run.log(wandb_metric_log)
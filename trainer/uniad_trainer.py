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
	
	def _compute_ssim_between_maps(self, pred, target, H=None, W=None):
		"""
		Compute SSIM between two feature maps of shape [B, C, H, W] or [L, B, C]
		Returns mean SSIM over batch and channels.
		"""
		if pred.dim() == 3:
			L, B, C = pred.shape
			# Nếu không có H,W từ ngoài, tự tính căn bậc hai (chỉ dùng khi feature map vuông)
			if H is None or W is None:
				H = W = int(L ** 0.5)
			pred = pred.permute(1, 2, 0).reshape(B, C, H, W)
			target = target.permute(1, 2, 0).reshape(B, C, H, W)
		
		B, C, H, W = pred.shape
		pred_flat = pred.view(B * C, H * W)
		target_flat = target.view(B * C, H * W)
		
		pred_mean = pred_flat.mean(dim=1, keepdim=True)
		target_mean = target_flat.mean(dim=1, keepdim=True)
		pred_var = pred_flat.var(dim=1, keepdim=True)
		target_var = target_flat.var(dim=1, keepdim=True)
		pred_centered = pred_flat - pred_mean
		target_centered = target_flat - target_mean
		cov = (pred_centered * target_centered).mean(dim=1, keepdim=True)
		
		c1, c2 = 0.01, 0.03
		numerator = (2 * pred_mean * target_mean + c1) * (2 * cov + c2)
		denominator = (pred_mean**2 + target_mean**2 + c1) * (pred_var + target_var + c2)
		ssim = numerator / (denominator + 1e-8)
		return ssim.mean().item()

	# ============================================================
	# _compute_diagnostic_metrics: tính tất cả diagnostic metrics
	# từ output_dict của 1 forward pass
	# ============================================================
	@torch.no_grad()
	def _compute_diagnostic_metrics(self, output_dict):
		metrics = {}
		model_ref = self._get_model_ref()

		if hasattr(model_ref, 'feature_size'):
			H, W = model_ref.feature_size
		elif hasattr(model_ref, 'net_ad') and hasattr(model_ref.net_ad, 'feature_size'):
			H, W = model_ref.net_ad.feature_size
		else:
			H = W = None

		# ----- 1. Non‑linear loss: delta normal/outlier (đã có trong _accumulate, nhưng vẫn tính lại ở đây để log) -----
		pre_rec = output_dict.get('pre_sigmoid_rec')
		pre_orig = output_dict.get('pre_sigmoid_orig')
		if pre_rec is not None and pre_orig is not None:
			lower = model_ref.lower_bound.detach()[None, :, None, None]
			upper = model_ref.upper_bound.detach()[None, :, None, None]
			normal_mask = (pre_orig >= lower) & (pre_orig <= upper)
			outlier_mask = ~normal_mask
			sq_diff = (pre_rec - pre_orig) ** 2
			if normal_mask.any():
				metrics['PreSigmoid/delta_normal'] = sq_diff[normal_mask].mean().item()
			if outlier_mask.any():
				metrics['PreSigmoid/delta_outlier'] = sq_diff[outlier_mask].mean().item()
			if 'PreSigmoid/delta_normal' in metrics and 'PreSigmoid/delta_outlier' in metrics:
				metrics['PreSigmoid/delta_outlier_normal_ratio'] = \
					metrics['PreSigmoid/delta_outlier'] / (metrics['PreSigmoid/delta_normal'] + 1e-8)
			metrics['PreSigmoid/normal_ratio'] = normal_mask.float().mean().item()

		# ----- 2. Channel memory: mean attention score (raw cosine) -----
		channel_res = output_dict.get('channel_result')
		if channel_res is not None:
			# attention_scores shape [N*B, mem_dim], chưa qua softmax, đã nhân scale=10
			att_scores = channel_res['attention_scores']   # có thể chứa -inf
			finite_mask = torch.isfinite(att_scores)
			if finite_mask.any():
				metrics['Memory/channel_attention_mean'] = att_scores[finite_mask].mean().item()
			else:
				metrics['Memory/channel_attention_mean'] = 0.0
			# active slot ratio & slot diversity
			att_w = channel_res['att_weight']
			mem_dim = att_w.shape[-1]
			entropy = -(att_w * torch.log(att_w + 1e-9)).sum(dim=-1).mean()
			max_entropy = torch.log(torch.tensor(float(mem_dim), device=att_w.device))
			metrics['Memory/active_slot_ratio_channel'] = (entropy / max_entropy).item()
			mem_slots = channel_res['memory']  # [mem_dim, C]
			mem_norm = F.normalize(mem_slots, p=2, dim=-1)
			cos_mat = torch.mm(mem_norm, mem_norm.t())
			mask_upper = torch.triu(torch.ones_like(cos_mat, dtype=torch.bool), diagonal=1)
			pairwise = cos_mat[mask_upper]
			metrics['Memory/channel_slot_cos_mean'] = pairwise.mean().item()

		# ----- 3. Spatial memory: mean SSIM similarity -----
		spatial_res = output_dict.get('spatial_result')
		if spatial_res is not None:
			ssim_sim = spatial_res['ssim_similarity']
			finite_mask = torch.isfinite(ssim_sim)
			if finite_mask.any():
				metrics['Memory/spatial_ssim_mean'] = ssim_sim[finite_mask].mean().item()
			else:
				metrics['Memory/spatial_ssim_mean'] = 0.0
			att_w = spatial_res['att_weight']
			mem_dim = att_w.shape[-1]
			entropy = -(att_w * torch.log(att_w + 1e-9)).sum(dim=-1).mean()
			max_entropy = torch.log(torch.tensor(float(mem_dim), device=att_w.device))
			metrics['Memory/active_slot_ratio_spatial'] = (entropy / max_entropy).item()
			mem_slots = spatial_res['memory']  # [mem_dim, H, W]
			mem_flat = mem_slots.view(mem_slots.shape[0], -1)
			mem_norm = F.normalize(mem_flat, p=2, dim=-1)
			cos_mat = torch.mm(mem_norm, mem_norm.t())
			mask_upper = torch.triu(torch.ones_like(cos_mat, dtype=torch.bool), diagonal=1)
			pairwise = cos_mat[mask_upper]
			metrics['Memory/spatial_slot_cos_mean'] = pairwise.mean().item()

		# ----- 4. Feature change before/after memory (fusion output) -----
		pre_mem = output_dict.get('pre_memory_tokens')          # [L,B,C]
		post_fusion = output_dict.get('post_fusion_tokens')     # after dual memory fusion
		if pre_mem is not None and post_fusion is not None:
			# per‑location cosine
			pre_flat = pre_mem.reshape(-1, pre_mem.shape[-1])
			post_flat = post_fusion.reshape(-1, post_fusion.shape[-1])
			cos_loc = F.cosine_similarity(pre_flat, post_flat, dim=-1)
			metrics['Memory/cos_pre_vs_post_fusion'] = cos_loc.mean().item()
			
			# global cosine (GAP)
			pre_gap = pre_mem.mean(dim=0)
			post_gap = post_fusion.mean(dim=0)
			cos_glob = F.cosine_similarity(pre_gap, post_gap, dim=-1).mean().item()
			metrics['Memory/cos_gap_pre_vs_post_fusion'] = cos_glob
			
			# SSIM giữa pre và post (tính theo từng channel)
			ssim_val = self._compute_ssim_between_maps(pre_mem, post_fusion,  H=H, W=W)
			metrics['Memory/ssim_pre_vs_post_fusion'] = ssim_val

		# ----- 5. Feature change after merge_proj (if exists) -----
		post_proj = output_dict.get('post_fusion_proj')
		if pre_mem is not None and post_proj is not None:
			# per‑location cosine
			pre_flat = pre_mem.reshape(-1, pre_mem.shape[-1])
			proj_flat = post_proj.reshape(-1, post_proj.shape[-1])
			cos_loc_proj = F.cosine_similarity(pre_flat, proj_flat, dim=-1)
			metrics['Memory/cos_pre_vs_post_proj'] = cos_loc_proj.mean().item()
			
			# global cosine (GAP)
			pre_gap = pre_mem.mean(dim=0)
			proj_gap = post_proj.mean(dim=0)
			cos_glob_proj = F.cosine_similarity(pre_gap, proj_gap, dim=-1).mean().item()
			metrics['Memory/cos_gap_pre_vs_post_proj'] = cos_glob_proj
			
			# SSIM
			ssim_proj_val = self._compute_ssim_between_maps(pre_mem, post_proj, H=H, W=W)
			metrics['Memory/ssim_pre_vs_post_proj'] = ssim_proj_val

		# ----- 6. Magnitude (min, mean, max) of pre_mem and post_fusion -----
		if pre_mem is not None and post_fusion is not None:
			# Lấy giá trị tuyệt đối để đo magnitude (có thể dùng cả âm/dương, nhưng abs là phổ biến)
			pre_abs = pre_mem.abs()
			post_abs = post_fusion.abs()
			
			metrics['Magnitude/pre_mem_min'] = pre_abs.min().item()
			metrics['Magnitude/pre_mem_mean'] = pre_abs.mean().item()
			metrics['Magnitude/pre_mem_max'] = pre_abs.max().item()
			
			metrics['Magnitude/post_fusion_min'] = post_abs.min().item()
			metrics['Magnitude/post_fusion_mean'] = post_abs.mean().item()
			metrics['Magnitude/post_fusion_max'] = post_abs.max().item()
		# ----- 7. Memory slots magnitude (channel memory) -----
		if channel_res is not None:
			mem_slots = channel_res['memory']  # [mem_dim, C]
			mem_abs = mem_slots.abs()
			metrics['Magnitude/channel_slots_min'] = mem_abs.min().item()
			metrics['Magnitude/channel_slots_mean'] = mem_abs.mean().item()
			metrics['Magnitude/channel_slots_max'] = mem_abs.max().item()

		# ----- 8. Memory slots magnitude (spatial memory) -----
		if spatial_res is not None:
			mem_slots = spatial_res['memory']  # [mem_dim, H, W]
			mem_abs = mem_slots.abs()
			metrics['Magnitude/spatial_slots_min'] = mem_abs.min().item()
			metrics['Magnitude/spatial_slots_mean'] = mem_abs.mean().item()
			metrics['Magnitude/spatial_slots_max'] = mem_abs.max().item()
		# ----- 9. Statistics of pre_sigmoid_rec and pre_sigmoid_orig (thêm vào nhóm PreSigmoid) -----
		pre_rec = output_dict.get('pre_sigmoid_rec')   # [B, C, H, W]
		pre_orig = output_dict.get('pre_sigmoid_orig')
		if pre_rec is not None and pre_orig is not None:
			rec_flat = pre_rec.flatten()
			orig_flat = pre_orig.flatten()
			metrics['PreSigmoid/rec_min'] = rec_flat.min().item()
			metrics['PreSigmoid/rec_mean'] = rec_flat.mean().item()
			metrics['PreSigmoid/rec_max'] = rec_flat.max().item()
			metrics['PreSigmoid/orig_min'] = orig_flat.min().item()
			metrics['PreSigmoid/orig_mean'] = orig_flat.mean().item()
			metrics['PreSigmoid/orig_max'] = orig_flat.max().item()
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
		
		# ---- Lấy các feature và mask ----
		pre_orig_map = self.output_dict.get('pre_sigmoid_orig')  # [B, C, H, W]
		feature_rec = self.output_dict['feature_rec']
		feature_align_out = self.output_dict['feature_align']
		model_ref = self._get_model_ref()
		lower = model_ref.lower_bound.detach()[None, :, None, None]
		upper = model_ref.upper_bound.detach()[None, :, None, None]
		normal_mask = (pre_orig_map >= lower) & (pre_orig_map <= upper)
		outlier_mask = ~normal_mask
		sq_diff = (feature_rec - feature_align_out) ** 2
		
		# ---- Loss chính với Gaussian weight ----
		weight = self.output_dict.get('gaussian_weight')
		if weight is not None:
			loss_main = (sq_diff * weight).mean()
		else:
			loss_main = sq_diff.mean()
		
		# ---- Loss normal và outlier chỉ để log gradient ----
		loss_normal = (sq_diff * normal_mask.float()).sum() / (normal_mask.float().sum() + 1e-9)
		loss_outlier = (sq_diff * outlier_mask.float()).sum() / (outlier_mask.float().sum() + 1e-9)
		
		# ---- Tính gradient cho logging ----
		weight_param = model_ref.net_ad.output_proj.weight
		grad_normal = torch.autograd.grad(loss_normal, weight_param, retain_graph=True, allow_unused=True)[0]
		grad_outlier = torch.autograd.grad(loss_outlier, weight_param, retain_graph=True, allow_unused=True)[0]
		
		if grad_normal is not None and grad_outlier is not None:
			grad_normal_mean = grad_normal.abs().mean().item()
			grad_outlier_mean = grad_outlier.abs().mean().item()
			ratio = grad_normal_mean / (grad_outlier_mean + 1e-9)
		else:
			grad_normal_mean = grad_outlier_mean = ratio = 0.0
		
		# Log
		if self.master and self.wandb_run and self.iter % 1000 == 0:
			self.wandb_run.log({
				'Gradient/weight_normal_mean': grad_normal_mean,
				'Gradient/weight_outlier_mean': grad_outlier_mean,
				'Gradient/weight_ratio': ratio,
			}, step=self.iter)
		
		# ---- Backward loss chính và cập nhật model ----
		self.optim.zero_grad()
		loss_main.backward()
		if self.cfg.loss.clip_grad is not None:
			dispatch_clip_grad(self.net.parameters(), value=self.cfg.loss.clip_grad)
		self.optim.step()
		
		# Log histogram (sau backward)
		if self.master and self.wandb_run and self.iter % 1000 == 0:
			if weight_param.grad is not None:
				grad_vals = weight_param.grad.detach().flatten().cpu().numpy()
				self.wandb_run.log({'Gradient/histogram': wandb.Histogram(grad_vals)}, step=self.iter)
		
		# Cập nhật log term (dùng loss_main để hiển thị)
		update_log_term(self.log_terms.get('pixel'), loss_main.detach().item(), 1, self.master)
		
		# Accumulate diagnostic metrics
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
			feature_rec = self.output_dict['feature_rec']
			feature_align_out = self.output_dict['feature_align']
			loss_mse = ((feature_rec - feature_align_out) ** 2).mean()
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
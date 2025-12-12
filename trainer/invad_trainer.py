import os
import time
import copy
import glob
import shutil
import datetime
import tabulate
import torch
import wandb
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


@TRAINER.register_module
class InvADTrainer(BaseTrainer):
    def __init__(self, cfg):
        super(InvADTrainer, self).__init__(cfg)

        # === ADDED: WandB Init (Giống ViTAD) ===
        if self.master and hasattr(self.cfg, 'wandb') and self.cfg.wandb.enabled:
            wandb_cfg = self.cfg.wandb
            if wandb_cfg.api_key:
                os.environ["WANDB_API_KEY"] = wandb_cfg.api_key
            
            self.wandb_run = wandb.init(
                project=wandb_cfg.project,
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

    def reset(self, isTrain=True):
        self.net.train(mode=isTrain)
        self.log_terms, self.progress = get_log_terms(able(self.cfg.logging.log_terms_train, isTrain, self.cfg.logging.log_terms_test), default_prefix=('Train' if isTrain else 'Test'))
        
    def scheduler_step(self, step):
        self.scheduler.step(step)
        update_log_term(self.log_terms.get('lr'), self.optim.param_groups[0]["lr"], 1, self.master)
        
    def set_input(self, inputs):
        self.imgs = inputs['img'].cuda()
        self.imgs_mask = inputs['img_mask'].cuda()
        self.cls_name = inputs['cls_name']
        self.anomaly = inputs['anomaly']
        self.img_path = inputs['img_path']
        self.bs = self.imgs.shape[0]
    
    def forward(self):
        self.feats, self.feats_pred = self.net(self.imgs)

    def backward_term(self, loss_term, optim):
        optim.zero_grad()
        if self.loss_scaler:
            self.loss_scaler(loss_term, optim, clip_grad=self.cfg.loss.clip_grad, parameters=self.net.parameters(), create_graph=self.cfg.loss.create_graph)
        else:
            loss_term.backward(retain_graph=self.cfg.loss.retain_graph)
            if self.cfg.loss.clip_grad is not None:
                dispatch_clip_grad(self.net.parameters(), value=self.cfg.loss.clip_grad)
            optim.step()

    def calculate_k_value(self):
        if not self.master:
            return

        model_ref = self.net.module if hasattr(self.net, 'module') else self.net
        
        stats_cfg = getattr(model_ref, 'stats_config', None)
        if not stats_cfg or not stats_cfg.get('enabled', False):
            return

        if model_ref.is_k_calculated and len(model_ref.k_scales) > 0:
            log_msg(self.logger, "K-Values already calculated.")
            return

        apply_indices = stats_cfg.get('apply_indices', [])
        log_msg(self.logger, f"Calculating K Stats (CI: {stats_cfg['ci_ratio']}) for layers {apply_indices} on 100% Data...")
        
        self.net.eval()
        train_loader = iter(self.train_loader)
        
        ci_ratio = stats_cfg['ci_ratio']
        q_lower = (100 - ci_ratio) / 2.0 / 100.0
        q_upper = 1.0 - q_lower
        numerator = 8.0 if stats_cfg.get('activation_type') == 'sigmoid' else 4.8 

        # Dictionary chỉ lưu thống kê của các layer cần thiết
        batch_stats = {} 
        num_scales = 0 # Để đếm tổng số scale output của model

        with torch.no_grad():
            for _ in tqdm(range(len(self.train_loader)), desc="Calculating K"):
                try:
                    data = next(train_loader)
                except StopIteration:
                    break
                
                self.set_input(data)
                feats = model_ref.net_encoder(self.imgs) 
                num_scales = len(feats) 
                
                for scale_idx, feat in enumerate(feats):
                    if scale_idx not in apply_indices:
                        continue 
                    
                    if scale_idx not in batch_stats:
                        batch_stats[scale_idx] = {'lower': [], 'upper': []}
                    
                    # Reshape: [B, C, H, W] -> [C, -1]
                    feat_flat = feat.permute(1, 0, 2, 3).reshape(feat.shape[1], -1)
                    
                    lower_val = torch.quantile(feat_flat, q_lower, dim=1)
                    upper_val = torch.quantile(feat_flat, q_upper, dim=1)
                    
                    batch_stats[scale_idx]['lower'].append(lower_val.cpu())
                    batch_stats[scale_idx]['upper'].append(upper_val.cpu())

        # Reset ParameterList
        model_ref.k_scales = torch.nn.ParameterList()
        
        # Duyệt qua tất cả các scale index
        for scale_idx in range(num_scales):
            
            # 1. Nếu layer này nằm trong danh sách cần tính -> Tính K từ stats
            if scale_idx in batch_stats:
                avg_lower = torch.stack(batch_stats[scale_idx]['lower']).mean(dim=0)
                avg_upper = torch.stack(batch_stats[scale_idx]['upper']).mean(dim=0)
                
                r = avg_upper - avg_lower

                # === DEBUG PRINT START: STATS TRƯỚC KHI TÍNH K ===
                if self.master:
                    print(f"\n{'='*20} LAYER {scale_idx} RAW STATS {'='*20}")
                    print(f">> Lower Bound (q={q_lower:.3f}):")
                    print(f"   Mean: {avg_lower.mean().item():.6f} | Min: {avg_lower.min().item():.6f} | Max: {avg_lower.max().item():.6f}")
                    
                    print(f">> Upper Bound (q={q_upper:.3f}):")
                    print(f"   Mean: {avg_upper.mean().item():.6f} | Min: {avg_upper.min().item():.6f} | Max: {avg_upper.max().item():.6f}")
                    
                    print(f">> Range R (Upper - Lower):")
                    # R rất nhỏ -> K rất lớn -> Dễ bão hòa Sigmoid
                    print(f"   Mean: {r.mean().item():.6f} | Min: {r.min().item():.6f} | Max: {r.max().item():.6f}")
                    
                    # Đếm số lượng kênh có Range gần bằng 0 (nguy hiểm)
                    zero_range_count = (r < 1e-6).sum().item()
                    if zero_range_count > 0:
                        print(f"!! WARNING: Có {zero_range_count} channels có Range ~ 0 (sẽ bị gán K=1.0)")
                    print(f"{'='*60}")
                # =================================================

                k_tensor = torch.zeros_like(r)
                mask = r > 1e-6
                k_tensor[mask] = numerator / r[mask]
                k_tensor[~mask] = 1.0 # Default cho kênh chết
                
                log_msg(self.logger, f"Scale {scale_idx}: Calculated Mean K={k_tensor.mean():.4f} (Max K={k_tensor.max():.4f})")
            
            # 2. Nếu layer này bị bỏ qua
            else:
                k_tensor = torch.tensor([1.0]) 
                log_msg(self.logger, f"Scale {scale_idx}: Skipped (Default K=1.0)")

            if k_tensor.numel() > 1:
                k_tensor = k_tensor.view(1, -1, 1, 1).cuda()
            else:
                k_tensor = k_tensor.cuda()

            param = torch.nn.Parameter(k_tensor, requires_grad=False)
            model_ref.k_scales.append(param)

        model_ref.is_k_calculated = True
        
        del batch_stats
        torch.cuda.empty_cache()
        
    def optimize_parameters(self):
        if self.mixup_fn is not None:
            self.imgs, _ = self.mixup_fn(self.imgs, torch.ones(self.imgs.shape[0], device=self.imgs.device))
        
        with self.amp_autocast():
            self.forward() # Sinh ra self.feats (Target) và self.feats_pred (Reconstructed)
            
            model_ref = self.net.module if hasattr(self.net, 'module') else self.net
            stats_cfg = getattr(model_ref, 'stats_config', None)
            
            loss_mse = 0
            
            # --- KIỂM TRA ENABLE CONFIG ---
            use_sigmoid_logic = False
            if stats_cfg and stats_cfg.get('enabled', False) and model_ref.is_k_calculated:
                use_sigmoid_logic = True
                apply_indices = stats_cfg.get('apply_indices', []) # Ví dụ [1, 2]

            # Tính Loss
            # InvAD feats thường là list [scale1, scale2, scale3]
            if isinstance(self.feats, list):
                for idx, (feat_gt, feat_pred) in enumerate(zip(self.feats, self.feats_pred)):
                    
                    # LOGIC CHÍNH: Nếu enable và đúng layer -> Áp dụng Sigmoid + K
                    if use_sigmoid_logic and idx in apply_indices and idx < len(model_ref.k_scales):
                        k = model_ref.k_scales[idx]
                        
                        # Apply Sigmoid(k * feature) cho cả Target và Pred
                        feat_gt_trans = torch.sigmoid(feat_gt * k)
                        feat_pred_trans = torch.sigmoid(feat_pred * k)
                        
                        loss_mse += self.loss_terms['pixel'](feat_gt_trans, feat_pred_trans)
                    
                    # LOGIC GỐC: MSE thuần
                    else:
                        loss_mse += self.loss_terms['pixel'](feat_gt, feat_pred)
            else:
                loss_mse = self.loss_terms['pixel'](self.feats, self.feats_pred)

        self.backward_term(loss_mse, self.optim)
        update_log_term(self.log_terms.get('pixel'), reduce_tensor(loss_mse, self.world_size).clone().detach().item(), 1, self.master)
    
    def _finish(self):
        log_msg(self.logger, 'finish training')
        self.writer.close() if self.master else None
        
        # === ADDED: WandB Finish ===
        if self.master and self.wandb_run:
            self.wandb_run.finish()
        # ===========================

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
    
    def train(self):
        self.reset(isTrain=True)
        self.train_loader.sampler.set_epoch(int(self.epoch)) if self.cfg.dist else None
        
        # === ADDED: Calculate K at Epoch 0 ===
        if self.epoch == 0 and self.iter == 0:
            self.calculate_k_value()
            if self.cfg.dist:
                torch.distributed.barrier()
        # =====================================

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
                    
                    # === ADDED: WandB Log Train ===
                    if self.wandb_run:
                        log_data = {f'Train/{k}': v.val for k, v in self.log_terms.items()}
                        log_data['lr'] = self.optim.param_groups[0]["lr"]
                        log_data['pixel_loss'] = self.log_terms.get('pixel').val
                        self.wandb_run.log(log_data, step=self.iter)
                    # ==============================

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
                eta_time_str = str(datetime.timedelta(seconds=int(self.cfg.total_time / self.epoch * (self.epoch_full - self.epoch))))
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
        batch_idx = 0
        test_length = self.cfg.data.test_size
        test_loader = iter(self.test_loader)
        while batch_idx < test_length:
            t1 = get_timepc()
            batch_idx += 1
            test_data = next(test_loader)
            self.set_input(test_data)
            self.forward()
            
            # Tính loss để log (nếu cần)
            loss_mse = 0
            if isinstance(self.feats, list):
                for ft, fp in zip(self.feats, self.feats_pred):
                    loss_mse += self.loss_terms['pixel'](ft, fp)
            else:
                loss_mse = self.loss_terms['pixel'](self.feats, self.feats_pred)
            
            update_log_term(self.log_terms.get('pixel'), reduce_tensor(loss_mse, self.world_size).clone().detach().item(), 1, self.master)
            
            # get anomaly maps
            anomaly_map, _ = self.evaluator.cal_anomaly_map(self.feats, self.feats_pred, [self.imgs.shape[2], self.imgs.shape[3]], uni_am=self.cfg.uni_am, use_cos=self.cfg.use_cos, amap_mode='add', gaussian_sigma=4)
            
            self.imgs_mask[self.imgs_mask > 0.5], self.imgs_mask[self.imgs_mask <= 0.5] = 1, 0
            if self.cfg.vis and self.master:
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
            # ---------- log ----------
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
            wandb_metric_log = {} # === ADDED: WandB Avg Log ===

            for idx, cls_name in enumerate(self.cls_names):
                metric_results = self.evaluator.run(results, cls_name, self.logger)
                msg['Name'] = msg.get('Name', [])
                msg['Name'].append(cls_name)
                avg_act = True if len(self.cls_names) > 1 and idx == len(self.cls_names) - 1 else False
                msg['Name'].append('Avg') if avg_act else None
                # msg += f'\n{cls_name:<10}'
                for metric in self.metrics:
                    metric_result = metric_results[metric] * 100
                    self.metric_recorder[f'{metric}_{cls_name}'].append(metric_result)
                    max_metric = max(self.metric_recorder[f'{metric}_{cls_name}'])
                    max_metric_idx = self.metric_recorder[f'{metric}_{cls_name}'].index(max_metric) + 1
                    msg[metric] = msg.get(metric, [])
                    msg[metric].append(metric_result)
                    msg[f'{metric} (Max)'] = msg.get(f'{metric} (Max)', [])
                    msg[f'{metric} (Max)'].append(f'{max_metric:.3f} ({max_metric_idx:<3d} epoch)')
                    if avg_act:
                        metric_result_avg = sum(msg[metric]) / len(msg[metric])
                        self.metric_recorder[f'{metric}_Avg'].append(metric_result_avg)
                        max_metric = max(self.metric_recorder[f'{metric}_Avg'])
                        max_metric_idx = self.metric_recorder[f'{metric}_Avg'].index(max_metric) + 1
                        msg[metric].append(metric_result_avg)
                        msg[f'{metric} (Max)'].append(f'{max_metric:.3f} ({max_metric_idx:<3d} epoch)')
                        
                        # === ADDED: Collect Avg for WandB ===
                        wandb_metric_log[f'Test/Avg/{metric}'] = metric_result_avg / 100.0
                        
            msg = tabulate.tabulate(msg, headers='keys', tablefmt="pipe", floatfmt='.3f', numalign="center", stralign="center", )
            log_msg(self.logger, f'\n{msg}')
            
            # === ADDED: Send WandB Log ===
            if self.wandb_run:
                wandb_metric_log['epoch'] = self.epoch
                self.wandb_run.log(wandb_metric_log)
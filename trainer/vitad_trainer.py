import os
import copy
import glob
import shutil
import datetime
import time
import wandb # Import wandb

import tabulate
import torch
from util.util import makedirs, log_cfg, able, log_msg, get_log_terms, update_log_term
from util.net import trans_state_dict, print_networks, get_timepc, reduce_tensor
from util.net import get_loss_scaler, get_autocast, distribute_bn
from tqdm import tqdm
import numpy as np
from ._base_trainer import BaseTrainer
from . import TRAINER
from util.vis import vis_rgb_gt_amp

@TRAINER.register_module
class ViTADTrainer(BaseTrainer):
    def __init__(self, cfg):
        super(ViTADTrainer, self).__init__(cfg)
        
        # === WandB Init (Copy từ UniAD logic) ===
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

    def calculate_k_values(self):
        """
        Tính toán giá trị K cho từng channel của từng scale (Multi-scale Statistics)
        """
        if not self.master:
            return

        # Lấy tham chiếu đến model thực (xử lý trường hợp DDP)
        if hasattr(self.net, 'module'):
            model_ref = self.net.module
        else:
            model_ref = self.net
            
        if not hasattr(model_ref, 'stats_config') or not model_ref.stats_config or not model_ref.stats_config.get('enabled', False):
            return

        # Nếu đã tính rồi thì bỏ qua (trường hợp resume training)
        if model_ref.is_k_calculated and len(model_ref.k_scales) > 0:
            log_msg(self.logger, "K-Values already loaded/calculated.")
            return

        log_msg(self.logger, f"Started calculating K-Channel stats (CI Ratio: {model_ref.stats_config['ci_ratio']})...")
        
        self.net.eval()
        train_loader = iter(self.train_loader)
        
        # Danh sách lưu features cho từng scale. 
        # Ví dụ: list of list. [ [Scale1_Batch1, Scale1_Batch2], [Scale2_Batch1, ...] ]
        all_features_by_scale = {} 
        
        with torch.no_grad():
            for i in tqdm(range(len(self.train_loader)), desc="Calculating K"):
                try:
                    data = next(train_loader)
                except StopIteration:
                    break
                
                self.set_input(data)
                
                # CHÚ Ý: Chúng ta gọi trực tiếp net_t để lấy RAW features (chưa qua sigmoid)
                # ViTAD.forward đã áp dụng sigmoid bên trong, nên không gọi self.net(imgs)
                feats_t, _ = model_ref.net_t(self.imgs)
                
                for scale_idx, feat in enumerate(feats_t):
                    if scale_idx not in all_features_by_scale:
                        all_features_by_scale[scale_idx] = []
                    # Move về CPU để tiết kiệm GPU memory
                    all_features_by_scale[scale_idx].append(feat.detach().cpu())

        # Tính K cho từng scale
        ci_ratio = model_ref.stats_config['ci_ratio']
        tail = (100 - ci_ratio) / 2.0
        numerator = 8.0 if model_ref.activation_type == 'sigmoid' else 4.8
        
        # Reset k_scales trong model
        model_ref.k_scales = torch.nn.ParameterList()
        
        for scale_idx in sorted(all_features_by_scale.keys()):
            # Gộp các batch lại: (N_total, C, H, W)
            full_features = torch.cat(all_features_by_scale[scale_idx], dim=0)
            N, C, H, W = full_features.shape
            
            # Reshape: (C, N*H*W)
            feature_np = full_features.permute(1, 0, 2, 3).reshape(C, -1).numpy()
            
            k_list_scale = []
            for c in range(C):
                channel_data = feature_np[c]
                lower = np.percentile(channel_data, tail)
                upper = np.percentile(channel_data, 100 - tail)
                r = upper - lower
                
                if r > 1e-6:
                    k = numerator / r
                else:
                    k = 1.0
                k_list_scale.append(k)
            
            # Tạo tensor (1, C, 1, 1) để broadcast
            k_tensor = torch.tensor(k_list_scale, dtype=torch.float32).view(1, -1, 1, 1).cuda()
            
            # Đưa vào ParameterList (requires_grad=False để cố định)
            param = torch.nn.Parameter(k_tensor, requires_grad=False)
            model_ref.k_scales.append(param)
            
            log_msg(self.logger, f"Scale {scale_idx}: Mean K: {k_tensor.mean():.4f}")

        model_ref.is_k_calculated = True
        
        # Dọn dẹp bộ nhớ
        del all_features_by_scale, full_features, feature_np
        torch.cuda.empty_cache()

    def set_input(self, inputs):
        self.imgs = inputs['img'].cuda()
        self.imgs_mask = inputs['img_mask'].cuda()
        self.cls_name = inputs['cls_name']
        self.anomaly = inputs['anomaly']
        self.img_path = inputs['img_path']
        self.bs = self.imgs.shape[0]

    def forward(self):
        self.feats_t, self.feats_s = self.net(self.imgs)

    def optimize_parameters(self):
        if self.mixup_fn is not None:
            self.imgs, _ = self.mixup_fn(self.imgs, torch.ones(self.imgs.shape[0], device=self.imgs.device))
        with self.amp_autocast():
            self.forward()
            # Tính loss tổng hợp từ tất cả các scales
            loss_mse = 0
            # UniAD dùng 'pixel' loss term, ViTAD cũng vậy nhưng feats là list
            # Cần đảm bảo hàm loss xử lý được list hoặc ta loop ở đây
            if isinstance(self.feats_t, list):
                for ft, fs in zip(self.feats_t, self.feats_s):
                    loss_mse += self.loss_terms['pixel'](ft, fs)
            else:
                loss_mse = self.loss_terms['pixel'](self.feats_t, self.feats_s)
                
        self.backward_term(loss_mse, self.optim)
        update_log_term(self.log_terms.get('pixel'), reduce_tensor(loss_mse, self.world_size).clone().detach().item(), 1,
                        self.master)

    def train(self):
        self.reset(isTrain=True)
        self.train_loader.sampler.set_epoch(int(self.epoch)) if self.cfg.dist else None
        
        # === TÍNH TOÁN K VALUES TẠI ĐÂY ===
        if self.epoch == 0 and self.iter == 0:
            self.calculate_k_values()
            
            # Broadcast K values nếu dùng DDP
            if self.cfg.dist:
                # Logic broadcast cho ParameterList phức tạp hơn 1 chút, 
                # nhưng do nó là buffer/param của model, PyTorch DDP sẽ tự sync
                # khi forward pass đầu tiên diễn ra nếu nó được đăng ký đúng cách.
                # Tuy nhiên, để an toàn, ta nên đảm bảo rank 0 tính xong rồi mới train.
                torch.distributed.barrier()
                # Lưu ý: Với DDP chuẩn, buffers sẽ được broadcast từ rank 0 sang các rank khác
                # ở mỗi lần forward. Vì ta set self.net.module.k_scales ở rank 0,
                # ta cần đảm bảo các rank khác cũng nhận được.
                # Cách đơn giản nhất: load state dict từ rank 0 (nhưng hơi chậm).
                # Hoặc tin tưởng vào cơ chế broadcast buffers của DDP.
                pass

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
                    
                    # === Log WandB ===
                    if self.wandb_run:
                        log_data = {f'Train/{k}': v.val for k, v in self.log_terms.items()}
                        log_data['lr'] = self.optim.param_groups[0]["lr"]
                        log_data['pixel_loss'] = self.log_terms.get('pixel').val
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
                    self.test_ghost() # Log 0 cho metrics nếu không test
                    
                self.cfg.total_time = get_timepc() - self.cfg.task_start_time
                self.save_checkpoint()
                self.reset(isTrain=True)
                self.train_loader.sampler.set_epoch(int(self.epoch)) if self.cfg.dist else None
                train_loader = iter(self.train_loader)
                
        self._finish()

    def _finish(self):
        log_msg(self.logger, 'finish training')
        self.writer.close() if self.master else None
        if self.master and self.wandb_run:
            self.wandb_run.finish()

    @torch.no_grad()
    def test_ghost(self):
        # Hàm này để giữ cho log file có định dạng đúng ngay cả khi không chạy test thật
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
            
            # Tính loss pixel để log (nếu cần)
            loss_mse = 0
            if isinstance(self.feats_t, list):
                for ft, fs in zip(self.feats_t, self.feats_s):
                    loss_mse += self.loss_terms['pixel'](ft, fs)
            else:
                loss_mse = self.loss_terms['pixel'](self.feats_t, self.feats_s)
            
            update_log_term(self.log_terms.get('pixel'), reduce_tensor(loss_mse, self.world_size).clone().detach().item(),
                            1, self.master)
                            
            # Get anomaly maps
            # Lưu ý: cal_anomaly_map cần xử lý list features. Đảm bảo evaluator của bạn hỗ trợ list.
            # Thường thì evaluator sẽ sum anomaly map của các scales lại.
            anomaly_map, _ = self.evaluator.cal_anomaly_map(self.feats_t, self.feats_s,
                                                            [self.imgs.shape[2], self.imgs.shape[3]], uni_am=False,
                                                            amap_mode='add', gaussian_sigma=4)
                                                            
            self.imgs_mask[self.imgs_mask > 0.5], self.imgs_mask[self.imgs_mask <= 0.5] = 1, 0
            if self.cfg.vis and self.master: # Chỉ master visualize
                if self.cfg.vis_dir is not None:
                    root_out = self.cfg.vis_dir
                else:
                    root_out = self.writer.logdir
                vis_rgb_gt_amp(self.img_path, self.imgs, self.imgs_mask.cpu().numpy().astype(int), anomaly_map,
                               self.cfg.model.name, root_out, self.cfg.data.root.split('/')[1])
                               
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
                    
        # Merge results (giữ nguyên logic DDP của bạn)
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
            wandb_metric_log = {} # Log AVG cho Wandb
            
            for idx, cls_name in enumerate(self.cls_names):
                metric_results = self.evaluator.run(results, cls_name, self.logger)
                msg['Name'] = msg.get('Name', [])
                msg['Name'].append(cls_name)
                
                avg_act = True if len(self.cls_names) > 1 and idx == len(self.cls_names) - 1 else False
                msg['Name'].append('Avg') if avg_act else None
                
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
                        
                        # Set WandB log for AVG
                        wandb_metric_log[f'Test/Avg/{metric}'] = metric_result_avg / 100.0

            msg = tabulate.tabulate(msg, headers='keys', tablefmt="pipe", floatfmt='.3f', numalign="center", stralign="center")
            log_msg(self.logger, f'\n{msg}')
            
            # Log WandB Metric
            if self.wandb_run:
                wandb_metric_log['epoch'] = self.epoch
                self.wandb_run.log(wandb_metric_log)
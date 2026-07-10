"""
Experiment runner for TimeMosaic_FusionV2.

Key improvements over exp_Fusion.py:
  1. Log-decay time-decay loss weights
     → Near-term predictions get higher weight, distant predictions lower.
     → Applied multiplicatively to the L1 forecast loss.
     → Eliminates the need for lam_moe and lam_prefix_moe hyperparameters
       since MoE load balancing is now handled by learnable bias gates.

  2. Simplified loss: no explicit MoE load-balancing loss terms
     → MoE gates use auto-updating bias (see layers/MoE_Gate.py)

  3. Keeps: patch classification regularization
"""

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
warnings.filterwarnings('ignore')


def get_log_decay_weights(pred_len: int, device: torch.device) -> torch.Tensor:
    """
    Logarithmically decayed time-step weights.

    Earlier time steps get higher weight, encouraging the model to
    prioritize near-term accuracy while still optimizing for the full horizon.

    Formula (from Kairos):
      w_i = (1 / L) * (log(L) - log(i + 1))
      where L = pred_len, i in [0, L-1]

    Returns:
        weights: [pred_len]  sum-normalized to pred_len
    """
    i_array = np.linspace(1 + 1e-5, pred_len - 1e-3, pred_len)
    weights = (1.0 / pred_len) * (np.log(pred_len) - np.log(i_array))
    weights = torch.tensor(weights, dtype=torch.float32, device=device)
    # Normalize so mean weight = 1.0 (same scale as uniform weighting)
    weights = weights / weights.mean()
    return weights


class Exp_FusionV2(Exp_Basic):
    def __init__(self, args):
        super(Exp_FusionV2, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.L1Loss(reduction='none')

    def _compute_patch_reg_loss(self, criterion):
        """Patch classification regularization: encourage uniform usage of patch lengths."""
        # Access patch_embedding inside DataParallel wrapper
        model = self.model.module if hasattr(self.model, 'module') else self.model
        cls_soft = model.patch_embedding.latest_cls_soft
        if cls_soft is not None:
            current_ratio = cls_soft.mean(dim=0)
            target_ratio = torch.full_like(current_ratio, 1.0 / len(current_ratio)).detach()
            # Use MAE criterion
            nn_l1 = nn.L1Loss()
            return 0.001 * nn_l1(current_ratio, target_ratio)
        return torch.tensor(0.)

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        pred_len = self.args.pred_len
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs, _ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs, _ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -pred_len:, f_dim:]
                batch_y = batch_y[:, -pred_len:, f_dim:].to(self.device)
                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()
                # Validation uses uniform MAE for fair comparison
                vali_criterion = nn.L1Loss()
                total_loss.append(vali_criterion(pred, true))
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()  # L1Loss(reduction='none')

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        accumulation_steps = getattr(self.args, 'accumulation_steps', 1)
        use_log_decay = getattr(self.args, 'use_log_decay', True)
        pred_len = self.args.pred_len

        # Pre-compute log-decay weights
        if use_log_decay:
            time_weights = get_log_decay_weights(pred_len, self.device)  # [pred_len]
        else:
            time_weights = None

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                if i % accumulation_steps == 0:
                    model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # Masking for reconstruction
                if getattr(self.args, 'mask_ratio', 0) > 0:
                    B, T, C = batch_x.shape
                    mask = torch.rand(B, T, C, device=batch_x.device) < self.args.mask_ratio
                elif getattr(self.args, 'mask_ratio_patch', 0) > 0:
                    patch_num = int((self.args.seq_len - self.args.patch_len) / self.args.stride + 2)
                    B, T, C = batch_x.shape
                    mask = (torch.rand(B * C, patch_num, device=batch_x.device) < self.args.mask_ratio_patch)
                    mask = mask.unsqueeze(-1).expand(-1, -1, self.args.d_model)
                else:
                    mask = None

                # ── Forward ──
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        result = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, mask)
                        loss = self._compute_fusionv2_loss(
                            criterion, result, batch_x, batch_y, mask,
                            time_weights)
                        train_loss.append(loss.item())
                else:
                    result = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, mask)
                    loss = self._compute_fusionv2_loss(
                        criterion, result, batch_x, batch_y, mask,
                        lam_reconstruct, time_weights)
                    train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                        i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                loss = loss / accumulation_steps
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    if (i + 1) % accumulation_steps == 0 or (i + 1) == train_steps:
                        scaler.step(model_optim)
                        scaler.update()
                else:
                    loss.backward()
                    if (i + 1) % accumulation_steps == 0 or (i + 1) == train_steps:
                        model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    def _compute_fusionv2_loss(self, criterion, result, batch_x, batch_y, mask,
                                time_weights):
        """
        Compute the FusionV2 loss.

        Loss = L_forecast + L_patch_reg

          1. Forecast loss (L1 with optional log-decay time weighting)
          2. Patch classification regularization (uniform usage)

        Args:
            criterion: L1Loss(reduction='none')
            result:    tuple (outputs, cls_pred, moe_prefix_w)
            batch_x:   [B, T, C]  input
            batch_y:   [B, T+pred_len, C]  target
            mask:      optional masking tensor
            time_weights: [pred_len] or None, log-decay weights
        """
        f_dim = -1 if self.args.features == 'MS' else 0
        pred_len = self.args.pred_len

        outputs, cls_pred, moe_prefix_w = result

        outputs = outputs[:, -pred_len:, f_dim:]
        batch_y = batch_y[:, -pred_len:, f_dim:].to(self.device)

        # ── 1. Forecast loss (with optional time-decay) ──
        per_step_loss = criterion(outputs, batch_y)  # [B, pred_len, C]
        if time_weights is not None:
            # time_weights: [pred_len] → [1, pred_len, 1]
            tw = time_weights.view(1, -1, 1).to(per_step_loss.device)
            forecast_loss = (per_step_loss * tw).mean()
        else:
            forecast_loss = per_step_loss.mean()

        # ── 2. Patch classification regularization ──
        patch_reg_loss = self._compute_patch_reg_loss(criterion)

        total_loss = forecast_loss + patch_reg_loss

        return total_loss

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(
                torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds, trues = [], []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs, cls_pred = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs, cls_pred = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :]
                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()
                if test_data.scale and self.args.inverse:
                    shape = batch_y.shape
                    if outputs.shape[-1] != batch_y.shape[-1]:
                        outputs = np.tile(outputs, [1, 1, int(batch_y.shape[-1] / outputs.shape[-1])])
                    outputs = test_data.inverse_transform(outputs.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    batch_y = test_data.inverse_transform(batch_y.reshape(shape[0] * shape[1], -1)).reshape(shape)
                outputs = outputs[:, :, f_dim:]
                batch_y = batch_y[:, :, f_dim:]
                preds.append(outputs)
                trues.append(batch_y)
                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    if test_data.scale and self.args.inverse:
                        shape = input.shape
                        input = test_data.inverse_transform(input.reshape(shape[0] * shape[1], -1)).reshape(shape)
                    gt = np.concatenate((input[0, :, -1], batch_y[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], outputs[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, _ = metric(preds, trues)
        print('mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{}'.format(mse, mae, rmse, mape, mspe))
        f = open(self.args.result_file, 'a')
        f.write(setting + "  \n")
        f.write('mse:{}, mae:{}, rmse:{}, mape:{}, mspe:{}'.format(mse, mae, rmse, mape, mspe))
        f.write('\n')
        f.write('\n')
        f.close()
        self.profile_model(test_loader)
        return

    def profile_model(self, test_loader):
        self.model.eval()
        with torch.no_grad():
            batch_x, batch_y, batch_x_mark, batch_y_mark = next(iter(test_loader))
            batch_x = batch_x.float().to(self.device)
            batch_y = batch_y.float().to(self.device)
            batch_x_mark = batch_x_mark.float().to(self.device)
            batch_y_mark = batch_y_mark.float().to(self.device)
            dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
            dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start_time = time.time()
            _ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            torch.cuda.synchronize()
            end_time = time.time()
            inference_time = end_time - start_time
            gpu_mem = torch.cuda.memory_allocated(self.device) / 1024 / 1024
            peak_mem = torch.cuda.max_memory_allocated(self.device) / 1024 / 1024
            total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print("=" * 80)
            print("Model Profiling Summary")
            print(f"{'Total Params':<25}: {total_params:,}")
            print(f"{'Inference Time (s)':<25}: {inference_time:.6f}")
            print(f"{'GPU Mem Footprint (MB)':<25}: {gpu_mem:.2f}")
            print(f"{'Peak Mem (MB)':<25}: {peak_mem:.2f}")
            print("=" * 80)

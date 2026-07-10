"""
Experiment handler for TimeMosaic Fusion V3.

Key differences from exp_Fusion.py (V1):
  - No MoE prompt load balancing loss (static prompts)
  - No MoE prefix load balancing loss (SoftGate auto-balances via learnable bias)
  - No patch ratio regularization (MoS Gate uses learnable bias)
  - Simplified loss: L1 forecast loss only
  - Optional log-decay time-decay loss weighting
"""

import torch
import torch.nn as nn
import numpy as np
import time
import os

from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


class Exp_FusionV3(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)
        self.device = self._acquire_device()

    def _build_model(self):
        model_dict = {
            'TimeMosaic_FusionV3': __import__('models.TimeMosaic_FusionV3',
                                              fromlist=['Model']).Model,
        }
        model = model_dict[self.args.model](self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        from data_provider.data_factory import data_provider
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)

    def _select_criterion(self):
        return nn.L1Loss()

    # ─── Optional log-decay time weights ────────────────────────────

    def _get_time_weights(self, pred_len, device):
        """Log-decay weights: higher weight for near-term predictions."""
        use_decay = getattr(self.args, 'use_log_decay', False)
        if not use_decay:
            return torch.ones(pred_len, device=device) / pred_len

        i_array = np.linspace(1 + 1e-5, pred_len - 1e-3, pred_len)
        weights = (1.0 / pred_len) * (np.log(pred_len) - np.log(i_array))
        weights = weights / weights.sum()  # normalize
        return torch.tensor(weights, dtype=torch.float32, device=device)

    # ─── Validation ────────────────────────────────────────────────

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs, _, _ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs, _ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs, _, _ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs, _ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)
                total_loss.append(loss.item())

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    # ─── Training ──────────────────────────────────────────────────

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
        criterion = self._select_criterion()
        time_weights = self._get_time_weights(self.args.pred_len, self.device)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        outputs, cls_pred, moe_prefix_weights = self.model(
                            batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs, cls_pred, moe_prefix_weights = self.model(
                        batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                # ── Forecast loss with optional time-decay weights ──
                forecast_loss = criterion(
                    outputs * time_weights.unsqueeze(0).unsqueeze(-1),
                    batch_y * time_weights.unsqueeze(0).unsqueeze(-1))

                total_loss = forecast_loss

                train_loss.append(total_loss.item())

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print(
                        f"\titers: {i + 1}, epoch: {epoch + 1} "
                        f"| loss: {total_loss.item():.7f} "
                        f"| speed: {speed:.4f}s/iter "
                        f"| left: {time.strftime('%H:%M:%S', time.gmtime(left_time))}"
                    )
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(total_loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    total_loss.backward()
                    model_optim.step()

            print(f"Epoch: {epoch + 1} cost time: {time.time() - epoch_time:.2f}s")
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print(
                f"Epoch: {epoch + 1}, Steps: {train_steps} "
                f"| Train Loss: {train_loss:.7f} "
                f"| Vali Loss: {vali_loss:.7f} "
                f"| Test Loss: {test_loss:.7f}"
            )

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        return self.model

    # ─── Testing ──────────────────────────────────────────────────

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print(f'loading pretrained model from {self.args.checkpoints}')
            ckpt_path = os.path.join(self.args.checkpoints, setting, 'checkpoint.pth')
            self.model.load_state_dict(torch.load(ckpt_path))

        preds = []
        trues = []
        folder_path = f'./test_results/{setting}/'
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
                        outputs, _ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs, _ = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                outputs = outputs.detach().cpu()
                batch_y = batch_y.detach().cpu()

                if test_data.scale and self.args.inverse:
                    outputs = test_data.inverse_transform(outputs.reshape(-1, outputs.shape[-1]))
                    batch_y = test_data.inverse_transform(batch_y.reshape(-1, batch_y.shape[-1]))
                    outputs = outputs.reshape(-1, self.args.pred_len, outputs.shape[-1])
                    batch_y = batch_y.reshape(-1, self.args.pred_len, batch_y.shape[-1])

                pred = outputs.numpy()
                true = batch_y.numpy()

                preds.append(pred)
                trues.append(true)

                if i % 20 == 0:
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, f'{i}.pdf'))

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])

        mae, mse, rmse, mape, mspe, wape = metric(preds, trues)
        print(f'mse:{mse}, mae:{mae}, rmse:{rmse}, mape:{mape}, mspe:{mspe}')

        # Save results
        result_path = 'result.txt'
        if hasattr(self.args, 'result_file') and self.args.result_file:
            result_path = self.args.result_file

        with open(result_path, 'a') as f:
            f.write(setting + "  \n")
            f.write(f'mse:{mse}, mae:{mae}, rmse:{rmse}, mape:{mape}, mspe:{mspe}')
            f.write('\n\n')

        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        return

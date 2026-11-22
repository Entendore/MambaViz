"""
Mamba Neural Network Studio V11
- Ensemble Learning (Mean + Variance Uncertainty)
- Synthetic Time-Series Data Generator
- Integrated Gradients for robust Interpretability
- Refactored Worker for multi-model training
"""

import sys
import os
import math
import tempfile
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox,
    QFileDialog, QTabWidget, QTextEdit, QProgressBar, QGroupBox,
    QFormLayout, QCheckBox, QMessageBox, QSplitter, QListWidget,
    QRadioButton, QGridLayout
)
from PySide6.QtGui import QFont

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


# =====================================================================
#  MAMBA MODEL IMPLEMENTATION
# =====================================================================

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return output * self.weight


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, bidirectional=False):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = int(expand * d_model)
        self.bidirectional = bidirectional
        self.dt_rank = math.ceil(self.d_model / 16)

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner, out_channels=self.d_inner,
            kernel_size=d_conv, padding=d_conv-1, groups=self.d_inner
        )
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = torch.arange(1, self.d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        
        self.last_hs = None  
        self.last_dt = None  

    def forward(self, x):
        B, L, _ = x.shape
        xz = self.in_proj(x)
        x_branch, z = xz.chunk(2, dim=-1)

        x_branch = x_branch.transpose(1, 2)
        x_branch = self.conv1d(x_branch)[:, :, :L]
        x_branch = x_branch.transpose(1, 2)
        x_branch = F.silu(x_branch)

        x_dbl = self.x_proj(x_branch)
        dt, B_p, C_p = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))
        A = -torch.exp(self.A_log)
        
        self.last_dt = dt.detach()

        if self.bidirectional:
            y_fwd, hs_fwd = self._selective_scan(x_branch, dt, A, B_p, C_p, B, L)
            x_bwd = torch.flip(x_branch, [1]); dt_bwd = torch.flip(dt, [1])
            B_bwd = torch.flip(B_p, [1]); C_bwd = torch.flip(C_p, [1])
            y_bwd, hs_bwd = self._selective_scan(x_bwd, dt_bwd, A, B_bwd, C_bwd, B, L)
            y_bwd = torch.flip(y_bwd, [1])
            y = y_fwd + y_bwd
            self.last_hs = (hs_fwd + hs_bwd) / 2
        else:
            y, self.last_hs = self._selective_scan(x_branch, dt, A, B_p, C_p, B, L)

        y = y * F.silu(z)
        return self.out_proj(y)

    def _selective_scan(self, x, dt, A, B, C, batch, length):
        d_inner = self.d_inner
        d_state = self.d_state
        h = torch.zeros(batch, d_inner, d_state, device=x.device, dtype=x.dtype)
        ys = []
        hs_track = []

        for t in range(length):
            xt = x[:, t, :]
            dtt = dt[:, t, :]
            Bt = B[:, t, :]
            Ct = C[:, t, :]
            dA = torch.exp(dtt.unsqueeze(-1) * A.unsqueeze(0))
            dB = dtt.unsqueeze(-1) * Bt.unsqueeze(1)
            h = dA * h + dB * xt.unsqueeze(-1)
            y_t = (h * Ct.unsqueeze(1)).sum(dim=-1)
            ys.append(y_t)
            hs_track.append(h.mean(dim=1)) 

        y = torch.stack(ys, dim=1)
        y = y + x * self.D.unsqueeze(0).unsqueeze(0)
        return y, torch.stack(hs_track, dim=1) 


class MambaModel(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=64, n_layers=2,
                 d_state=16, d_conv=4, expand=2, dropout=0.1, 
                 bidirectional=False, pooling='last', task='regression'):
        super().__init__()
        self.task = task
        self.pooling = pooling
        self.input_proj = nn.Linear(input_dim, d_model)
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand, bidirectional) for _ in range(n_layers)
        ])
        self.norms = nn.ModuleList([RMSNorm(d_model) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        
        if self.pooling == 'attn':
            self.attn_pool = nn.Linear(d_model, 1)
            self.last_attn = None
            
        self.output_proj = nn.Linear(d_model, output_dim)

    def forward(self, x):
        x = self.input_proj(x)
        for layer, norm in zip(self.layers, self.norms):
            x = x + self.dropout(layer(x))
            x = norm(x)
            
        if self.pooling == 'mean':
            x = x.mean(dim=1)
        elif self.pooling == 'max':
            x = x.max(dim=1)[0]
        elif self.pooling == 'attn':
            w = F.softmax(self.attn_pool(x).squeeze(-1), dim=1) 
            self.last_attn = w.detach()
            x = torch.bmm(x.transpose(1, 2), w.unsqueeze(2)).squeeze(2) 
        else: 
            x = x[:, -1, :]
            
        return self.output_proj(x)


# =====================================================================
#  LR FINDER WORKER
# =====================================================================
class LRFinderWorker(QThread):
    log = Signal(str)
    plot = Signal(np.ndarray, np.ndarray)
    finished_signal = Signal(float)
    error = Signal(str)

    def __init__(self, model_cfg, train_loader, cfg):
        super().__init__()
        self.model_cfg = model_cfg
        self.train_loader = train_loader
        self.cfg = cfg
        self._running = True

    def run(self):
        try:
            device = self.cfg['device']
            model = MambaModel(**self.model_cfg).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-7)
            
            lrs, losses = [], []
            start_lr, end_lr = 1e-7, 1.0
            num_batches = min(100, len(self.train_loader))
            lr_factor = (end_lr / start_lr) ** (1.0 / num_batches)
            
            model.train()
            for i, (xb, yb) in enumerate(self.train_loader):
                if not self._running or i >= num_batches: break
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                pred = model(xb)
                
                if self.cfg['task'] == 'classification' and yb.dim() > 1: yb = yb.argmax(-1)
                    
                loss = nn.CrossEntropyLoss()(pred, yb) if self.cfg['task'] == 'classification' else nn.MSELoss()(pred, yb)
                loss.backward()
                optimizer.step()
                
                lrs.append(optimizer.param_groups[0]['lr'])
                losses.append(loss.item())
                optimizer.param_groups[0]['lr'] *= lr_factor
                
            self.plot.emit(np.array(lrs), np.array(losses))
            
            if len(losses) > 10:
                log_losses = np.log10(losses + 1e-8)
                gradients = np.gradient(log_losses)
                min_grad_idx = np.argmin(gradients[5:-5]) + 5
                best_lr = lrs[min_grad_idx] / 10.0 
                self.finished_signal.emit(best_lr)
            else:
                self.finished_signal.emit(0.001)
        except Exception as e:
            self.error.emit(str(e))

    def stop(self): self._running = False


# =====================================================================
#  TRAINING WORKER THREAD (V11 - Ensemble Loop)
# =====================================================================

class TrainingWorker(QThread):
    epoch_finished = Signal(int, float, float, float) 
    eval_plot = Signal(np.ndarray, np.ndarray)
    heatmap_plot = Signal(np.ndarray, np.ndarray) 
    log = Signal(str)
    finished_signal = Signal(dict, list) # history, list_of_state_dicts
    error = Signal(str)

    def __init__(self, model_cfg, train_loader, val_loader, cfg, target_norm, ema_decay=0.0):
        super().__init__()
        self.model_cfg = model_cfg
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.cfg = cfg
        self.target_norm = target_norm
        self.ema_decay = ema_decay
        self._running = True
        self.best_model_path = os.path.join(tempfile.gettempdir(), "mamba_best_v11.pt")
        self.ema_state = None

    def _update_ema(self, model):
        if self.ema_state is None:
            self.ema_state = {k: v.clone().detach() for k, v in model.state_dict().items()}
        else:
            for k, v in model.state_dict().items():
                if v.dtype.is_floating_point:
                    self.ema_state[k].mul_(self.ema_decay).add_(v.detach(), alpha=1 - self.ema_decay)

    def _apply_ema(self, model):
        self._orig_state = {k: v.clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.ema_state)

    def _restore_ema(self, model):
        model.load_state_dict(self._orig_state)

    def run(self):
        try:
            device = self.cfg.get('device', 'cpu')
            use_amp = (device == 'cuda')
            scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
            ensemble_states = []

            def quantile_loss(preds, target, quantiles=[0.1, 0.5, 0.9]):
                target = target.unsqueeze(-1).expand(-1, -1, len(quantiles))
                preds = preds.view(target.size(0), target.size(1), len(quantiles))
                errors = target - preds
                loss = 0
                for i, q in enumerate(quantiles):
                    loss += torch.max((q-1) * errors[:,:,i], q * errors[:,:,i]).mean()
                return loss / len(quantiles)

            if self.cfg['task'] == 'classification': criterion = nn.CrossEntropyLoss()
            elif self.cfg['loss_fn'] == 'huber': criterion = nn.SmoothL1Loss()
            elif self.cfg['loss_fn'] == 'quantile': criterion = quantile_loss
            else: criterion = nn.MSELoss()

            history = {'train_loss': [], 'val_loss': [], 'grad_norm': []}
            best_val = float('inf')
            accum_steps = self.cfg['accum_steps']

            for ens_i in range(self.cfg['ensemble_size']):
                if not self._running: break
                self.log.emit(f"\n🧠 Training Ensemble Model {ens_i+1}/{self.cfg['ensemble_size']}...")
                
                model = MambaModel(**self.model_cfg).to(device)
                self.ema_state = None # Reset EMA for new model
                optimizer = torch.optim.AdamW(model.parameters(), lr=self.cfg['lr'], weight_decay=self.cfg['weight_decay'])
                
                scheduler = None
                if self.cfg['scheduler'] == 'cosine':
                    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cfg['epochs'], eta_min=1e-6)
                elif self.cfg['scheduler'] == 'onecycle':
                    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.cfg['lr'], steps_per_epoch=len(self.train_loader), epochs=self.cfg['epochs'])

                best_val_ens = float('inf')
                patience_counter = 0

                for epoch in range(self.cfg['epochs']):
                    if not self._running: break

                    model.train()
                    running, n, total_grad_norm = 0.0, 0, 0.0
                    optimizer.zero_grad()
                    
                    for i, (xb, yb) in enumerate(self.train_loader):
                        xb, yb = xb.to(device), yb.to(device)
                        if self.cfg.get('aug_jitter'): xb = xb + torch.randn_like(xb) * 0.01
                        if self.cfg.get('aug_scale'):
                            scale_factor = 1.0 + torch.randn(xb.size(0), 1, xb.size(2), device=device) * 0.1
                            xb = xb * scale_factor
                        
                        with torch.cuda.amp.autocast(enabled=use_amp):
                            pred = model(xb)
                            if self.cfg['task'] == 'classification' and yb.dim() > 1: yb = yb.argmax(-1)
                            loss = criterion(pred, yb)
                        
                        scaler.scale(loss / accum_steps).backward()
                        
                        if (i + 1) % accum_steps == 0 or (i + 1) == len(self.train_loader):
                            scaler.unscale_(optimizer)
                            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                            scaler.step(optimizer)
                            scaler.update()
                            optimizer.zero_grad()
                            if self.ema_decay > 0: self._update_ema(model)
                            total_grad_norm += grad_norm.item()
                            n += 1
                            if scheduler and self.cfg['scheduler'] == 'onecycle': scheduler.step()

                        running += loss.item()

                    train_loss = running / max(1, len(self.train_loader))
                    avg_grad_norm = total_grad_norm / max(1, n)
                    if scheduler and self.cfg['scheduler'] == 'cosine': scheduler.step()

                    if self.ema_decay > 0: self._apply_ema(model)
                    model.eval()
                    
                    v_running, v_n = 0.0, 0
                    all_preds, all_acts = [], []
                    last_hs, last_dt = None, None
                    
                    with torch.no_grad():
                        for xb, yb in self.val_loader:
                            xb, yb = xb.to(device), yb.to(device)
                            with torch.cuda.amp.autocast(enabled=use_amp):
                                pred = model(xb)
                            if self.cfg['task'] == 'classification':
                                if yb.dim() > 1: yb = yb.argmax(-1)
                                p = pred.argmax(dim=-1); yb_viz = yb
                            else:
                                if self.cfg['loss_fn'] == 'quantile': p = pred.view(pred.size(0), -1, 3)[:, 0, 1]
                                else: p = pred.squeeze(-1) if pred.shape[-1] == 1 else pred[:, :, 0]
                                yb_viz = yb if yb.dim() == 1 else yb[:, 0]
                                if self.target_norm:
                                    p = p * self.target_norm[1] + self.target_norm[0]
                                    yb_viz = yb_viz * self.target_norm[1] + self.target_norm[0]
                                    
                            loss = criterion(pred, yb)
                            v_running += loss.item(); v_n += 1
                            all_preds.append(p.cpu().numpy()); all_acts.append(yb_viz.cpu().numpy())
                            if last_hs is None:
                                last_hs = model.layers[-1].last_hs[0].cpu().numpy()
                                last_dt = model.layers[-1].last_dt[0].mean(dim=-1).cpu().numpy()

                    val_loss = v_running / max(1, v_n)
                    preds_np = np.concatenate(all_preds); acts_np = np.concatenate(all_acts)

                    history['train_loss'].append(train_loss)
                    history['val_loss'].append(val_loss)
                    history['grad_norm'].append(avg_grad_norm)
                    
                    self.epoch_finished.emit(epoch + 1, train_loss, val_loss, avg_grad_norm)
                    self.eval_plot.emit(acts_np, preds_np)
                    if last_hs is not None: self.heatmap_plot.emit(last_hs, last_dt)
                    
                    self.log.emit(f"Ep {epoch+1:3d}/{self.cfg['epochs']} | Tr: {train_loss:.4f} | Va: {val_loss:.4f} | Grd: {avg_grad_norm:.2f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

                    if val_loss < best_val_ens:
                        best_val_ens = val_loss
                        save_state = self.ema_state if self.ema_decay > 0 else model.state_dict()
                        torch.save(save_state, self.best_model_path)
                        patience_counter = 0
                    else:
                        patience_counter += 1
                        if self.cfg['patience'] > 0 and patience_counter >= self.cfg['patience']:
                            self.log.emit(f"⏹️ Early stopping triggered (patience={self.cfg['patience']})")
                            break

                    if self.ema_decay > 0: self._restore_ema(model)

                if os.path.exists(self.best_model_path):
                    ensemble_states.append(torch.load(self.best_model_path))
                    if best_val_ens < best_val: best_val = best_val_ens

            self.finished_signal.emit(history, ensemble_states)
        except Exception as e:
            self.error.emit(f"Training error: {str(e)}")

    def stop(self): self._running = False


# =====================================================================
#  MATPLOTLIB CANVAS
# =====================================================================
class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=6, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)
        self.setParent(parent)
        self.fig.tight_layout()


# =====================================================================
#  MAIN WINDOW
# =====================================================================
STYLE = """
QMainWindow { background: #1e1e2e; }
QWidget { color: #cdd6f4; font-size: 13px; }
QTabWidget::pane { border: 1px solid #45475a; background: #181825; border-radius: 6px; }
QTabBar::tab { background: #181825; color: #a6adc8; padding: 8px 16px; border-radius: 4px; }
QTabBar::tab:selected { background: #313244; color: #cdd6f4; }
QPushButton { background: #313244; color: #cdd6f4; border: 1px solid #45475a; padding: 6px 14px; border-radius: 5px; }
QPushButton:hover { background: #45475a; }
QPushButton:pressed { background: #585b70; }
QPushButton#primary { background: #89b4fa; color: #11111b; font-weight: bold; }
QPushButton#primary:hover { background: #74c7ec; }
QPushButton#danger { background: #f38ba8; color: #11111b; }
QPushButton#success { background: #a6e3a1; color: #11111b; font-weight: bold; }
QGroupBox { border: 1px solid #45475a; border-radius: 6px; margin-top: 12px; padding-top: 8px; font-weight: bold; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; }
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox { background: #313244; border: 1px solid #45475a; padding: 4px; border-radius: 4px; color: #cdd6f4; }
QTextEdit { background: #11111b; border: 1px solid #45475a; border-radius: 4px; }
QProgressBar { background: #313244; border: 1px solid #45475a; border-radius: 4px; text-align: center; }
QProgressBar::chunk { background: #89b4fa; border-radius: 4px; }
QListWidget { background: #181825; border: 1px solid #45475a; border-radius: 4px; }
QLabel#header { font-size: 20px; font-weight: bold; color: #89b4fa; }
QLabel#subheader { font-size: 13px; color: #a6adc8; }
"""

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mamba Neural Network Studio V11")
        self.resize(1600, 1000)

        self.df = None
        self.test_df = None
        self.model = None
        self.worker = None
        self.lr_finder = None
        self.train_loader = None
        self.val_loader = None
        self.history = {'train_loss': [], 'val_loss': [], 'grad_norm': []}
        self.normalization = None
        self.target_norm = None
        self.feature_cols = []
        self.target_col = None
        self.n_classes = 0
        self.horizon = 1
        self.datetime_index = None
        self.ensemble_states = []

        self._build_ui()
        self.setStyleSheet(STYLE)

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)

        header = QLabel("🧠 Mamba Neural Network Studio V11")
        header.setObjectName("header")
        sub = QLabel("Ensemble Uncertainty, Synthetic Generator & Integrated Gradients")
        sub.setObjectName("subheader")
        main_layout.addWidget(header)
        main_layout.addWidget(sub)

        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs, 1)

        self._build_dataset_tab()
        self._build_model_tab()
        self._build_training_tab()
        self._build_results_tab()

        self.status = self.statusBar()
        self.status.showMessage("Ready")

    def _build_dataset_tab(self):
        tab = QWidget()
        layout = QHBoxLayout(tab)
        left = QVBoxLayout()
        g_file = QGroupBox("Dataset")
        fl = QVBoxLayout(g_file)
        row = QHBoxLayout()
        self.path_edit = QLineEdit()
        self.path_edit.setPlaceholderText("Select Training CSV file...")
        btn_browse = QPushButton("Browse")
        btn_browse.clicked.connect(self._browse_file)
        row.addWidget(self.path_edit, 1)
        row.addWidget(btn_browse)
        fl.addLayout(row)

        btn_row_load = QHBoxLayout()
        btn_load = QPushButton("Load Dataset")
        btn_load.setObjectName("primary")
        btn_load.clicked.connect(self._load_dataset)
        btn_synth = QPushButton(" Generate Synthetic Data")
        btn_synth.setObjectName("success")
        btn_synth.clicked.connect(self._generate_synthetic_data)
        btn_row_load.addWidget(btn_load)
        btn_row_load.addWidget(btn_synth)
        fl.addLayout(btn_row_load)
        left.addWidget(g_file)

        g_dt = QGroupBox("Datetime & Resampling")
        dt_form = QFormLayout(g_dt)
        self.cb_datetime = QComboBox()
        self.cb_datetime.addItem("None")
        self.le_resample = QLineEdit()
        self.le_resample.setPlaceholderText("e.g., 1H, 1D, 1W (Leave blank to skip)")
        dt_form.addRow("Datetime Column:", self.cb_datetime)
        dt_form.addRow("Resample Freq:", self.le_resample)
        left.addWidget(g_dt)

        g_task = QGroupBox("Task Type & Preprocessing")
        tl = QVBoxLayout(g_task)
        self.rb_reg = QRadioButton("Regression (predict numeric value)")
        self.rb_clf = QRadioButton("Classification (predict category)")
        self.rb_reg.setChecked(True)
        tl.addWidget(self.rb_reg); tl.addWidget(self.rb_clf)
        nan_form = QFormLayout()
        self.cb_nan = QComboBox()
        self.cb_nan.addItems(["Fill with 0.0", "Forward Fill", "Mean Imputation"])
        nan_form.addRow("Handle Missing Values:", self.cb_nan)
        tl.addLayout(nan_form)
        left.addWidget(g_task)

        g_seq = QGroupBox("Sequence Configuration")
        sf = QFormLayout(g_seq)
        self.sp_seq_len = QSpinBox(); self.sp_seq_len.setRange(2, 512); self.sp_seq_len.setValue(16)
        self.sp_horizon = QSpinBox(); self.sp_horizon.setRange(1, 64); self.sp_horizon.setValue(1)
        self.sp_stride = QSpinBox(); self.sp_stride.setRange(1, 512); self.sp_stride.setValue(1)
        sf.addRow("Sequence Length (L):", self.sp_seq_len)
        sf.addRow("Forecast Horizon (H):", self.sp_horizon)
        sf.addRow("Sequence Stride:", self.sp_stride)
        left.addWidget(g_seq)

        g_feat = QGroupBox("Columns")
        cf = QVBoxLayout(g_feat)
        cf.addWidget(QLabel("Features (input columns):"))
        self.lw_features = QListWidget()
        self.lw_features.setSelectionMode(QListWidget.MultiSelection)
        cf.addWidget(self.lw_features)
        cf.addWidget(QLabel("Target (column to predict):"))
        self.cb_target = QComboBox()
        cf.addWidget(self.cb_target)
        btn_row = QHBoxLayout()
        btn_select_all = QPushButton("Select All Features")
        btn_select_all.clicked.connect(self._select_all_features)
        btn_viz_split = QPushButton("Visualize Train/Val Split")
        btn_viz_split.clicked.connect(self._plot_train_val_split)
        btn_row.addWidget(btn_select_all)
        btn_row.addWidget(btn_viz_split)
        cf.addLayout(btn_row)
        left.addWidget(g_feat)
        left.addStretch(1)
        layout.addLayout(left, 1)

        right = QVBoxLayout()
        g_preview = QGroupBox("Data Preview")
        pl = QVBoxLayout(g_preview)
        self.preview_text = QTextEdit()
        self.preview_text.setReadOnly(True)
        pl.addWidget(self.preview_text)
        right.addWidget(g_preview)

        viz_split = QSplitter(Qt.Vertical)
        g_corr = QGroupBox("Feature Correlation Heatmap")
        cl = QVBoxLayout(g_corr)
        self.corr_canvas = MplCanvas(self, width=8, height=3)
        cl.addWidget(self.corr_canvas)
        viz_split.addWidget(g_corr)
        g_dist = QGroupBox("Target Variable Distribution")
        dl = QVBoxLayout(g_dist)
        self.dist_canvas = MplCanvas(self, width=8, height=3)
        dl.addWidget(self.dist_canvas)
        viz_split.addWidget(g_dist)
        right.addWidget(viz_split, 2)
        layout.addLayout(right, 2)
        self.tabs.addTab(tab, "📊 Dataset")

    def _build_model_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        top_row = QHBoxLayout()
        form_box = QGroupBox("Mamba Architecture")
        form = QFormLayout(form_box)
        self.cb_preset = QComboBox()
        self.cb_preset.addItems(["Custom", "Tiny (CPU)", "Small", "Standard", "Heavy"])
        self.cb_preset.currentIndexChanged.connect(self._apply_preset)
        self.sp_d_model = QSpinBox(); self.sp_d_model.setRange(8, 1024); self.sp_d_model.setValue(64)
        self.sp_n_layers = QSpinBox(); self.sp_n_layers.setRange(1, 32); self.sp_n_layers.setValue(2)
        self.sp_d_state = QSpinBox(); self.sp_d_state.setRange(4, 256); self.sp_d_state.setValue(16)
        self.sp_d_conv = QSpinBox(); self.sp_d_conv.setRange(2, 32); self.sp_d_conv.setValue(4)
        self.sp_expand = QSpinBox(); self.sp_expand.setRange(1, 8); self.sp_expand.setValue(2)
        self.sp_dropout = QDoubleSpinBox(); self.sp_dropout.setRange(0.0, 0.9); self.sp_dropout.setSingleStep(0.05); self.sp_dropout.setValue(0.1)
        self.chk_bidir = QCheckBox("Enable Bi-Mamba (Bidirectional)")
        self.cb_pooling = QComboBox()
        self.cb_pooling.addItems(["last", "mean", "max", "attn"])
        form.addRow("Quick Presets:", self.cb_preset)
        form.addRow("d_model (hidden size):", self.sp_d_model)
        form.addRow("Number of Mamba layers:", self.sp_n_layers)
        form.addRow("d_state (SSM state size):", self.sp_d_state)
        form.addRow("d_conv (conv kernel):", self.sp_d_conv)
        form.addRow("expand factor:", self.sp_expand)
        form.addRow("Dropout:", self.sp_dropout)
        form.addRow("Sequence Pooling:", self.cb_pooling)
        form.addRow("", self.chk_bidir)
        top_row.addWidget(form_box)

        io_box = QGroupBox("Model Persistence & Info")
        io_layout = QVBoxLayout(io_box)
        btn_summary = QPushButton("ℹ️ Print Model Summary")
        btn_summary.clicked.connect(self._print_model_summary)
        btn_viz_a = QPushButton("🧬 Visualize A-Matrix (Decay)")
        btn_viz_a.clicked.connect(self._visualize_a_matrix)
        btn_save = QPushButton("💾 Save Model")
        btn_save.setObjectName("success")
        btn_save.clicked.connect(self._save_model)
        btn_load = QPushButton("📂 Load Model")
        btn_load.clicked.connect(self._load_model)
        io_layout.addWidget(btn_summary)
        io_layout.addWidget(btn_viz_a)
        io_layout.addWidget(btn_save)
        io_layout.addWidget(btn_load)
        io_layout.addStretch(1)
        top_row.addWidget(io_box)
        layout.addLayout(top_row)

        info_box = QGroupBox("About V11 Features")
        il = QVBoxLayout(info_box)
        info = QLabel(
            "1. Ensemble Learning: Train N models and plot mean prediction + variance uncertainty.\n"
            "2. Synthetic Generator: Instantly create complex seasonal time-series for testing.\n"
            "3. Integrated Gradients: Robust interpretability via interpolating gradients from baseline to input."
        )
        info.setWordWrap(True)
        il.addWidget(info)
        layout.addWidget(info_box)
        layout.addStretch(1)
        self.tabs.addTab(tab, "🧩 Model")

    def _build_training_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        cfg_box = QGroupBox("Training Configuration")
        form = QFormLayout(cfg_box)
        self.sp_epochs = QSpinBox(); self.sp_epochs.setRange(1, 10000); self.sp_epochs.setValue(50)
        self.sp_batch = QSpinBox(); self.sp_batch.setRange(1, 4096); self.sp_batch.setValue(32)
        self.sp_accum = QSpinBox(); self.sp_accum.setRange(1, 100); self.sp_accum.setValue(1)
        self.sp_ensemble = QSpinBox(); self.sp_ensemble.setRange(1, 10); self.sp_ensemble.setValue(1)
        self.sp_ensemble.setToolTip("Number of independent models to train. >1 enables Ensemble Uncertainty.")
        self.sp_lr = QDoubleSpinBox(); self.sp_lr.setRange(0.000001, 1.0); self.sp_lr.setDecimals(6); self.sp_lr.setValue(0.001)
        self.sp_wd = QDoubleSpinBox(); self.sp_wd.setRange(0.0, 1.0); self.sp_wd.setDecimals(6); self.sp_wd.setValue(0.01)
        self.sp_val_split = QDoubleSpinBox(); self.sp_val_split.setRange(0.0, 0.9); self.sp_val_split.setValue(0.2)
        self.cb_device = QComboBox()
        self.cb_device.addItems(["Auto", "CPU", "CUDA", "MPS (Apple)"])
        self.sp_patience = QSpinBox(); self.sp_patience.setRange(0, 100); self.sp_patience.setValue(5)
        self.cb_scheduler = QComboBox()
        self.cb_scheduler.addItems(["None", "Cosine Annealing", "OneCycleLR"])
        self.cb_loss = QComboBox()
        self.cb_loss.addItems(["MSE", "SmoothL1 (Huber)", "Quantile"])
        self.chk_normalize = QCheckBox("Normalize inputs (z-score)"); self.chk_normalize.setChecked(True)
        self.chk_tnorm = QCheckBox("Normalize Target (Regression only)"); self.chk_tnorm.setChecked(True)
        self.chk_chrono = QCheckBox("Chronological Split (No Shuffle)"); self.chk_chrono.setChecked(True)
        self.chk_compile = QCheckBox("Use torch.compile (PyTorch 2.0+)")
        self.chk_ema = QCheckBox("Use EMA Weights (Decay=0.999)")
        form.addRow("Epochs:", self.sp_epochs)
        form.addRow("Batch size:", self.sp_batch)
        form.addRow("Grad Accum Steps:", self.sp_accum)
        form.addRow("Ensemble Size:", self.sp_ensemble)
        form.addRow("Learning rate:", self.sp_lr)
        form.addRow("Weight decay:", self.sp_wd)
        form.addRow("Validation split:", self.sp_val_split)
        form.addRow("Device:", self.cb_device)
        form.addRow("Scheduler:", self.cb_scheduler)
        form.addRow("Loss (Reg only):", self.cb_loss)
        form.addRow("Early Stopping Patience:", self.sp_patience)
        form.addRow("", self.chk_normalize)
        form.addRow("", self.chk_tnorm)
        form.addRow("", self.chk_chrono)
        form.addRow("", self.chk_compile)
        form.addRow("", self.chk_ema)
        layout.addWidget(cfg_box)

        aug_box = QGroupBox("Time-Series Augmentation")
        aug_l = QHBoxLayout(aug_box)
        self.chk_jitter = QCheckBox("Jitter (Gaussian Noise)")
        self.chk_scale = QCheckBox("Scaling (Multiplicative)")
        aug_l.addWidget(self.chk_jitter)
        aug_l.addWidget(self.chk_scale)
        aug_l.addStretch(1)
        layout.addWidget(aug_box)

        btn_row = QHBoxLayout()
        self.btn_train = QPushButton("▶  Train Model"); self.btn_train.setObjectName("primary")
        self.btn_train.clicked.connect(self._start_training)
        self.btn_stop = QPushButton("■  Stop"); self.btn_stop.setObjectName("danger")
        self.btn_stop.setEnabled(False); self.btn_stop.clicked.connect(self._stop_training)
        self.btn_lr_find = QPushButton("🔍 Find Learning Rate")
        self.btn_lr_find.clicked.connect(self._run_lr_finder)
        self.btn_export_hist = QPushButton("📝 Export History CSV")
        self.btn_export_hist.clicked.connect(self._export_history)
        btn_row.addWidget(self.btn_train)
        btn_row.addWidget(self.btn_stop)
        btn_row.addWidget(self.btn_lr_find)
        btn_row.addWidget(self.btn_export_hist)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        self.progress = QProgressBar()
        layout.addWidget(self.progress)

        grid = QGridLayout()
        self.loss_canvas = MplCanvas(self, width=5, height=3)
        self.grad_canvas = MplCanvas(self, width=5, height=3)
        self.live_pred_canvas = MplCanvas(self, width=5, height=3)
        self.heatmap_canvas = MplCanvas(self, width=5, height=3)
        self.dt_canvas = MplCanvas(self, width=5, height=3)
        grid.addWidget(QLabel("Loss Curves"), 0, 0)
        grid.addWidget(QLabel("Gradient Norm"), 0, 1)
        grid.addWidget(QLabel("Delta ($\Delta$) Selectivity"), 0, 2)
        grid.addWidget(self.loss_canvas, 1, 0)
        grid.addWidget(self.grad_canvas, 1, 1)
        grid.addWidget(self.dt_canvas, 1, 2)
        grid.addWidget(QLabel("Live Predictions"), 2, 0)
        grid.addWidget(QLabel("SSM Hidden State (Memory)"), 2, 1)
        grid.addWidget(self.live_pred_canvas, 3, 0)
        grid.addWidget(self.heatmap_canvas, 3, 1)
        layout.addLayout(grid, 2)

        log_box = QGroupBox("Training Log")
        lol = QVBoxLayout(log_box)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        lol.addWidget(self.log_text)
        layout.addWidget(log_box, 1)
        self.tabs.addTab(tab, "🚀 Training")

    def _build_results_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        infer_box = QGroupBox("Inference & Interpretability")
        il = QHBoxLayout(infer_box)
        self.chk_mc_dropout = QCheckBox("Enable MC Dropout (Uncertainty)")
        btn_imp = QPushButton("📊 Calculate Feature Importance (Occlusion)")
        btn_imp.setObjectName("success")
        btn_imp.clicked.connect(self._run_feature_importance)
        btn_ig = QPushButton("🔍 Generate Integrated Gradients")
        btn_ig.setObjectName("success")
        btn_ig.clicked.connect(self._generate_integrated_gradients)
        btn_attn = QPushButton("🧠 Plot Attention Weights")
        btn_attn.setObjectName("success")
        btn_attn.clicked.connect(self._plot_attention_weights)
        il.addWidget(self.chk_mc_dropout)
        il.addWidget(btn_imp)
        il.addWidget(btn_ig)
        il.addWidget(btn_attn)
        layout.addWidget(infer_box)

        test_box = QGroupBox("Test Set Inference & Export")
        tl = QHBoxLayout(test_box)
        self.test_path = QLineEdit()
        self.test_path.setPlaceholderText("Optional: Select Test CSV...")
        btn_test_browse = QPushButton("Browse")
        btn_test_browse.clicked.connect(self._browse_test_file)
        self.btn_test_infer = QPushButton("Run Inference & Export CSV")
        self.btn_test_infer.setObjectName("success")
        self.btn_test_infer.clicked.connect(self._run_test_inference)
        self.btn_test_infer.setEnabled(False)
        tl.addWidget(QLabel("Test Data:"))
        tl.addWidget(self.test_path, 1)
        tl.addWidget(btn_test_browse)
        tl.addWidget(self.btn_test_infer)
        layout.addWidget(test_box)

        splitter = QSplitter(Qt.Vertical)
        g_pred = QGroupBox("Validation Predictions vs Actual")
        pl = QVBoxLayout(g_pred)
        self.pred_canvas = MplCanvas(self, width=10, height=3)
        pl.addWidget(self.pred_canvas)
        splitter.addWidget(g_pred)
        g_res = QGroupBox("Residuals / Confusion Matrix / Saliency")
        rl = QVBoxLayout(g_res)
        self.res_canvas = MplCanvas(self, width=10, height=2)
        rl.addWidget(self.res_canvas)
        splitter.addWidget(g_res)
        g_metrics = QGroupBox("Evaluation Metrics")
        ml = QVBoxLayout(g_metrics)
        self.metrics_text = QTextEdit()
        self.metrics_text.setReadOnly(True)
        ml.addWidget(self.metrics_text)
        splitter.addWidget(g_metrics)
        splitter.setSizes([300, 200, 200])
        layout.addWidget(splitter, 1)
        self.tabs.addTab(tab, "📈 Results")

    # -------- DATASET OPS --------
    def _browse_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open CSV", "", "CSV files (*.csv);;All files (*)")
        if path: self.path_edit.setText(path)

    def _browse_test_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Test CSV", "", "CSV files (*.csv);;All files (*)")
        if path: self.test_path.setText(path)

    def _generate_synthetic_data(self):
        dates = pd.date_range(start='2020-01-01', periods=1000, freq='D')
        t = np.arange(1000)
        trend = t * 0.05
        seasonality = 10 * np.sin(t * 2 * np.pi / 30) + 5 * np.sin(t * 2 * np.pi / 365)
        noise = np.random.normal(0, 2, 1000)
        target = trend + seasonality + noise
        
        df = pd.DataFrame({
            'Date': dates, 'Trend': trend, 
            'Seasonality_30': 10 * np.sin(t * 2 * np.pi / 30), 
            'Seasonality_365': 5 * np.sin(t * 2 * np.pi / 365), 
            'Noise': noise, 'Target': target
        })
        temp_path = os.path.join(tempfile.gettempdir(), "synthetic_mamba_data.csv")
        df.to_csv(temp_path, index=False)
        self.path_edit.setText(temp_path)
        self._load_dataset()
        self.cb_datetime.setCurrentText("Date")
        self.le_resample.setText("1D")
        self._select_all_features()
        self.cb_target.setCurrentText("Target")

    def _impute_missing(self, df):
        strategy = self.cb_nan.currentText()
        if strategy == "Forward Fill": return df.ffill().bfill()
        elif strategy == "Mean Imputation": return df.fillna(df.mean(numeric_only=True))
        else: return df.fillna(0.0)

    def _load_dataset(self):
        path = self.path_edit.text().strip()
        if not path or not os.path.exists(path):
            QMessageBox.warning(self, "Error", "Please select a valid CSV file.")
            return
        try: self.df = pd.read_csv(path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to read CSV:\n{e}"); return

        self.preview_text.setPlainText(self.df.head(20).to_string())
        self.cb_datetime.clear(); self.cb_datetime.addItem("None")
        self.lw_features.clear(); self.cb_target.clear()
        for col in self.df.columns:
            self.cb_datetime.addItem(col)
            self.lw_features.addItem(col)
            self.cb_target.addItem(col)
        if self.df.shape[1] >= 2: self.cb_target.setCurrentIndex(self.df.shape[1] - 1)
        self._plot_correlation(); self._plot_distribution()
        self.status.showMessage(f"Loaded {self.df.shape[0]} rows × {self.df.shape[1]} cols")

    def _process_datetime(self, df):
        dt_col = self.cb_datetime.currentText()
        if dt_col != "None" and dt_col in df.columns:
            df[dt_col] = pd.to_datetime(df[dt_col], errors='coerce')
            df = df.set_index(dt_col).sort_index()
            resample_freq = self.le_resample.text().strip()
            if resample_freq: df = df.resample(resample_freq).mean()
            self.datetime_index = df.index
        else: self.datetime_index = None
        return df

    def _plot_correlation(self):
        if self.df is None: return
        ax = self.corr_canvas.ax; ax.clear()
        numeric_df = self.df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] > 0:
            corr = numeric_df.corr()
            cax = ax.imshow(corr.values, cmap='coolwarm', vmin=-1, vmax=1)
            ax.set_xticks(range(len(corr.columns)))
            ax.set_yticks(range(len(corr.columns)))
            ax.set_xticklabels(corr.columns, rotation=45, ha='right')
            ax.set_yticklabels(corr.columns)
            if hasattr(self, '_cbar_corr'): self._cbar_corr.remove()
            self._cbar_corr = self.corr_canvas.fig.colorbar(cax, ax=ax)
        ax.set_title("Feature Correlation Matrix")
        self.corr_canvas.fig.tight_layout(); self.corr_canvas.draw()

    def _plot_distribution(self):
        if self.df is None: return
        ax = self.dist_canvas.ax; ax.clear()
        target = self.cb_target.currentText()
        if target and target in self.df:
            data = self.df[target].dropna()
            if pd.api.types.is_numeric_dtype(data):
                ax.hist(data, bins=30, color='#89b4fa', edgecolor='#11111b')
            else:
                vals = data.value_counts()
                ax.bar(vals.index.astype(str), vals.values, color='#89b4fa')
            ax.set_title(f"Distribution of {target}"); ax.set_ylabel("Count")
        self.dist_canvas.fig.tight_layout(); self.dist_canvas.draw()

    def _plot_train_val_split(self):
        if self.df is None: return
        df = self._process_datetime(self.df.copy())
        target = self.cb_target.currentText()
        if target not in df: return
        y_raw = df[target].astype(np.float32).values
        seq_len = self.sp_seq_len.value(); horizon = self.sp_horizon.value()
        n = len(df)
        if n <= seq_len + horizon:
            QMessageBox.warning(self, "Error", "Dataset too small for sequence config."); return
        n_seq = n - seq_len - horizon + 1
        val_split = self.sp_val_split.value()
        n_val = max(1, int(n_seq * val_split))
        n_train = n_seq - n_val
        train_end_idx = n_train + seq_len + horizon - 1
        
        ax = self.dist_canvas.ax; ax.clear()
        x = self.datetime_index[:train_end_idx] if self.datetime_index else range(train_end_idx)
        ax.plot(x, y_raw[:train_end_idx], color='#89b4fa', label='Train')
        x_val = self.datetime_index[train_end_idx-1:] if self.datetime_index else range(train_end_idx-1, n)
        ax.plot(x_val, y_raw[train_end_idx-1:], color='#f38ba8', label='Validation')
        ax.set_title("Train/Validation Chronological Split")
        ax.legend()
        if self.datetime_index: ax.tick_params(axis='x', rotation=30)
        self.dist_canvas.fig.tight_layout(); self.dist_canvas.draw()

    def _select_all_features(self):
        for i in range(self.lw_features.count()): self.lw_features.item(i).setSelected(True)

    def _apply_preset(self, idx):
        presets = {0: None, 1: {"d_model": 32, "n_layers": 1, "d_state": 8, "expand": 1},
                   2: {"d_model": 64, "n_layers": 2, "d_state": 16, "expand": 2},
                   3: {"d_model": 128, "n_layers": 4, "d_state": 32, "expand": 2},
                   4: {"d_model": 256, "n_layers": 6, "d_state": 64, "expand": 4}}
        cfg = presets.get(idx)
        if cfg:
            self.sp_d_model.setValue(cfg["d_model"]); self.sp_n_layers.setValue(cfg["n_layers"])
            self.sp_d_state.setValue(cfg["d_state"]); self.sp_expand.setValue(cfg["expand"])

    def _get_selected_features(self): return [item.text() for item in self.lw_features.selectedItems()]

    def _prepare_data(self, df_override=None):
        is_test = df_override is not None
        df_src = df_override if is_test else self.df
        if df_src is None: QMessageBox.warning(self, "Warning", "Load a dataset first."); return None

        if not is_test:
            self.feature_cols = self._get_selected_features()
            self.target_col = self.cb_target.currentText()
            
        if not self.feature_cols: QMessageBox.warning(self, "Warning", "Select features in the Dataset tab first."); return None

        df = df_src.copy()
        df = self._process_datetime(df)
        
        missing_cols = set(self.feature_cols + [self.target_col]) - set(df.columns)
        for c in missing_cols: df[c] = 0.0

        for col in df.columns:
            if df[col].dtype == 'object' or str(df[col].dtype).startswith('category'):
                df[col] = pd.Categorical(df[col]).codes.astype(np.float32)
        df = self._impute_missing(df).astype(np.float32)

        X_raw = df[self.feature_cols].values.astype(np.float32)
        y_raw = df[self.target_col].values.astype(np.float32)
        dates = df.index if self.datetime_index is not None else None

        seq_len = self.sp_seq_len.value(); horizon = self.sp_horizon.value(); stride = self.sp_stride.value()
        task = 'classification' if self.rb_clf.isChecked() else 'regression'

        if task == 'classification' and not is_test:
            y_raw = pd.Categorical(y_raw).codes.astype(np.int64); self.n_classes = int(y_raw.max()) + 1
        elif task == 'classification' and is_test:
            y_raw = pd.Categorical(y_raw).codes.astype(np.int64)

        n = len(df)
        if n <= seq_len + horizon: QMessageBox.warning(self, "Error", f"Dataset too small ({n} rows) for sequence length {seq_len}."); return None

        Xs, ys, ds = [], [], []
        for i in range(0, n - seq_len - horizon + 1, stride):
            Xs.append(X_raw[i:i+seq_len])
            if horizon == 1: ys.append(y_raw[i+seq_len+horizon-1])
            else: ys.append(y_raw[i+seq_len : i+seq_len+horizon])
            if dates is not None: ds.append(dates[i+seq_len+horizon-1])
                
        Xs = np.stack(Xs); ys = np.array(ys)
        if not is_test: self.eval_dates = ds if dates is not None else None

        if not is_test:
            self.horizon = horizon
            if self.chk_normalize.isChecked():
                mean = Xs.reshape(-1, Xs.shape[-1]).mean(axis=0)
                std = Xs.reshape(-1, Xs.shape[-1]).std(axis=0) + 1e-8
                self.normalization = (mean, std)
            else: self.normalization = None
            if task == 'regression' and self.chk_tnorm.isChecked(): self.target_norm = (ys.mean(), ys.std() + 1e-8)
            else: self.target_norm = None

        if self.normalization is not None: Xs = (Xs - self.normalization[0]) / self.normalization[1]
        if not is_test and self.target_norm is not None: ys = (ys - self.target_norm[0]) / self.target_norm[1]
        if is_test: return Xs, ys

        val_split = self.sp_val_split.value()
        n_val = max(1, int(len(Xs) * val_split))
        if self.chk_chrono.isChecked():
            X_train, X_val = Xs[:-n_val], Xs[-n_val:]; y_train, y_val = ys[:-n_val], ys[-n_val:]
        else:
            idx = np.random.permutation(len(Xs)); Xs, ys = Xs[idx], ys[idx]
            X_train, X_val = Xs[:-n_val], Xs[-n_val:]; y_train, y_val = ys[:-n_val], ys[-n_val:]

        X_train_t = torch.from_numpy(X_train).float(); X_val_t = torch.from_numpy(X_val).float()
        if task == 'classification':
            y_train_t = torch.from_numpy(y_train).long(); y_val_t = torch.from_numpy(y_val).long(); output_dim = self.n_classes
        else:
            y_train_t = torch.from_numpy(y_train).float(); 
            if horizon == 1: y_train_t = y_train_t.unsqueeze(-1)
            y_val_t = torch.from_numpy(y_val).float()
            if horizon == 1: y_val_t = y_val_t.unsqueeze(-1)
            output_dim = horizon * 3 if self.cb_loss.currentText() == "Quantile" else horizon

        self.train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=self.sp_batch.value(), shuffle=True)
        self.val_loader = DataLoader(TensorDataset(X_val_t, y_val_t), batch_size=self.sp_batch.value(), shuffle=False)
        return {'input_dim': Xs.shape[-1], 'output_dim': output_dim, 'task': task, 'n_train': len(X_train), 'n_val': len(X_val)}

    def _get_model_cfg(self, info):
        return {
            'input_dim': info['input_dim'], 'output_dim': info['output_dim'],
            'd_model': self.sp_d_model.value(), 'n_layers': self.sp_n_layers.value(),
            'd_state': self.sp_d_state.value(), 'd_conv': self.sp_d_conv.value(),
            'expand': self.sp_expand.value(), 'dropout': self.sp_dropout.value(),
            'bidirectional': self.chk_bidir.isChecked(), 
            'pooling': self.cb_pooling.currentText(), 'task': info['task']
        }

    # -------- MODEL I/O & VIZ --------
    def _print_model_summary(self):
        if self.model is None: QMessageBox.warning(self, "Warning", "Initialize model by starting training or loading weights."); return
        self._log("\n" + "="*50); self._log(str(self.model))
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self._log(f"Total Trainable Parameters: {total_params:,}"); self._log("="*50 + "\n")

    def _visualize_a_matrix(self):
        if self.model is None: QMessageBox.warning(self, "Warning", "Train or load a model first."); return
        ax = self.dt_canvas.ax; ax.clear()
        A_log = self.model.layers[-1].A_log.detach().cpu().numpy(); A = -np.exp(A_log)
        cax = ax.imshow(A.T, aspect='auto', cmap='viridis')
        ax.set_xlabel("d_inner (Inner Dimension)"); ax.set_ylabel("d_state (State Dimension)")
        ax.set_title("SSM A-Matrix (State Decay Parameters)\n(More negative = faster forgetting)")
        if hasattr(self, '_cbar_a'): self._cbar_a.remove()
        self._cbar_a = self.dt_canvas.fig.colorbar(cax, ax=ax)
        self.dt_canvas.fig.tight_layout(); self.dt_canvas.draw()
        self.tabs.setCurrentIndex(2)

    def _save_model(self):
        if not self.ensemble_states: QMessageBox.warning(self, "Warning", "No trained ensemble states found."); return
        path, _ = QFileDialog.getSaveFileName(self, "Save Model", "mamba_model_v11.pt", "PyTorch Files (*.pt)")
        if path:
            torch.save({
                'state_dicts': self.ensemble_states,
                'model_cfg': self._get_model_cfg({'input_dim': self.model.input_proj.in_features, 'output_dim': self.model.output_proj.out_features, 'task': self.model.task}),
                'norm': self.normalization, 'target_norm': self.target_norm,
                'feats': self.feature_cols, 'target': self.target_col
            }, path)
            self.status.showMessage(f"Model saved to {path}")

    def _load_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load Model", "", "PyTorch Files (*.pt)")
        if path:
            try:
                checkpoint = torch.load(path, map_location='cpu')
                cfg = checkpoint['model_cfg']
                self.sp_d_model.setValue(cfg['d_model']); self.sp_n_layers.setValue(cfg['n_layers'])
                self.sp_d_state.setValue(cfg['d_state']); self.sp_d_conv.setValue(cfg['d_conv'])
                self.sp_expand.setValue(cfg['expand']); self.sp_dropout.setValue(cfg['dropout'])
                self.chk_bidir.setChecked(cfg['bidirectional']); self.cb_pooling.setCurrentText(cfg.get('pooling', 'last'))
                if cfg['task'] == 'classification': self.rb_clf.setChecked(True)
                else: self.rb_reg.setChecked(True)
                
                self.model = MambaModel(**cfg)
                self.ensemble_states = checkpoint['state_dicts']
                self.model.load_state_dict(self.ensemble_states[0]) # Load first for viz
                self.normalization = checkpoint.get('norm'); self.target_norm = checkpoint.get('target_norm')
                self.feature_cols = checkpoint.get('feats', []); self.target_col = checkpoint.get('target')
                self.btn_test_infer.setEnabled(True)
                self.status.showMessage("Model loaded successfully.")
                QMessageBox.information(self, "Success", f"Loaded Ensemble of size {len(self.ensemble_states)}!")
            except Exception as e: QMessageBox.critical(self, "Error", f"Failed to load model:\n{e}")

    # -------- LR FINDER --------
    def _run_lr_finder(self):
        info = self._prepare_data()
        if info is None: return
        cfg = self._get_model_cfg(info)
        device = self._resolve_device()
        worker_cfg = {'device': device, 'task': info['task']}
        self.lr_finder = LRFinderWorker(cfg, self.train_loader, worker_cfg)
        self.lr_finder.plot.connect(self._plot_lr_finder)
        self.lr_finder.finished_signal.connect(self._on_lr_finder_done)
        self.lr_finder.error.connect(lambda e: QMessageBox.critical(self, "LR Finder Error", e))
        self.lr_finder.start()
        self.status.showMessage("Finding optimal learning rate...")

    def _plot_lr_finder(self, lrs, losses):
        ax = self.loss_canvas.ax; ax.clear()
        ax.plot(lrs, losses, color='#89b4fa', lw=2)
        ax.set_xscale('log')
        ax.set_xlabel("Learning Rate (Log Scale)"); ax.set_ylabel("Loss")
        ax.set_title("Learning Rate Finder"); ax.grid(True, alpha=0.3)
        self.loss_canvas.fig.tight_layout(); self.loss_canvas.draw()

    def _on_lr_finder_done(self, best_lr):
        self.sp_lr.setValue(best_lr)
        self.status.showMessage(f"LR Finder suggests: {best_lr:.6f}")
        QMessageBox.information(self, "LR Finder Complete", f"Suggested Learning Rate: {best_lr:.6f}\n\nIt has been automatically applied.")

    # -------- TRAINING --------
    def _resolve_device(self):
        choice = self.cb_device.currentText()
        if choice == "CPU": return "cpu"
        if choice == "CUDA": return "cuda" if torch.cuda.is_available() else "cpu"
        if choice == "MPS (Apple)": return "mps" if torch.backends.mps.is_available() else "cpu"
        if torch.cuda.is_available(): return "cuda"
        if torch.backends.mps.is_available(): return "mps"
        return "cpu"

    def _start_training(self):
        info = self._prepare_data()
        if info is None: return
        cfg = self._get_model_cfg(info)
        
        # Build a single instance for immediate UI interaction (viz A, etc)
        self.model = MambaModel(**cfg)
        if self.chk_compile.isChecked():
            try:
                self.model = torch.compile(self.model)
                self._log("⚡ torch.compile enabled.")
            except Exception as e: self._log(f"torch.compile failed: {e}")

        device = self._resolve_device()
        train_cfg = {
            'epochs': self.sp_epochs.value(), 'lr': self.sp_lr.value(),
            'weight_decay': self.sp_wd.value(), 'task': info['task'],
            'device': device, 'patience': self.sp_patience.value(),
            'scheduler': self.cb_scheduler.currentText().lower(),
            'loss_fn': self.cb_loss.currentText().lower(),
            'accum_steps': self.sp_accum.value(),
            'aug_jitter': self.chk_jitter.isChecked(),
            'aug_scale': self.chk_scale.isChecked(),
            'ensemble_size': self.sp_ensemble.value()
        }

        self.history = {'train_loss': [], 'val_loss': [], 'grad_norm': []}
        for c in [self.loss_canvas, self.grad_canvas, self.live_pred_canvas, self.heatmap_canvas, self.dt_canvas]:
            c.ax.clear(); c.draw()
        self.progress.setValue(0); self.progress.setMaximum(train_cfg['epochs'] * train_cfg['ensemble_size']); self.log_text.clear()

        amp_str = "+ AMP" if device == "cuda" else ""
        self._log(f"🚀 Starting training on [{device}] {amp_str}")
        self._log(f"Split: {'Chronological' if self.chk_chrono.isChecked() else 'Random'} | EMA: {self.chk_ema.isChecked()} | Ensemble: {train_cfg['ensemble_size']}")
        self._log("-" * 60)

        ema_decay = 0.999 if self.chk_ema.isChecked() else 0.0
        self.worker = TrainingWorker(cfg, self.train_loader, self.val_loader, train_cfg, self.target_norm, ema_decay)
        self.worker.log.connect(self._log)
        self.worker.epoch_finished.connect(self._on_epoch)
        self.worker.eval_plot.connect(self._update_live_plot)
        self.worker.heatmap_plot.connect(self._update_heatmap)
        self.worker.finished_signal.connect(self._on_train_done)
        self.worker.error.connect(self._on_train_error)
        self.worker.start()

        self.btn_train.setEnabled(False); self.btn_stop.setEnabled(True)
        self.btn_test_infer.setEnabled(False)
        self.status.showMessage(f"Training in progress on {device}...")

    def _stop_training(self):
        if self.worker: self.worker.stop()

    def _on_epoch(self, epoch, train_loss, val_loss, grad_norm):
        self.history['train_loss'].append(train_loss)
        self.history['val_loss'].append(val_loss)
        self.history['grad_norm'].append(grad_norm)
        self.progress.setValue(self.progress.value() + 1)
        self._update_loss_plot(); self._update_grad_plot()

    def _on_train_done(self, history, ensemble_states):
        self.btn_train.setEnabled(True); self.btn_stop.setEnabled(False); self.btn_test_infer.setEnabled(True)
        self.status.showMessage("Training complete. Best models saved to memory.")
        self.ensemble_states = ensemble_states
        if self.ensemble_states:
            # Load the first model to allow architecture viz (A-matrix, Attn)
            self.model.load_state_dict(self.ensemble_states[0])
            self._log("✅ Loaded first ensemble model weights for visualization.")
        self._log("-" * 60)
        self._log(f"✅ Training finished. Best val_loss={min(history['val_loss']):.6f}")

    def _on_train_error(self, msg):
        self.btn_train.setEnabled(True); self.btn_stop.setEnabled(False)
        self._log("❌ " + msg); QMessageBox.critical(self, "Training Error", msg)

    def _log(self, msg): self.log_text.append(msg)

    def _update_loss_plot(self):
        ax = self.loss_canvas.ax; ax.clear()
        epochs = range(1, len(self.history['train_loss']) + 1)
        ax.plot(epochs, self.history['train_loss'], label='Train', color='#89b4fa', lw=2)
        ax.plot(epochs, self.history['val_loss'], label='Val', color='#f9e2af', lw=2)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss")
        ax.set_title("Training & Validation Loss")
        ax.legend(); ax.grid(True, alpha=0.3)
        self.loss_canvas.fig.tight_layout(); self.loss_canvas.draw()

    def _update_grad_plot(self):
        ax = self.grad_canvas.ax; ax.clear()
        epochs = range(1, len(self.history['grad_norm']) + 1)
        ax.plot(epochs, self.history['grad_norm'], label='Grad Norm', color='#f38ba8', lw=2)
        ax.set_xlabel("Epoch"); ax.set_ylabel("L2 Norm")
        ax.set_title("Gradient Flow (Stability)")
        ax.legend(); ax.grid(True, alpha=0.3)
        self.grad_canvas.fig.tight_layout(); self.grad_canvas.draw()

    def _update_live_plot(self, actuals, preds):
        ax = self.live_pred_canvas.ax; ax.clear()
        n = min(150, len(preds)); x = range(n)
        if self.rb_clf.isChecked():
            ax.scatter(x, actuals[:n], label='Actual', color='#a6e3a1', s=20)
            ax.scatter(x, preds[:n], label='Pred', color='#f38ba8', marker='x', s=20)
        else:
            ax.plot(x, actuals[:n], label='Actual', color='#a6e3a1', lw=2)
            ax.plot(x, preds[:n], label='Pred (t+1)', color='#f38ba8', lw=2, linestyle='--')
        ax.set_xlabel("Sample"); ax.set_ylabel("Value")
        ax.set_title("Live Validation Predictions")
        ax.legend(); ax.grid(True, alpha=0.3)
        self.live_pred_canvas.fig.tight_layout(); self.live_pred_canvas.draw()

    def _update_heatmap(self, hs_matrix, dt_array):
        ax = self.heatmap_canvas.ax; ax.clear()
        cax = ax.imshow(hs_matrix.T, aspect='auto', cmap='magma')
        ax.set_xlabel("Sequence Time Step (t)"); ax.set_ylabel("State Dimension (h)")
        ax.set_title("SSM Hidden State Memory (Last Layer)")
        if hasattr(self, '_cbar_hs'): self._cbar_hs.remove()
        self._cbar_hs = self.heatmap_canvas.fig.colorbar(cax, ax=ax)
        self.heatmap_canvas.fig.tight_layout(); self.heatmap_canvas.draw()
        
        ax2 = self.dt_canvas.ax; ax2.clear()
        ax2.plot(dt_array, color='#fab387', lw=2)
        ax2.fill_between(range(len(dt_array)), dt_array, color='#fab387', alpha=0.3)
        ax2.set_xlabel("Sequence Time Step (t)"); ax2.set_ylabel("Avg $\Delta$ (Focus)")
        ax2.set_title("Selective Focus ($\Delta$) Over Sequence")
        ax2.grid(True, alpha=0.3)
        self.dt_canvas.fig.tight_layout(); self.dt_canvas.draw()

    # -------- INTERPRETABILITY --------
    def _run_feature_importance(self):
        if not self.ensemble_states or self.val_loader is None: QMessageBox.warning(self, "Warning", "Train a model first."); return
        self.status.showMessage("Calculating Feature Importance via Occlusion...")
        self.model.load_state_dict(self.ensemble_states[0]) # Use first model
        self.model.eval()
        device = next(self.model.parameters()).device
        base_losses, feat_losses = [], {f: [] for f in self.feature_cols}
        
        if self.rb_clf.isChecked(): criterion = nn.CrossEntropyLoss()
        elif self.cb_loss.currentText().lower() == 'huber': criterion = nn.SmoothL1Loss()
        else: criterion = nn.MSELoss()
        
        with torch.no_grad():
            for xb, yb in self.val_loader:
                xb, yb = xb.to(device), yb.to(device)
                out = self.model(xb)
                if self.rb_clf.isChecked() and yb.dim() > 1: yb = yb.argmax(-1)
                loss = criterion(out, yb); base_losses.append(loss.item())
                for i, f in enumerate(self.feature_cols):
                    xb_pert = xb.clone(); xb_pert[:, :, i] = 0.0 
                    out_p = self.model(xb_pert); loss_p = criterion(out_p, yb)
                    feat_losses[f].append(loss_p.item())
                    
        base_loss = np.mean(base_losses)
        importance = {f: ((np.mean(feat_losses[f]) - base_loss) / base_loss) * 100 for f in self.feature_cols}
        ax = self.res_canvas.ax; ax.clear()
        sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)
        feats, vals = zip(*sorted_imp)
        ax.barh(feats, vals, color='#89b4fa')
        ax.set_xlabel(f"% Increase in Loss (Base: {base_loss:.4f})"); ax.set_title("Feature Importance (Occlusion)")
        ax.invert_yaxis()
        self.res_canvas.fig.tight_layout(); self.res_canvas.draw()
        self.status.showMessage("Feature Importance calculated.")

    def _generate_integrated_gradients(self):
        if not self.ensemble_states or self.val_loader is None: QMessageBox.warning(self, "Warning", "Train a model first."); return
        self.status.showMessage("Generating Integrated Gradients...")
        self.model.load_state_dict(self.ensemble_states[0])
        self.model.eval()
        device = next(self.model.parameters()).device
        
        xb, yb = next(iter(self.val_loader))
        xb = xb.to(device); yb = yb.to(device)
        
        target_idx = 0
        xb_req = xb[target_idx:target_idx+1].clone().detach().requires_grad_(True)
        baseline = torch.zeros_like(xb_req)
        
        steps = 50
        alphas = np.linspace(0, 1, steps)
        total_grads = torch.zeros_like(xb_req)
        
        for alpha in alphas:
            x_interp = baseline + alpha * (xb_req - baseline)
            x_interp.requires_grad_(True)
            out = self.model(x_interp)
            
            if self.rb_clf.isChecked():
                pred_class = out.argmax(dim=-1)[0]
                out[0, pred_class].backward()
            else:
                out[0, 0].backward()
                
            total_grads += x_interp.grad.detach()
            x_interp.grad.zero_()
            
        avg_grads = (total_grads / steps)[0].abs().mean(dim=-1).cpu().numpy() 
        input_seq = xb[0, :, 0].cpu().detach().numpy() 
        
        ax = self.res_canvas.ax; ax.clear()
        ax.plot(input_seq, color='#89b4fa', lw=2, label='Feature 0 (Norm)')
        ax.imshow(avg_grads[np.newaxis, :], cmap='Reds', aspect='auto', alpha=0.5, 
                  extent=[0, len(avg_grads), input_seq.min(), input_seq.max()])
        ax.set_title("Integrated Gradients (Robust Saliency Map)")
        ax.set_xlabel("Sequence Time Step (t)"); ax.legend()
        self.res_canvas.fig.tight_layout(); self.res_canvas.draw()
        self.status.showMessage("Integrated Gradients generated.")

    def _plot_attention_weights(self):
        if not self.ensemble_states or self.val_loader is None: QMessageBox.warning(self, "Warning", "Train a model first."); return
        if self.cb_pooling.currentText() != 'attn': QMessageBox.warning(self, "Warning", "Attention pooling is not enabled."); return
        
        self.model.load_state_dict(self.ensemble_states[0])
        self.model.eval()
        device = next(self.model.parameters()).device
        xb, yb = next(iter(self.val_loader))
        xb = xb.to(device)
        _ = self.model(xb) 
        
        attn_w = self.model.last_attn[0].cpu().numpy()
        ax = self.res_canvas.ax; ax.clear()
        ax.plot(attn_w, color='#fab387', lw=3)
        ax.fill_between(range(len(attn_w)), attn_w, color='#fab387', alpha=0.3)
        ax.set_title("Attention Pooling Weights (Time-Step Importance)")
        ax.set_xlabel("Sequence Time Step (t)"); ax.set_ylabel("Attention Weight")
        ax.grid(True, alpha=0.3)
        self.res_canvas.fig.tight_layout(); self.res_canvas.draw()

    # -------- FINAL EVALUATION & TEST --------
    def _run_test_inference(self):
        path = self.test_path.text().strip()
        if not path or not os.path.exists(path): QMessageBox.warning(self, "Error", "Please select a valid Test CSV file."); return
        try: self.test_df = pd.read_csv(path)
        except Exception as e: QMessageBox.critical(self, "Error", f"Failed to read Test CSV:\n{e}"); return

        Xs, ys = self._prepare_data(df_override=self.test_df)
        if Xs is None: return

        device = next(self.model.parameters()).device
        X_tensor = torch.from_numpy(Xs).float().to(device)
        
        use_mc = self.chk_mc_dropout.isChecked() and self.rb_reg.isChecked()
        use_quantile = (self.cb_loss.currentText() == "Quantile") and self.rb_reg.isChecked()
        use_ensemble = len(self.ensemble_states) > 1 and self.rb_reg.isChecked()
        
        if use_ensemble:
            self.model.eval()
            ens_preds = []
            with torch.no_grad():
                for state in self.ensemble_states:
                    self.model.load_state_dict(state)
                    out = self.model(X_tensor)
                    p = out if out.shape[-1] == 1 else out[:, 0]
                    ens_preds.append(p.squeeze(-1).cpu().numpy())
            preds_arr = np.stack(ens_preds)
            preds = preds_arr.mean(axis=0)
            preds_std = preds_arr.std(axis=0)
            if self.target_norm:
                preds = preds * self.target_norm[1] + self.target_norm[0]
                preds_std = preds_std * self.target_norm[1]
        elif use_mc:
            self.model.train()
            preds_list = []
            with torch.no_grad():
                for _ in range(30):
                    out = self.model(X_tensor)
                    p = out if out.shape[-1] == 1 else out[:, 0]
                    preds_list.append(p.squeeze(-1).cpu().numpy())
            preds_arr = np.stack(preds_list)
            preds = preds_arr.mean(axis=0); preds_std = preds_arr.std(axis=0)
            if self.target_norm:
                preds = preds * self.target_norm[1] + self.target_norm[0]
                preds_std = preds_std * self.target_norm[1]
        else:
            self.model.eval()
            with torch.no_grad():
                out = self.model(X_tensor)
                if self.rb_clf.isChecked(): preds = out.argmax(dim=-1).cpu().numpy()
                elif use_quantile:
                    out_reshaped = out.view(out.size(0), self.horizon, 3)
                    p10 = out_reshaped[:, 0, 0].cpu().numpy(); p50 = out_reshaped[:, 0, 1].cpu().numpy(); p90 = out_reshaped[:, 0, 2].cpu().numpy()
                else:
                    p = out if out.shape[-1] == 1 else out[:, 0]
                    preds = p.squeeze(-1).cpu().numpy()
                    if self.target_norm: preds = preds * self.target_norm[1] + self.target_norm[0]

        if use_ensemble or use_mc: self._plot_mc_predictions(preds, preds_std, ys)
        elif use_quantile:
            if self.target_norm:
                p10 = p10 * self.target_norm[1] + self.target_norm[0]; p50 = p50 * self.target_norm[1] + self.target_norm[0]; p90 = p90 * self.target_norm[1] + self.target_norm[0]
            self._plot_quantile_predictions(p10, p50, p90, ys)
        else: self._plot_final_predictions(preds, ys)
            
        self._compute_metrics(preds if not use_quantile else p50, ys if ys.ndim == 1 else ys[:, 0], use_ensemble or use_mc, preds_std if use_ensemble or use_mc else None)

        save_path, _ = QFileDialog.getSaveFileName(self, "Save Predictions", "predictions.csv", "CSV files (*.csv)")
        if save_path:
            offset = self.sp_seq_len.value() + self.sp_horizon.value() - 1
            export_df = self.test_df.iloc[offset:].copy()
            if use_quantile:
                export_df['Pred_P10'] = p10; export_df['Pred_P50'] = p50; export_df['Pred_P90'] = p90
            else:
                export_df['Predicted'] = preds
                if use_ensemble or use_mc:
                    export_df['CI_Lower_95'] = preds - 1.96 * preds_std
                    export_df['CI_Upper_95'] = preds + 1.96 * preds_std
            export_df.to_csv(save_path, index=False)
            self.status.showMessage(f"Predictions exported to {save_path}")
            QMessageBox.information(self, "Success", f"Predictions exported successfully!\nRows: {len(export_df)}")

    def _plot_final_predictions(self, preds, acts):
        ax = self.pred_canvas.ax; ax.clear()
        n = min(500, len(preds))
        x = self.eval_dates[:n] if self.eval_dates is not None else range(n)
        if self.rb_clf.isChecked():
            ax.scatter(x, acts[:n], label='Actual', color='#a6e3a1', s=15)
            ax.scatter(x, preds[:n], label='Predicted', color='#f38ba8', marker='x', s=15)
        else:
            ax.plot(x, acts[:n], label='Actual', color='#a6e3a1', lw=2)
            ax.plot(x, preds[:n], label='Predicted (t+1)', color='#f38ba8', lw=2, linestyle='--')
        ax.set_xlabel("Date" if self.eval_dates is not None else "Sample"); ax.set_ylabel("Value")
        ax.set_title("Final Predictions vs Actual")
        ax.legend(); ax.grid(True, alpha=0.3)
        if self.eval_dates is not None: ax.tick_params(axis='x', rotation=30)
        self.pred_canvas.fig.tight_layout(); self.pred_canvas.draw()
        
        ax_res = self.res_canvas.ax; ax_res.clear()
        if self.rb_clf.isChecked():
            cm = confusion_matrix(acts, preds)
            cax = ax_res.imshow(cm, cmap='Blues')
            ax_res.set_title("Confusion Matrix"); ax_res.set_xlabel("Predicted"); ax_res.set_ylabel("Actual")
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    ax_res.text(j, i, str(cm[i, j]), ha='center', va='center', color='white' if cm[i,j] > cm.max()/2 else 'black')
        else:
            residuals = acts[:n] - preds[:n]
            ax_res.scatter(preds[:n], residuals, color='#89b4fa', s=10)
            ax_res.axhline(0, color='#f38ba8', linestyle='--')
            ax_res.set_title("Residuals Plot"); ax_res.set_xlabel("Predicted"); ax_res.set_ylabel("Residual (Actual - Pred)")
        self.res_canvas.fig.tight_layout(); self.res_canvas.draw()

    def _plot_mc_predictions(self, preds, stds, acts):
        ax = self.pred_canvas.ax; ax.clear()
        n = min(500, len(preds))
        x = self.eval_dates[:n] if self.eval_dates is not None else range(n)
        ax.fill_between(x, preds[:n] - 1.96*stds[:n], preds[:n] + 1.96*stds[:n], color='#f38ba8', alpha=0.2, label='95% CI')
        ax.plot(x, acts[:n], label='Actual', color='#a6e3a1', lw=2)
        ax.plot(x, preds[:n], label='Mean Pred', color='#f38ba8', lw=2, linestyle='--')
        ax.set_xlabel("Date" if self.eval_dates is not None else "Sample"); ax.set_ylabel("Value")
        ax.set_title("Uncertainty Predictions (Ensemble or MC Dropout)")
        ax.legend(); ax.grid(True, alpha=0.3)
        if self.eval_dates is not None: ax.tick_params(axis='x', rotation=30)
        self.pred_canvas.fig.tight_layout(); self.pred_canvas.draw()
        
        ax_res = self.res_canvas.ax; ax_res.clear()
        ax_res.plot(stds[:n], color='#fab387', lw=2)
        ax_res.set_title("Prediction Uncertainty (Std Dev)"); ax_res.set_xlabel("Sample"); ax_res.set_ylabel("Standard Deviation")
        self.res_canvas.fig.tight_layout(); self.res_canvas.draw()

    def _plot_quantile_predictions(self, p10, p50, p90, acts):
        ax = self.pred_canvas.ax; ax.clear()
        n = min(500, len(p50))
        x = self.eval_dates[:n] if self.eval_dates is not None else range(n)
        ax.fill_between(x, p10[:n], p90[:n], color='#f38ba8', alpha=0.2, label='80% CI (P10-P90)')
        ax.plot(x, acts[:n], label='Actual', color='#a6e3a1', lw=2)
        ax.plot(x, p50[:n], label='Median Pred (P50)', color='#f38ba8', lw=2, linestyle='--')
        ax.set_xlabel("Date" if self.eval_dates is not None else "Sample"); ax.set_ylabel("Value")
        ax.set_title("Probabilistic Forecast (Quantile Regression)")
        ax.legend(); ax.grid(True, alpha=0.3)
        if self.eval_dates is not None: ax.tick_params(axis='x', rotation=30)
        self.pred_canvas.fig.tight_layout(); self.pred_canvas.draw()
        
        ax_res = self.res_canvas.ax; ax_res.clear()
        residuals = acts[:n] - p50[:n]
        ax_res.scatter(p50[:n], residuals, color='#89b4fa', s=10)
        ax_res.axhline(0, color='#f38ba8', linestyle='--')
        ax_res.set_title("Residuals Plot (Median)"); ax_res.set_xlabel("Predicted"); ax_res.set_ylabel("Residual (Actual - Pred)")
        self.res_canvas.fig.tight_layout(); self.res_canvas.draw()

    def _compute_metrics(self, preds, acts, use_uncertainty=False, stds=None):
        if self.rb_clf.isChecked():
            report = classification_report(acts, preds, zero_division=0)
            text = f"--- Classification Report ---\n{report}\n\nSamples: {len(preds)}"
        else:
            mse = np.mean((preds - acts) ** 2); rmse = np.sqrt(mse); mae = np.mean(np.abs(preds - acts))
            ss_res = np.sum((acts - preds) ** 2); ss_tot = np.sum((acts - acts.mean()) ** 2) + 1e-12
            r2 = 1 - ss_res / ss_tot
            text = (f"--- Regression Metrics ---\n"
                    f"MSE:  {mse:.6f}\nRMSE: {rmse:.6f}\nMAE:  {mae:.6f}\nR²:   {r2:.6f}\n\nSamples: {len(preds)}")
            if use_uncertainty:
                avg_std = np.mean(stds)
                text += f"\nAvg Uncertainty (Std): {avg_std:.4f}"
        self.metrics_text.setPlainText(text)


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
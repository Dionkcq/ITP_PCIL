"""
Cyclical anomaly pipeline — Step 4: model
==========================================
Two models sharing the same .fit(X) / .score(X) interface.

  IsolationForestModel — sklearn-native, unsupervised baseline.
  AutoencoderModel     — 1D CNN autoencoder. Selected model.

Selected: AutoencoderModel
----------------------------
Based on Winardi's recommendation (meeting 22 May 2026):
  "You can also explore 1D CNN autoencoder. That one was more accurate
   than just a normal encoder for the curve, because it will treat the
   curve like a 1D image and take note of all the highs and lows."

The autoencoder works with ANY feature set (stats=15, waveform=100, fft=20).
The CNN architecture adapts automatically to the input length — shallower
for short inputs (stats/fft), deeper for longer ones (waveform).

Architecture (adaptive)
-----------------------
For input_len >= 32  (e.g. waveform=100, fft=20 padded):
  Encoder: 3x Conv1d blocks with progressive pooling
  Bottleneck: Linear -> 16 dimensions
  Decoder: mirrors encoder with ConvTranspose1d

For input_len < 32  (e.g. stats=15):
  Encoder: 2x Conv1d blocks, no pooling on last
  Bottleneck: Linear -> 8 dimensions
  Decoder: mirrors accordingly

Loss      : MSE reconstruction loss
Optimiser : Adam, lr=1e-3
Stopping  : early stopping on val loss, patience=10, max 200 epochs
Score     : MSE reconstruction error per cycle — higher = more anomalous
"""

from __future__ import annotations

import numpy as np
from pcil.utils.anomaly.base import AnomalyModel


# ─────────────────────────────────────────────────────────────
# Isolation Forest (baseline)
# ─────────────────────────────────────────────────────────────

class IsolationForestModel(AnomalyModel):
    """sklearn IsolationForest. Higher score = more anomalous."""

    def __init__(self, **kwargs):
        from sklearn.ensemble import IsolationForest
        params = {"n_estimators": 100, "contamination": 0.05, "random_state": 42}
        params.update(kwargs)
        self._model = IsolationForest(**params)

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "IsolationForestModel":
        self._model.fit(X)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        return -self._model.score_samples(X)


# ─────────────────────────────────────────────────────────────
# Adaptive 1D CNN Autoencoder network (module-level for pickle)
# ─────────────────────────────────────────────────────────────

try:
    import torch.nn as nn

    class _AdaptiveCNN1DAutoencoder(nn.Module):
        """
        1D CNN autoencoder that adapts its architecture to the input length.
        Defined at module level so joblib/pickle can serialise it.

        Works with:
          - waveform features (input_len=100) — full 3-block architecture
          - fft features     (input_len=20)  — 2-block architecture
          - stats features   (input_len=15)  — 2-block architecture, no pooling
        """

        def __init__(self, input_len: int, bottleneck: int = 16):
            super().__init__()
            self.input_len = input_len

            if input_len >= 64:
                # Deep architecture for long inputs (waveform=100)
                # Encoder: 100 -> 50 -> 25 -> 5
                self.enc = nn.Sequential(
                    nn.Conv1d(1, 16, kernel_size=7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
                    nn.Conv1d(16, 32, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
                    nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool1d(5),
                )
                self._flat = 64 * (input_len // 20)
                self.enc_fc = nn.Linear(self._flat, bottleneck)
                self.dec_fc = nn.Sequential(nn.Linear(bottleneck, self._flat), nn.ReLU())
                self.dec = nn.Sequential(
                    nn.ConvTranspose1d(64, 32, kernel_size=5, stride=5),  nn.ReLU(),
                    nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1), nn.ReLU(),
                    nn.ConvTranspose1d(16,  1, kernel_size=4, stride=2, padding=1),
                )

            elif input_len >= 16:
                # Shallow architecture for medium inputs (fft=20, stats padded)
                # Encoder: L -> L//2 -> L//4
                pool = 2
                after_pool = input_len // (pool * pool)
                self.enc = nn.Sequential(
                    nn.Conv1d(1, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool1d(pool),
                    nn.Conv1d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool1d(pool),
                )
                self._flat = 32 * after_pool
                self.enc_fc = nn.Linear(self._flat, bottleneck)
                self.dec_fc = nn.Sequential(nn.Linear(bottleneck, self._flat), nn.ReLU())
                self.dec = nn.Sequential(
                    nn.ConvTranspose1d(32, 16, kernel_size=4, stride=pool, padding=1), nn.ReLU(),
                    nn.ConvTranspose1d(16,  1, kernel_size=4, stride=pool, padding=1),
                )

            else:
                # Minimal architecture for short inputs (stats=15)
                # No pooling — just conv + bottleneck
                self.enc = nn.Sequential(
                    nn.Conv1d(1, 8, kernel_size=3, padding=1), nn.ReLU(),
                    nn.Conv1d(8, 16, kernel_size=3, padding=1), nn.ReLU(),
                )
                self._flat = 16 * input_len
                bn = max(4, bottleneck // 2)
                self.enc_fc = nn.Linear(self._flat, bn)
                self.dec_fc = nn.Sequential(nn.Linear(bn, self._flat), nn.ReLU())
                self.dec = nn.Sequential(
                    nn.ConvTranspose1d(16, 8, kernel_size=3, padding=1), nn.ReLU(),
                    nn.ConvTranspose1d(8,  1, kernel_size=3, padding=1),
                )

        def forward(self, x):
            # Encode
            z = self.enc(x)                          # (B, C, L')
            z_flat = z.view(z.size(0), -1)           # (B, C*L')
            z_bottle = self.enc_fc(z_flat)           # (B, bottleneck)
            # Decode
            z_up = self.dec_fc(z_bottle)             # (B, C*L')
            z_up = z_up.view(z.size(0), z.size(1), z.size(2))  # (B, C, L')
            out = self.dec(z_up)                     # (B, 1, ~input_len)
            # Trim or pad to exactly input_len
            if out.size(2) > self.input_len:
                out = out[:, :, :self.input_len]
            elif out.size(2) < self.input_len:
                out = nn.functional.pad(out, (0, self.input_len - out.size(2)))
            return out

except ImportError:
    _AdaptiveCNN1DAutoencoder = None


def _build_net(input_len: int, bottleneck: int = 16):
    if _AdaptiveCNN1DAutoencoder is None:
        raise ImportError("PyTorch required. Run: pip install torch")
    return _AdaptiveCNN1DAutoencoder(input_len=input_len, bottleneck=bottleneck)


# ─────────────────────────────────────────────────────────────
# 1D CNN Autoencoder  ← SELECTED
# ─────────────────────────────────────────────────────────────

class AutoencoderModel(AnomalyModel):
    """
    Adaptive 1D CNN Autoencoder for cyclical anomaly detection.

    Works with any feature method:
      - waveform (100 features) — full deep architecture
      - fft      (20 features)  — shallow architecture
      - stats    (15 features)  — minimal architecture

    input_len is inferred automatically from X.shape[1] at fit time.

    Parameters
    ----------
    bottleneck : latent dimension (default 16)
    epochs     : max training epochs (default 200)
    batch_size : mini-batch size (default 16)
    lr         : Adam learning rate (default 1e-3)
    patience   : early stopping patience (default 10)
    val_frac   : validation fraction for early stopping (default 0.15)
    verbose    : print training progress (default True)
    """

    def __init__(
        self,
        bottleneck: int = 16,
        epochs: int = 200,
        batch_size: int = 16,
        lr: float = 1e-3,
        patience: int = 10,
        val_frac: float = 0.15,
        verbose: bool = True,
    ):
        self.bottleneck = bottleneck
        self.epochs     = epochs
        self.batch_size = batch_size
        self.lr         = lr
        self.patience   = patience
        self.val_frac   = val_frac
        self.verbose    = verbose
        self._net       = None
        self._device    = None
        self.input_len  = None

    def _to_tensor(self, X: np.ndarray):
        import torch
        return torch.tensor(X, dtype=torch.float32, device=self._device).unsqueeze(1)

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "AutoencoderModel":
        """Train on cycle features. X shape: (N, n_features). input_len inferred from X."""
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        self.input_len = X.shape[1]
        self._device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.verbose:
            print(f"[AutoencoderModel] device={self._device}  "
                  f"cycles={len(X)}  input_len={self.input_len}  bottleneck={self.bottleneck}")

        n_val   = max(1, int(len(X) * self.val_frac))
        idx     = np.random.default_rng(42).permutation(len(X))
        X_val   = X[idx[:n_val]]
        X_train = X[idx[n_val:]]

        if len(X_train) < 2:
            raise ValueError(f"Too few training cycles: {len(X_train)}")

        loader  = DataLoader(TensorDataset(self._to_tensor(X_train)),
                             batch_size=min(self.batch_size, len(X_train)), shuffle=True)
        t_val   = self._to_tensor(X_val)
        loss_fn = nn.MSELoss()

        self._net = _build_net(self.input_len, self.bottleneck).to(self._device)
        opt       = torch.optim.Adam(self._net.parameters(), lr=self.lr)

        best_val, patience_ctr, best_state = float("inf"), 0, None

        for epoch in range(1, self.epochs + 1):
            self._net.train()
            t_loss = 0.0
            for (batch,) in loader:
                opt.zero_grad()
                loss = loss_fn(self._net(batch), batch)
                loss.backward()
                opt.step()
                t_loss += loss.item() * len(batch)
            t_loss /= len(X_train)

            self._net.eval()
            with torch.no_grad():
                v_loss = loss_fn(self._net(t_val), t_val).item()

            if self.verbose and (epoch % 20 == 0 or epoch == 1):
                print(f"  epoch {epoch:>4}  train={t_loss:.5f}  val={v_loss:.5f}")

            if v_loss < best_val - 1e-6:
                best_val, patience_ctr = v_loss, 0
                best_state = {k: v.clone() for k, v in self._net.state_dict().items()}
            else:
                patience_ctr += 1
                if patience_ctr >= self.patience:
                    if self.verbose:
                        print(f"  Early stop at epoch {epoch}  best_val={best_val:.5f}")
                    break

        if best_state:
            self._net.load_state_dict(best_state)
        if self.verbose:
            print(f"[AutoencoderModel] Done. Best val loss: {best_val:.5f}")

        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        """Return per-cycle MSE reconstruction error. Higher = more anomalous."""
        import torch
        if self._net is None:
            raise RuntimeError("Model not trained — call fit() first.")
        self._net.eval()
        t = self._to_tensor(X)
        with torch.no_grad():
            mse = ((self._net(t) - t) ** 2).mean(dim=2).squeeze(1)
        return mse.cpu().numpy()
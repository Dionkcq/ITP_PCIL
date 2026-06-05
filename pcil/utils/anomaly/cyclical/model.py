"""
Cyclical anomaly pipeline — Step 4: model
==========================================
Two models, both sharing the same .fit(X) / .score(X) interface.

Models
------
  IsolationForestModel — sklearn-native, unsupervised baseline.
  AutoencoderModel     — 1D CNN autoencoder. Selected model.

Selected: AutoencoderModel
----------------------------
Based on Winardi's recommendation (meeting 22 May 2026):
  "You can also explore 1D CNN autoencoder. That one was more accurate
   than just a normal encoder for the curve, because it will treat the
   curve like a 1D image and take note of all the highs and lows."

Architecture
------------
  Input  : (batch, 1, 100) — 100-sample resampled waveform

  Encoder:
    Conv1d(1  -> 16, kernel=7, padding=3) + ReLU + MaxPool(2)  -> (batch, 16, 50)
    Conv1d(16 -> 32, kernel=5, padding=2) + ReLU + MaxPool(2)  -> (batch, 32, 25)
    Conv1d(32 -> 64, kernel=3, padding=1) + ReLU + MaxPool(5)  -> (batch, 64,  5)
    Flatten -> Linear(320 -> bottleneck)                        -> (batch, 16)

  Decoder:
    Linear(16 -> 320) + ReLU -> Reshape (batch, 64, 5)
    ConvTranspose1d(64 -> 32, kernel=5, stride=5)              -> (batch, 32, 25)
    ConvTranspose1d(32 -> 16, kernel=4, stride=2, padding=1)   -> (batch, 16, 50)
    ConvTranspose1d(16 ->  1, kernel=4, stride=2, padding=1)   -> (batch,  1,100)

  Loss      : MSE reconstruction loss
  Optimiser : Adam, lr=1e-3
  Stopping  : early stopping on validation loss, patience=10, max 200 epochs

  Anomaly score: MSE reconstruction error — higher = more anomalous.
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
# 1D CNN Autoencoder  ← SELECTED
# ─────────────────────────────────────────────────────────────

try:
    import torch.nn as nn

    class _CNN1DAutoencoder(nn.Module):
        """
        1D CNN autoencoder network.
        Defined at module level so joblib/pickle can serialise it.
        """
        def __init__(self, bottleneck: int = 16):
            super().__init__()
            self.enc_conv = nn.Sequential(
                nn.Conv1d(1, 16, kernel_size=7, padding=3), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(16, 32, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool1d(2),
                nn.Conv1d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool1d(5),
            )
            self.enc_fc = nn.Linear(64 * 5, bottleneck)
            self.dec_fc = nn.Sequential(nn.Linear(bottleneck, 64 * 5), nn.ReLU())
            self.dec_conv = nn.Sequential(
                nn.ConvTranspose1d(64, 32, kernel_size=5, stride=5),  nn.ReLU(),
                nn.ConvTranspose1d(32, 16, kernel_size=4, stride=2, padding=1), nn.ReLU(),
                nn.ConvTranspose1d(16,  1, kernel_size=4, stride=2, padding=1),
            )

        def forward(self, x):
            z = self.enc_conv(x)
            z = self.enc_fc(z.view(z.size(0), -1))
            z = self.dec_fc(z).view(z.size(0), 64, 5)
            return self.dec_conv(z)

except ImportError:
    _CNN1DAutoencoder = None


def _build_net(bottleneck: int = 16):
    if _CNN1DAutoencoder is None:
        raise ImportError("PyTorch required. Run: pip install torch")
    return _CNN1DAutoencoder(bottleneck=bottleneck)


class AutoencoderModel(AnomalyModel):
    """
    1D CNN Autoencoder for cyclical anomaly detection.

    Trains on normal cycle waveforms only (unsupervised).
    Learns to reconstruct normal cycles accurately. Anomalous cycles
    reconstruct poorly, giving a high MSE anomaly score.

    Parameters
    ----------
    input_len  : waveform length — must match N_WAVEFORM in features.py (default 100)
    bottleneck : latent dimension (default 16)
    epochs     : max training epochs (default 200)
    batch_size : mini-batch size (default 16)
    lr         : Adam learning rate (default 1e-3)
    patience   : early stopping patience in epochs (default 10)
    val_frac   : fraction held out for early stopping validation (default 0.15)
    verbose    : print training progress (default True)
    """

    def __init__(
        self,
        input_len: int = 100,
        bottleneck: int = 16,
        epochs: int = 200,
        batch_size: int = 16,
        lr: float = 1e-3,
        patience: int = 10,
        val_frac: float = 0.15,
        verbose: bool = True,
    ):
        self.input_len  = input_len
        self.bottleneck = bottleneck
        self.epochs     = epochs
        self.batch_size = batch_size
        self.lr         = lr
        self.patience   = patience
        self.val_frac   = val_frac
        self.verbose    = verbose
        self._net       = None
        self._device    = None

    def _to_tensor(self, X: np.ndarray):
        import torch
        return torch.tensor(X, dtype=torch.float32, device=self._device).unsqueeze(1)

    def fit(self, X: np.ndarray, y: np.ndarray | None = None) -> "AutoencoderModel":
        """Train on normal cycle waveforms. X shape: (N, input_len)."""
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if X.shape[1] != self.input_len:
            raise ValueError(
                f"Expected {self.input_len} features, got {X.shape[1]}. "
                "Ensure FEATURE_METHOD='waveform' and N_WAVEFORM=100."
            )

        if self.verbose:
            print(f"[AutoencoderModel] device={self._device}  "
                  f"cycles={len(X)}  input_len={self.input_len}  bottleneck={self.bottleneck}")

        n_val   = max(1, int(len(X) * self.val_frac))
        idx     = np.random.default_rng(42).permutation(len(X))
        X_val   = X[idx[:n_val]]
        X_train = X[idx[n_val:]]

        if len(X_train) < 2:
            raise ValueError(f"Too few training cycles: {len(X_train)}")

        loader = DataLoader(
            TensorDataset(self._to_tensor(X_train)),
            batch_size=min(self.batch_size, len(X_train)),
            shuffle=True,
        )
        t_val   = self._to_tensor(X_val)
        loss_fn = nn.MSELoss()

        self._net = _build_net(self.bottleneck).to(self._device)
        opt = torch.optim.Adam(self._net.parameters(), lr=self.lr)

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
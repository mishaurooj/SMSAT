# -*- coding: utf-8 -*-
"""
SMSAT/QMSAT Full Ablation Runner (robust, folder auto-discovery)

- Auto-discovers class folders under root_dir (no hard-coded names).
- Defensive augmentations (handles Windows/sox quirks; always crops to fixed length).
- Clear sanity checks (prints per-class counts; fails early if dataset empty).
- Same ablation matrix, metrics printing, linear eval, and Excel output.
"""

import os
import random
import glob
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split

import torchaudio
from torchaudio.transforms import MelSpectrogram, AmplitudeToDB

# -----------------------
# 0. Reproducibility
# -----------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# -----------------------
# 1. Augmentations
# -----------------------

@dataclass
class AugConfig:
    sample_rate: int = 16000
    crop_size: int = 16000
    use_noise: bool = True
    noise_factor: float = 0.005
    use_time_stretch: bool = True
    time_stretch_range: Tuple[float, float] = (0.8, 1.2)
    use_pitch_shift: bool = True
    pitch_steps_range: Tuple[float, float] = (-2.0, 2.0)
    use_spec_augment: bool = True  # applied on mel inside the model

class AudioAugmentations:
    def __init__(self, cfg: AugConfig):
        self.cfg = cfg
        self.sample_rate = cfg.sample_rate
        self.crop_size = cfg.crop_size

    def add_noise(self, waveform: torch.Tensor) -> torch.Tensor:
        if not self.cfg.use_noise:
            return waveform
        return waveform + torch.randn_like(waveform) * self.cfg.noise_factor

    def time_stretch(self, waveform: torch.Tensor) -> torch.Tensor:
        if not self.cfg.use_time_stretch:
            return waveform
        rate = float(random.uniform(*self.cfg.time_stretch_range))
        effects = [['speed', f'{rate}'], ['rate', f'{self.sample_rate}']]
        try:
            stretched, _ = torchaudio.sox_effects.apply_effects_tensor(
                waveform.float().cpu(), self.sample_rate, effects
            )
            # Always crop/pad to keep shapes consistent
            return self.random_crop(stretched)
        except Exception:
            return waveform

    def pitch_shift_fn(self, waveform: torch.Tensor) -> torch.Tensor:
        if not self.cfg.use_pitch_shift:
            return waveform
        steps = float(random.uniform(*self.cfg.pitch_steps_range))
        try:
            # PitchShift prefers CPU float tensors
            shift = torchaudio.transforms.PitchShift(self.sample_rate, n_steps=steps)
            return shift(waveform.float().cpu())
        except Exception:
            return waveform

    def random_crop(self, waveform: torch.Tensor) -> torch.Tensor:
        # waveform: (1, T) or (C, T). We keep channel count unchanged.
        T = waveform.shape[-1]
        if T > self.crop_size:
            start = random.randint(0, T - self.crop_size)
            return waveform[..., start:start + self.crop_size]
        elif T < self.crop_size:
            pad_amt = self.crop_size - T
            return F.pad(waveform, (0, pad_amt))
        return waveform

    def __call__(self, waveform: torch.Tensor) -> torch.Tensor:
        # Defensive wrapper: never let aug errors kill the worker
        x = waveform
        try:
            x = self.add_noise(x)
            x = self.time_stretch(x)
            x = self.pitch_shift_fn(x)
        except Exception as e:
            print(f"[AUG WARN] {e}. Using original waveform.")
            x = waveform
        # Final shape enforcement
        x = self.random_crop(x)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return x

# -----------------------
# 2. Datasets (auto-discovery)
# -----------------------

# Will be filled dynamically from the folder names in root_dir
LABELS: List[str] = []
LABEL_TO_IDX: Dict[str, int] = {}

def discover_labels(root_dir: str) -> List[str]:
    classes = sorted([p.name for p in Path(root_dir).iterdir() if p.is_dir()])
    if not classes:
        raise RuntimeError(f"No class subfolders found in '{root_dir}'.")
    return classes

class QMSAT(Dataset):
    """Contrastive dataset: two augmented views + label string."""
    def __init__(self, root_dir, sample_rate=16000, transform: AudioAugmentations=None):
        self.sample_rate = sample_rate
        self.transform = transform

        # Discover classes dynamically
        global LABELS, LABEL_TO_IDX
        LABELS = discover_labels(root_dir)
        LABEL_TO_IDX = {lab: i for i, lab in enumerate(LABELS)}

        self.file_paths: List[str] = []
        self.labels: List[str] = []

        per_class_counts = {}
        for sub_dir in LABELS:
            folder = Path(root_dir) / sub_dir
            wav_files = glob.glob(str(folder / '*.wav'))
            self.file_paths.extend(wav_files)
            self.labels.extend([sub_dir] * len(wav_files))
            per_class_counts[sub_dir] = len(wav_files)

        total = len(self.file_paths)
        print("Discovered classes:", LABELS)
        print("Class file counts:", per_class_counts, "TOTAL:", total)
        if total == 0:
            raise RuntimeError(f"No .wav files found in {root_dir}/*/*.wav")

    def __len__(self): return len(self.file_paths)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        waveform, sr = torchaudio.load(file_path)
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if sr != self.sample_rate:
            waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)

        if self.transform is not None:
            aug1 = self.transform(waveform.clone())
            aug2 = self.transform(waveform.clone())
        else:
            # still enforce fixed length if no transform
            aug1 = aug2 = AudioAugmentations(AugConfig(sample_rate=self.sample_rate)).random_crop(waveform)

        # Final guards: mono and (1, T)
        if aug1.dim() == 1: aug1 = aug1.unsqueeze(0)
        if aug2.dim() == 1: aug2 = aug2.unsqueeze(0)
        label = self.labels[idx]
        return aug1, aug2, label

class QMSAT_Eval(Dataset):
    """Supervised eval dataset: returns (waveform fixed len, label_idx) without aug."""
    def __init__(self, root_dir, sample_rate=16000, crop_size=16000):
        self.sample_rate = sample_rate
        self.crop_size = crop_size

        # Use the same dynamic labels
        global LABELS, LABEL_TO_IDX
        if not LABELS:
            LABELS = discover_labels(root_dir)
            LABEL_TO_IDX = {lab: i for i, lab in enumerate(LABELS)}

        self.paths: List[str] = []
        self.labels: List[int] = []
        per_class_counts = {}
        for sub in LABELS:
            folder = Path(root_dir) / sub
            files = glob.glob(str(folder / '*.wav'))
            self.paths.extend(files)
            self.labels.extend([LABEL_TO_IDX[sub]] * len(files))
            per_class_counts[sub] = len(files)

        total = len(self.paths)
        print("[Eval] Class file counts:", per_class_counts, "TOTAL:", total)
        if total == 0:
            raise RuntimeError(f"No .wav files found in {root_dir}/*/*.wav")

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        p = self.paths[idx]
        wav, sr = torchaudio.load(p)
        if wav.dim() == 1:
            wav = wav.unsqueeze(0)
        if sr != self.sample_rate:
            wav = torchaudio.transforms.Resample(sr, self.sample_rate)(wav)
        T = self.crop_size
        if wav.shape[1] > T:
            start = (wav.shape[1] - T) // 2
            wav = wav[:, start:start + T]
        elif wav.shape[1] < T:
            wav = F.pad(wav, (0, T - wav.shape[1]))
        return wav, self.labels[idx]

# -----------------------
# 3. Model
# -----------------------

@dataclass
class ModelConfig:
    projection_dim: int = 128
    pretrained: bool = True
    freeze_backbone: bool = False
    apply_spec_augment: bool = True  # apply SpecAug on mel inside the encoder

class QSMATATSEncoder(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.mel_spec = MelSpectrogram(sample_rate=16000, n_mels=64)
        self.db_transform = AmplitudeToDB()
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=15) if cfg.apply_spec_augment else None

        # torchvision v0.10.0 hub is fine for resnet18 weights
        resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=cfg.pretrained)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

        if cfg.freeze_backbone:
            for p in resnet.parameters():
                p.requires_grad = False

        self.encoder = nn.Sequential(*list(resnet.children())[:-1])  # (B, 512, 1, 1)
        self.projection = nn.Linear(512, cfg.projection_dim)

    def forward(self, x):
        # x: (B,1,T) or (B,T)
        if x.dim() == 3:
            x = x.squeeze(1)
        mel = self.mel_spec(x)          # (B, 64, time)
        mel_db = self.db_transform(mel)  # (B, 64, time)
        if self.freq_mask is not None:
            mel_db = self.freq_mask(mel_db)
        mel_db = mel_db.unsqueeze(1)     # (B,1,64,time)
        feats = self.encoder(mel_db).squeeze(-1).squeeze(-1)  # (B,512)
        proj = self.projection(feats)    # (B,D)
        return proj

    def encode(self, x):
        return self.forward(x)

# -----------------------
# 4. Train config / Loss
# -----------------------

@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    loss_type: str = "mse"    # 'mse' or 'ntxent'
    temperature: float = 0.1  # for NT-Xent
    seed: int = 42

def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    B, _ = z1.shape
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    z = torch.cat([z1, z2], dim=0)                   # (2B, D)
    sim = torch.matmul(z, z.t()) / temperature       # (2B, 2B)
    mask = torch.eye(2 * B, device=z.device, dtype=torch.bool)
    sim.masked_fill_(mask, -9e15)
    pos = torch.cat([torch.diag(sim, B), torch.diag(sim, -B)], dim=0)  # (2B,)
    denom = torch.logsumexp(sim, dim=1)
    loss = -pos + denom
    return loss.mean()

# -----------------------
# 5. SSL training (one experiment)
# -----------------------

def train_ssl_one_experiment(
    root_dir: str,
    aug_cfg: AugConfig,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    num_workers: int = 0,     # start at 0 on Windows to get real errors; raise to 2 later
    return_model: bool = True
) -> Dict[str, Any]:
    set_seed(train_cfg.seed)

    augmentations = AudioAugmentations(aug_cfg)
    dataset = QMSAT(root_dir=root_dir, sample_rate=aug_cfg.sample_rate, transform=augmentations)

    if len(dataset) == 0:
        raise RuntimeError(f"Dataset is empty. Check path: {root_dir}")

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    if train_size == 0 or val_size == 0:
        raise RuntimeError(f"Split too small (train={train_size}, val={val_size}). Add more data.")

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size],
                                              generator=torch.Generator().manual_seed(train_cfg.seed))

    pin = (device.type == "cuda")
    train_loader = DataLoader(train_dataset, batch_size=train_cfg.batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=pin, persistent_workers=False)
    val_loader = DataLoader(val_dataset, batch_size=train_cfg.batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin, persistent_workers=False)

    # Pass SpecAug flag from aug_cfg into model_cfg.apply_spec_augment
    model_cfg = ModelConfig(**{**asdict(model_cfg), "apply_spec_augment": aug_cfg.use_spec_augment})
    model = QSMATATSEncoder(model_cfg).to(device)
    optimizer = optim.Adam(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    train_losses, train_cosine = [], []
    val_losses, val_cosine = [], []
    epoch_times = []

    best_val = float('inf')
    best_epoch = -1
    best_snapshot: Dict[str, float] = {}

    total_start = time.time()
    print(f"Starting Self-Supervised Training on {device}")
    for epoch in range(1, train_cfg.epochs + 1):
        model.train()
        t0 = time.time()
        run_loss, run_cos, n_batches = 0.0, 0.0, 0

        for a1, a2, _ in train_loader:
            a1, a2 = a1.to(device), a2.to(device)
            optimizer.zero_grad()

            p1 = model(a1)
            p2 = model(a2)

            if train_cfg.loss_type == 'mse':
                loss = F.mse_loss(p1, p2)
            elif train_cfg.loss_type == 'ntxent':
                loss = nt_xent_loss(p1, p2, temperature=train_cfg.temperature)
            else:
                raise ValueError(f"Unknown loss_type={train_cfg.loss_type}")

            loss.backward()
            optimizer.step()

            run_loss += loss.item()
            cos = F.cosine_similarity(p1, p2, dim=1).mean().item()
            run_cos += cos
            n_batches += 1

        avg_train_loss = run_loss / max(n_batches, 1)
        avg_train_cos = run_cos / max(n_batches, 1)

        # Validation
        model.eval()
        vloss, vcos, v_batches = 0.0, 0.0, 0
        with torch.no_grad():
            for a1, a2, _ in val_loader:
                a1, a2 = a1.to(device), a2.to(device)
                p1 = model.encode(a1)
                p2 = model.encode(a2)

                if train_cfg.loss_type == 'mse':
                    l = F.mse_loss(p1, p2)
                else:
                    l = nt_xent_loss(p1, p2, temperature=train_cfg.temperature)

                vloss += l.item()
                vcos += F.cosine_similarity(p1, p2, dim=1).mean().item()
                v_batches += 1

        avg_val_loss = vloss / max(v_batches, 1)
        avg_val_cos  = vcos  / max(v_batches, 1)

        epoch_time = time.time() - t0
        epoch_times.append(epoch_time)
        train_losses.append(avg_train_loss)
        train_cosine.append(avg_train_cos)
        val_losses.append(avg_val_loss)
        val_cosine.append(avg_val_cos)

        print(f"[Epoch {epoch:02d}] Train Loss: {avg_train_loss:.4f} | Train Cos: {avg_train_cos:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | Val Cos: {avg_val_cos:.4f} | Time: {epoch_time:.2f} sec")

        if avg_val_loss < best_val:
            best_val = avg_val_loss
            best_epoch = epoch
            best_snapshot = {
                "best_epoch": best_epoch,
                "best_val_loss": float(avg_val_loss),
                "best_val_cosine": float(avg_val_cos),
                "train_loss_at_best": float(avg_train_loss),
                "train_cos_at_best": float(avg_train_cos),
            }

    total_time_sec = time.time() - total_start
    print(f"Total Training Time: {total_time_sec/60:.2f} minutes")

    history = {
        "train_losses": train_losses,
        "train_cosine": train_cosine,
        "val_losses": val_losses,
        "val_cosine": val_cosine,
        "epoch_times": epoch_times,
        "total_training_time_sec": total_time_sec,
        "best_snapshot": best_snapshot,
    }
    if return_model:
        history["model"] = model
    else:
        history["model_state_dict"] = model.state_dict()
    return history

# -----------------------
# 6. Linear evaluation
# -----------------------

def linear_eval_train_val_accuracy(
    model: nn.Module,
    root_dir: str,
    batch_size: int = 256,
    epochs: int = 10,
    lr: float = 1e-3,
    seed: int = 42,
    crop_size: int = 16000
) -> Tuple[float, float]:
    """
    Freeze encoder. Train a single linear layer on train split embeddings.
    Return (train_acc, val_acc).
    """
    set_seed(seed)
    full = QMSAT_Eval(root_dir=root_dir, sample_rate=16000, crop_size=crop_size)
    n = len(full); n_train = int(0.8*n); n_val = n - n_train
    if n_train == 0 or n_val == 0:
        raise RuntimeError(f"Not enough data for linear eval split (n={n}).")
    train_set, val_set = random_split(full, [n_train, n_val], generator=torch.Generator().manual_seed(seed))

    pin = (device.type == "cuda")
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=pin)

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # embedding dimension
    with torch.no_grad():
        x0, _ = next(iter(train_loader))
        emb0 = model.encode(x0.to(device))
        dim = emb0.shape[1]

    clf = nn.Linear(dim, len(LABELS)).to(device)
    opt = torch.optim.Adam(clf.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()

    def get_embeddings(loader):
        embs, ys = [], []
        with torch.no_grad():
            for x, y in loader:
                x = x.to(device)
                z = model.encode(x)
                embs.append(z.cpu())
                ys.append(torch.tensor(y))
        return torch.cat(embs, 0), torch.cat(ys, 0)

    Ztr, Ytr = get_embeddings(train_loader)
    Zva, Yva = get_embeddings(val_loader)
    Ztr = Ztr.to(device); Ytr = Ytr.to(device)
    Zva = Zva.to(device); Yva = Yva.to(device)

    for _ in range(epochs):
        clf.train()
        opt.zero_grad()
        logits = clf(Ztr)
        loss = ce(logits, Ytr)
        loss.backward()
        opt.step()

    clf.eval()
    with torch.no_grad():
        pred_tr = clf(Ztr).argmax(1)
        pred_va = clf(Zva).argmax(1)
        train_acc = (pred_tr == Ytr).float().mean().item()
        val_acc   = (pred_va == Yva).float().mean().item()

    return float(train_acc), float(val_acc)

# -----------------------
# 7. Excel writer
# -----------------------

def write_results_to_excel(
    excel_path: str,
    run_summaries: List[Dict[str, Any]],
    per_epoch_rows: List[Dict[str, Any]]
):
    os.makedirs(os.path.dirname(excel_path), exist_ok=True)
    df_runs = pd.DataFrame(run_summaries)
    df_epochs = pd.DataFrame(per_epoch_rows)
    with pd.ExcelWriter(excel_path, engine="xlsxwriter") as writer:
        df_runs.to_excel(writer, sheet_name="runs", index=False)
        df_epochs.to_excel(writer, sheet_name="per_epoch", index=False)
    print(f"Saved Excel to: {excel_path}")

# -----------------------
# 8. Experiments list (ALL ablations)
# -----------------------

@dataclass
class Experiment:
    name: str
    aug: AugConfig
    model: ModelConfig
    train: TrainConfig
    notes: str = ""
    do_linear_eval: bool = True  # compute accuracies for all runs

def build_ablation_list(baseline: Experiment) -> List[Experiment]:
    base = baseline

    exps: List[Experiment] = [
        base,  # Baseline: all augmentations on + MSE

        # ---- Augmentation ablations
        Experiment(
            name="no_spec_augment",
            aug=AugConfig(**{**asdict(base.aug), "use_spec_augment": False}),
            model=base.model, train=base.train,
            notes="SpecAug (mel masking) OFF"
        ),
        Experiment(
            name="no_pitch_shift",
            aug=AugConfig(**{**asdict(base.aug), "use_pitch_shift": False}),
            model=base.model, train=base.train,
            notes="Pitch shift OFF"
        ),
        Experiment(
            name="no_time_stretch",
            aug=AugConfig(**{**asdict(base.aug), "use_time_stretch": False}),
            model=base.model, train=base.train,
            notes="Time stretch OFF"
        ),
        Experiment(
            name="no_noise",
            aug=AugConfig(**{**asdict(base.aug), "use_noise": False}),
            model=base.model, train=base.train,
            notes="Additive noise OFF"
        ),
        Experiment(
            name="shorter_crop_0p5s",
            aug=AugConfig(**{**asdict(base.aug), "crop_size": int(0.5 * base.aug.sample_rate)}),
            model=base.model, train=base.train,
            notes="Shorter crops (0.5s)"
        ),

        # ---- Model ablations
        Experiment(
            name="proj64",
            aug=base.aug,
            model=ModelConfig(**{**asdict(base.model), "projection_dim": 64}),
            train=base.train,
            notes="Projection 64"
        ),
        Experiment(
            name="proj256",
            aug=base.aug,
            model=ModelConfig(**{**asdict(base.model), "projection_dim": 256}),
            train=base.train,
            notes="Projection 256"
        ),
        Experiment(
            name="no_pretrain",
            aug=base.aug,
            model=ModelConfig(**{**asdict(base.model), "pretrained": False}),
            train=base.train,
            notes="ResNet18 from scratch"
        ),
        Experiment(
            name="freeze_backbone",
            aug=base.aug,
            model=ModelConfig(**{**asdict(base.model), "freeze_backbone": True}),
            train=base.train,
            notes="Freeze backbone"
        ),

        # ---- Loss ablations
        Experiment(
            name="ntxent_temp_0p1",
            aug=base.aug,
            model=base.model,
            train=TrainConfig(**{**asdict(base.train), "loss_type": "ntxent", "temperature": 0.1}),
            notes="NT-Xent (τ=0.1)"
        ),
        Experiment(
            name="ntxent_temp_0p5",
            aug=base.aug,
            model=base.model,
            train=TrainConfig(**{**asdict(base.train), "loss_type": "ntxent", "temperature": 0.5}),
            notes="NT-Xent (τ=0.5)"
        ),

        # ---- Optim ablations
        Experiment(
            name="lr_3e-4",
            aug=base.aug,
            model=base.model,
            train=TrainConfig(**{**asdict(base.train), "lr": 3e-4}),
            notes="Lower LR 3e-4"
        ),
        Experiment(
            name="lr_1e-4",
            aug=base.aug,
            model=base.model,
            train=TrainConfig(**{**asdict(base.train), "lr": 1e-4}),
            notes="Lower LR 1e-4"
        ),
        Experiment(
            name="batch_64",
            aug=base.aug,
            model=base.model,
            train=TrainConfig(**{**asdict(base.train), "batch_size": 64}),
            notes="Smaller batch size 64"
        ),

        # ---- Special: original recipe style
        Experiment(
            name="original_99p75_ablation",
            aug=base.aug,
            model=ModelConfig(**{**asdict(base.model), "projection_dim": 256}),
            train=TrainConfig(**{**asdict(base.train), "epochs": 40, "loss_type": "ntxent", "temperature": 0.1}),
            notes="NT-Xent τ=0.1, proj=256, 40 epochs"
        ),
    ]
    return exps

# -----------------------
# 9. Run all ablations
# -----------------------

def run_all_ablations():
    # TODO: adjust these two paths for your machine
    root_dir = r"C:\Users\muroo\Downloads\archive\ATS-data"   # <--- your dataset root (matches your screenshot)
    out_dir  = r"C:\Users\muroo\Downloads\archive"            # <--- where to save Excel & checkpoints

    excel_path = os.path.join(out_dir, 'smsat_ablation_results.xlsx')

    baseline = Experiment(
        name="baseline_mse_all_aug_on",
        aug=AugConfig(sample_rate=16000, crop_size=16000,
                      use_noise=True, use_time_stretch=True,
                      use_pitch_shift=True, use_spec_augment=True),
        model=ModelConfig(projection_dim=128, pretrained=True, freeze_backbone=False, apply_spec_augment=True),
        train=TrainConfig(epochs=30, batch_size=32, lr=1e-3, weight_decay=1e-4, loss_type='mse', seed=42),
        notes="All augmentations ON (noise, time-stretch, pitch-shift, SpecAug in mel); MSE objective.",
        do_linear_eval=True
    )

    experiments = build_ablation_list(baseline)

    run_summaries: List[Dict[str, Any]] = []
    per_epoch_rows: List[Dict[str, Any]] = []

    for i, exp in enumerate(experiments, start=1):
        print("\n" + "="*100)
        print(f"Experiment {i}/{len(experiments)}: {exp.name}")
        print("="*100)

        # ----- SSL training -----
        hist = train_ssl_one_experiment(
            root_dir=root_dir,
            aug_cfg=exp.aug,
            model_cfg=exp.model,
            train_cfg=exp.train,
            num_workers=0,   # start with 0 to surface issues; switch to 2 after it's stable
            return_model=True
        )

        # Per-epoch rows for Excel
        for ep, (trL, trC, vL, vC, tS) in enumerate(zip(
            hist["train_losses"], hist["train_cosine"],
            hist["val_losses"], hist["val_cosine"],
            hist["epoch_times"]
        ), start=1):
            per_epoch_rows.append({
                "exp_id": i,
                "exp_name": exp.name,
                "epoch": ep,
                "train_loss": trL,
                "train_cosine": trC,
                "val_loss": vL,
                "val_cosine": vC,
                "epoch_time_sec": tS
            })

        best = hist["best_snapshot"]
        total_sec = hist["total_training_time_sec"]
        total_minutes = total_sec / 60.0

        # Save model checkpoint
        ckpt_path = os.path.join(out_dir, f"{exp.name}_encoder.pth")
        torch.save(hist["model"].state_dict(), ckpt_path)
        print(f"Saved checkpoint: {ckpt_path}")

        # ----- Linear evaluation (train & val accuracy) -----
        train_acc, val_acc = (None, None)
        if exp.do_linear_eval:
            print("Running linear evaluation (frozen encoder) to compute accuracy...")
            train_acc, val_acc = linear_eval_train_val_accuracy(
                model=hist["model"].to(device),
                root_dir=root_dir,
                batch_size=256,
                epochs=10,
                lr=1e-3,
                seed=exp.train.seed,
                crop_size=exp.aug.crop_size
            )
            print(f"[Linear Eval] Train Accuracy: {train_acc*100:.2f}% | Val Accuracy: {val_acc*100:.2f}%")

        # ----- Run-level summary -----
        summary = {
            "exp_id": i,
            "exp_name": exp.name,
            "notes": exp.notes,
            # Aug toggles
            "use_noise": exp.aug.use_noise,
            "use_time_stretch": exp.aug.use_time_stretch,
            "use_pitch_shift": exp.aug.use_pitch_shift,
            "use_spec_augment": exp.aug.use_spec_augment,
            "crop_size": exp.aug.crop_size,
            # Model
            "projection_dim": exp.model.projection_dim,
            "pretrained": exp.model.pretrained,
            "freeze_backbone": exp.model.freeze_backbone,
            # Train
            "loss_type": exp.train.loss_type,
            "temperature": exp.train.temperature if exp.train.loss_type == "ntxent" else None,
            "epochs": exp.train.epochs,
            "batch_size": exp.train.batch_size,
            "lr": exp.train.lr,
            "weight_decay": exp.train.weight_decay,
            "seed": exp.train.seed,
            # Best snapshot
            "best_epoch": best.get("best_epoch", None),
            "best_val_loss": best.get("best_val_loss", None),
            "best_val_cosine": best.get("best_val_cosine", None),
            "train_loss_at_best": best.get("train_loss_at_best", None),
            "train_cos_at_best": best.get("train_cos_at_best", None),
            # Timing
            "total_time_sec": total_sec,
            "training_time_minutes": total_minutes,
            # Linear eval accuracies
            "linear_train_acc": train_acc,
            "linear_val_acc": val_acc
        }
        run_summaries.append(summary)

    # ----- Write Excel -----
    write_results_to_excel(excel_path, run_summaries, per_epoch_rows)
    print("\nAll experiments finished.")
    print(f"Excel summary: {excel_path}")
    print("Sheets:")
    print(" - runs: per-experiment summary (includes linear_train_acc, linear_val_acc, training_time_minutes)")
    print(" - per_epoch: full SSL learning curves for every run")

if __name__ == "__main__":
    run_all_ablations()

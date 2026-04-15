import os
import glob
import random
from pathlib import Path

import numpy as np
import pandas as pd

# Conda Auto 환경 라이브러리를 우선 사용해 CUDA/cuDNN 충돌을 줄인다.
conda_prefix = os.environ.get("CONDA_PREFIX")
if conda_prefix:
    conda_lib = str(Path(conda_prefix) / "lib")
    ld_library_path = os.environ.get("LD_LIBRARY_PATH", "")
    ld_parts = [p for p in ld_library_path.split(":") if p]
    if conda_lib not in ld_parts:
        os.environ["LD_LIBRARY_PATH"] = (
            f"{conda_lib}:{ld_library_path}" if ld_library_path else conda_lib
        )

# 기본적으로 GPU 0,1을 사용하도록 설정
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

# 현재 서버 환경에서 cuDNN 라이브러리 충돌이 발생할 수 있으므로 비활성화
torch.backends.cudnn.enabled = False

# ==============================
# 0. Device & Seed
# ==============================
if torch.cuda.is_available():
    gpu_count = torch.cuda.device_count()
    if gpu_count >= 2:
        device_ids = [0, 1]
    else:
        device_ids = [0]
    device = torch.device("cuda:0")
else:
    gpu_count = 0
    device_ids = []
    device = torch.device("cpu")

print("Using device:", device)
print("Visible GPU count:", gpu_count)
if device_ids:
    print("DataParallel device_ids:", device_ids)
print("cuDNN enabled:", torch.backends.cudnn.enabled)

os.makedirs("models_ModernTCN", exist_ok=True)
os.makedirs("results", exist_ok=True)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

# ==============================
# 1. 사용자 보유 데이터 로드
# ==============================
DATA_DIR = Path("dataset_grid")

file_paths = sorted([
    p for p in DATA_DIR.glob("*.csv")
    if not p.name.startswith("Wave_")
])

print("발견된 학습 대상 파일 수:", len(file_paths))
if len(file_paths) == 0:
    raise RuntimeError("dataset_grid 폴더에서 학습할 CSV 파일을 찾지 못했습니다.")

sample_df = pd.read_csv(file_paths[0])
if sample_df.shape[1] < 7:
    raise RuntimeError("CSV는 최소 7개 컬럼(시간 1개 + 6자유도 6개)이 필요합니다.")

time_col = sample_df.columns[0]
dof_cols = list(sample_df.columns[1:7])

print("시간 컬럼:", time_col)
print("사용할 6자유도 컬럼:", dof_cols)

all_arrays = []
file_lengths = []

for p in file_paths:
    df = pd.read_csv(p)

    if list(df.columns[:7]) != list(sample_df.columns[:7]):
        raise RuntimeError(
            f"파일 {p.name} 의 앞 7개 컬럼 구조가 첫 파일과 다릅니다.\n"
            f"기준: {list(sample_df.columns[:7])}\n"
            f"현재: {list(df.columns[:7])}"
        )

    arr = df[dof_cols].to_numpy(dtype=np.float32)
    all_arrays.append(arr)
    file_lengths.append(len(arr))

print("각 파일 길이:", file_lengths)

# ==============================
# 2. 스케일링 정보 계산
# ==============================
concat_all = np.concatenate(all_arrays, axis=0)
data_min = concat_all.min(axis=0)
data_max = concat_all.max(axis=0)
data_range = data_max - data_min
data_range[data_range == 0] = 1.0

print("data_min:", data_min)
print("data_max:", data_max)

def minmax_scale(x, x_min, x_range):
    return (x - x_min) / x_range

def minmax_inverse(x_scaled, x_min, x_range):
    return x_scaled * x_range + x_min

scaled_arrays = [
    minmax_scale(arr, data_min, data_range).astype(np.float32)
    for arr in all_arrays
]

# ==============================
# 3. 윈도우 인덱스 구성
# ==============================
sampling_rate = 10
IN_LEN = 256
OUT_LEN = 256
STRIDE = 16

window_meta = []

for file_idx, arr in enumerate(scaled_arrays):
    n = len(arr)
    max_start = n - IN_LEN - OUT_LEN
    if max_start < 0:
        print(f"[건너뜀] {file_paths[file_idx].name}: 길이가 너무 짧아 윈도우 생성 불가")
        continue

    for s in range(0, max_start + 1, STRIDE):
        window_meta.append((file_idx, s))

window_meta = np.array(window_meta, dtype=np.int64)

if len(window_meta) == 0:
    raise RuntimeError("생성 가능한 윈도우가 없습니다. IN_LEN/OUT_LEN/STRIDE를 조정하세요.")

print("전체 윈도우 개수:", len(window_meta))

perm = np.random.permutation(len(window_meta))
window_meta = window_meta[perm]

train_ratio = 0.7
val_ratio = 0.15

n_total = len(window_meta)
n_train = int(n_total * train_ratio)
n_val = int(n_total * val_ratio)
n_test = n_total - n_train - n_val

meta_train = window_meta[:n_train]
meta_val = window_meta[n_train:n_train + n_val]
meta_test = window_meta[n_train + n_val:]

print(f"Train windows: {len(meta_train)}, Val: {len(meta_val)}, Test: {len(meta_test)}")

# ==============================
# 4. Dataset
# ==============================
class TimeWindowDataset(Dataset):
    def __init__(self, scaled_arrays, meta, in_len=IN_LEN, out_len=OUT_LEN):
        self.scaled_arrays = [torch.from_numpy(arr.astype(np.float32)) for arr in scaled_arrays]
        self.meta = meta.astype(np.int64)
        self.in_len = in_len
        self.out_len = out_len

    def __len__(self):
        return len(self.meta)

    def __getitem__(self, idx):
        file_idx, s = self.meta[idx]
        s = int(s)
        arr = self.scaled_arrays[file_idx]

        e_in = s + self.in_len
        e_out = e_in + self.out_len

        x = arr[s:e_in, :]
        y = arr[e_in:e_out, :]
        return x, y

# ==============================
# 5. ModernTCN Model
# ==============================
class PermuteToChannelsFirst(nn.Module):
    def forward(self, x):
        return x.transpose(1, 2)

class PermuteToTimeFirst(nn.Module):
    def forward(self, x):
        return x.transpose(1, 2)

class ChannelLayerNorm(nn.Module):
    """
    입력: (B, C, T)
    채널 방향 정규화를 위해 (B, T, C)로 바꿔 LayerNorm 후 다시 복귀
    """
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2)
        return x

class ModernTCNBlock(nn.Module):
    """
    ModernTCN 스타일의 블록:
    Depthwise Conv -> LayerNorm -> Pointwise Expansion -> GELU -> Dropout
    -> Pointwise Projection -> Residual
    """
    def __init__(self, channels, kernel_size=31, expansion=2.0, dropout=0.1):
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError("kernel_size는 홀수여야 합니다.")

        hidden_dim = int(channels * expansion)
        padding = kernel_size // 2

        self.dwconv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=channels,
            bias=True,
        )

        self.norm = ChannelLayerNorm(channels)

        self.pwffn = nn.Sequential(
            nn.Conv1d(channels, hidden_dim, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, channels, kernel_size=1, bias=True),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        residual = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwffn(x)
        x = x + residual
        return x

class ModernTCNBackbone(nn.Module):
    """
    입력: (B, T, in_dof)
    출력: (B, T, channels)
    """
    def __init__(
        self,
        in_dof=6,
        channels=128,
        num_blocks=6,
        kernel_size=31,
        expansion=2.0,
        dropout=0.1,
        stem_kernel_size=7,
    ):
        super().__init__()

        if stem_kernel_size % 2 == 0:
            raise ValueError("stem_kernel_size는 홀수여야 합니다.")

        stem_padding = stem_kernel_size // 2

        self.input_proj = nn.Sequential(
            PermuteToChannelsFirst(),
            nn.Conv1d(in_dof, channels, kernel_size=stem_kernel_size, padding=stem_padding, bias=True),
        )

        self.blocks = nn.ModuleList([
            ModernTCNBlock(
                channels=channels,
                kernel_size=kernel_size,
                expansion=expansion,
                dropout=dropout,
            )
            for _ in range(num_blocks)
        ])

        self.out_permute = PermuteToTimeFirst()

    def forward(self, x):
        x = self.input_proj(x)  # (B, C, T)
        for block in self.blocks:
            x = block(x)         # (B, C, T)
        x = self.out_permute(x) # (B, T, C)
        return x

class ModernTCNRegressor(nn.Module):
    """
    입력: (B, IN_LEN, 6)
    출력: (B, OUT_LEN, 6)
    """
    def __init__(
        self,
        in_dof=6,
        out_dof=6,
        in_len=256,
        out_len=256,
        channels=128,
        num_blocks=6,
        kernel_size=31,
        expansion=2.0,
        dropout=0.1,
        stem_kernel_size=7,
        head_hidden_dim=256,
    ):
        super().__init__()

        self.out_len = out_len
        self.out_dof = out_dof

        self.backbone = ModernTCNBackbone(
            in_dof=in_dof,
            channels=channels,
            num_blocks=num_blocks,
            kernel_size=kernel_size,
            expansion=expansion,
            dropout=dropout,
            stem_kernel_size=stem_kernel_size,
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, out_len * out_dof),
            nn.Sigmoid(),
        )

    def forward(self, src):
        x = self.backbone(src)     # (B, T, C)
        x = x.transpose(1, 2)      # (B, C, T)
        x = self.global_pool(x)    # (B, C, 1)
        x = self.head(x)           # (B, OUT_LEN*out_dof)
        x = x.view(-1, self.out_len, self.out_dof)
        return x

# ==============================
# 6. Train / Eval Utilities
# ==============================
class EarlyStopping:
    def __init__(self, patience=10, delta=0.0, path="models_ModernTCN/best_model.pt"):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float("inf")

    def __call__(self, val_loss, save_func):
        score = -val_loss
        if self.best_score is None or score > self.best_score + self.delta:
            self.best_score = score
            save_func(self.path)
            self.val_loss_min = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0

    for src, trg in tqdm(loader, desc="Train", leave=False):
        src = src.to(device)
        trg = trg.to(device)

        optimizer.zero_grad()
        pred = model(src)
        loss = criterion(pred, trg)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / max(1, len(loader))

@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0.0

    for src, trg in tqdm(loader, desc="Valid/Test", leave=False):
        src = src.to(device)
        trg = trg.to(device)

        pred = model(src)
        loss = criterion(pred, trg)
        total_loss += loss.item()

    return total_loss / max(1, len(loader))

# ==============================
# 7. 고정 파라미터 설정
# ==============================
MODEL_CONFIG = {
    "model_type": "ModernTCN",
    "in_dof": 6,
    "out_dof": 6,
    "in_len": IN_LEN,
    "out_len": OUT_LEN,
    "channels": 128,
    "num_blocks": 6,
    "kernel_size": 31,
    "expansion": 2.0,
    "dropout": 0.1,
    "stem_kernel_size": 7,
    "head_hidden_dim": 256,
}

TRAIN_CONFIG = {
    "batch_size": 64,
    "learning_rate": 1e-3,
    "max_epochs": 100,
    "patience": 10,
}

print("MODEL_CONFIG:", MODEL_CONFIG)
print("TRAIN_CONFIG:", TRAIN_CONFIG)

# ==============================
# 8. DataLoader
# ==============================
train_ds = TimeWindowDataset(scaled_arrays, meta_train, IN_LEN, OUT_LEN)
val_ds   = TimeWindowDataset(scaled_arrays, meta_val,   IN_LEN, OUT_LEN)
test_ds  = TimeWindowDataset(scaled_arrays, meta_test,  IN_LEN, OUT_LEN)

train_loader = DataLoader(
    train_ds,
    batch_size=TRAIN_CONFIG["batch_size"],
    shuffle=True,
    drop_last=True,
)
val_loader = DataLoader(
    val_ds,
    batch_size=TRAIN_CONFIG["batch_size"],
    shuffle=False,
    drop_last=False,
)
test_loader = DataLoader(
    test_ds,
    batch_size=TRAIN_CONFIG["batch_size"],
    shuffle=False,
    drop_last=False,
)

# ==============================
# 9. Model 생성
# ==============================
model = ModernTCNRegressor(
    in_dof=MODEL_CONFIG["in_dof"],
    out_dof=MODEL_CONFIG["out_dof"],
    in_len=MODEL_CONFIG["in_len"],
    out_len=MODEL_CONFIG["out_len"],
    channels=MODEL_CONFIG["channels"],
    num_blocks=MODEL_CONFIG["num_blocks"],
    kernel_size=MODEL_CONFIG["kernel_size"],
    expansion=MODEL_CONFIG["expansion"],
    dropout=MODEL_CONFIG["dropout"],
    stem_kernel_size=MODEL_CONFIG["stem_kernel_size"],
    head_hidden_dim=MODEL_CONFIG["head_hidden_dim"],
)

if len(device_ids) >= 2:
    model = nn.DataParallel(model, device_ids=device_ids)

model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=TRAIN_CONFIG["learning_rate"])
criterion = nn.MSELoss()

model_save_path = "models_ModernTCN/moderntcn_dataset_grid_best_full.pt"
early_stopping = EarlyStopping(
    patience=TRAIN_CONFIG["patience"],
    path=model_save_path
)

def save_full_checkpoint(path):
    state_dict = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
    checkpoint = {
        "model_state_dict": state_dict,
        "model_config": MODEL_CONFIG,
        "train_config": TRAIN_CONFIG,
        "time_col": time_col,
        "dof_cols": dof_cols,
        "data_min": data_min.astype(np.float32),
        "data_max": data_max.astype(np.float32),
        "data_range": data_range.astype(np.float32),
        "sampling_rate": sampling_rate,
        "stride": STRIDE,
        "file_paths_used": [str(p) for p in file_paths],
        "device_ids_used": device_ids,
    }
    torch.save(checkpoint, path)

# ==============================
# 10. 학습
# ==============================
history = {
    "epoch": [],
    "train_loss": [],
    "val_loss": [],
}

for epoch in range(1, TRAIN_CONFIG["max_epochs"] + 1):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
    val_loss = evaluate(model, val_loader, criterion)

    history["epoch"].append(epoch)
    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)

    print(f"Epoch {epoch:03d} | Train Loss: {train_loss:.8f} | Val Loss: {val_loss:.8f}")

    early_stopping(val_loss, save_full_checkpoint)
    if early_stopping.early_stop:
        print(f"Early stopping at epoch {epoch}")
        break

print("최적 모델 저장 완료:", model_save_path)

# ==============================
# 11. 최적 모델 다시 로드 후 Test 평가
# ==============================
checkpoint = torch.load(model_save_path, map_location=device)

best_model = ModernTCNRegressor(
    in_dof=checkpoint["model_config"]["in_dof"],
    out_dof=checkpoint["model_config"]["out_dof"],
    in_len=checkpoint["model_config"]["in_len"],
    out_len=checkpoint["model_config"]["out_len"],
    channels=checkpoint["model_config"]["channels"],
    num_blocks=checkpoint["model_config"]["num_blocks"],
    kernel_size=checkpoint["model_config"]["kernel_size"],
    expansion=checkpoint["model_config"]["expansion"],
    dropout=checkpoint["model_config"]["dropout"],
    stem_kernel_size=checkpoint["model_config"]["stem_kernel_size"],
    head_hidden_dim=checkpoint["model_config"]["head_hidden_dim"],
)

if len(device_ids) >= 2:
    best_model = nn.DataParallel(best_model, device_ids=device_ids)

best_model = best_model.to(device)

state = checkpoint["model_state_dict"]
if isinstance(best_model, nn.DataParallel):
    best_model.module.load_state_dict(state)
else:
    best_model.load_state_dict(state)

test_loss = evaluate(best_model, test_loader, criterion)
print(f"Best Model Test MSE: {test_loss:.8f}")

# ==============================
# 12. 학습 곡선 저장
# ==============================
hist_df = pd.DataFrame(history)
hist_df.to_csv("results/moderntcn_dataset_grid_history.csv", index=False)

plt.figure(figsize=(8, 5))
plt.plot(history["epoch"], history["train_loss"], label="Train Loss")
plt.plot(history["epoch"], history["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("ModernTCN Training History")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/moderntcn_dataset_grid_history.png", dpi=150)
plt.show()
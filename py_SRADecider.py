import os
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

# 현재 서버 환경에서 cuDNN 충돌 가능성을 줄이기 위해 비활성화
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

os.makedirs("models_SRADecider", exist_ok=True)
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
# 5. SRA Decider Hybrid Model
# ==============================
class ChannelLayerNorm(nn.Module):
    """
    입력: (B, C, T)
    채널 방향 LayerNorm
    """
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = x.transpose(1, 2)
        return x

def torch_rankdata(x: torch.Tensor) -> torch.Tensor:
    """
    x: (B, T, C)
    시간축 T에 대해 rank를 계산한다.
    tie가 거의 없는 연속값 시계열을 가정한 간단 구현.
    """
    order = torch.argsort(x, dim=1)
    ranks = torch.argsort(order, dim=1).float()
    return ranks

def batch_spearman_mean_abs_offdiag(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    x: (B, T, C)
    각 배치 샘플별로 채널 간 스피어만 상관계수 행렬을 구하고,
    off-diagonal 절대값 평균을 반환한다.
    반환: (B,)
    """
    B, T, C = x.shape
    ranks = torch_rankdata(x)                         # (B, T, C)
    ranks = ranks - ranks.mean(dim=1, keepdim=True)  # 중심화

    std = torch.sqrt((ranks ** 2).mean(dim=1, keepdim=True) + eps)
    z = ranks / std                                  # (B, T, C)

    corr = torch.matmul(z.transpose(1, 2), z) / T    # (B, C, C)

    eye = torch.eye(C, device=x.device).unsqueeze(0) # (1, C, C)
    offdiag_mask = 1.0 - eye

    abs_offdiag = corr.abs() * offdiag_mask
    denom = C * (C - 1)

    score = abs_offdiag.sum(dim=(1, 2)) / denom      # (B,)
    return score

class CISharedEncoder(nn.Module):
    """
    CI 경로: 각 변수를 독립 단변수 시계열로 간주하고 공유 가중치 적용
    입력: (B, T, C)
    출력: (B, T, hidden)
    """
    def __init__(self, hidden_channels=64, num_blocks=4, kernel_size=31, dropout=0.1):
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError("kernel_size는 홀수여야 합니다.")

        padding = kernel_size // 2
        self.stem = nn.Conv1d(1, hidden_channels, kernel_size=7, padding=3, bias=True)

        blocks = []
        for _ in range(num_blocks):
            blocks.append(nn.Sequential(
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size=kernel_size,
                          padding=padding, groups=hidden_channels, bias=True),
                ChannelLayerNorm(hidden_channels),
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1, bias=True),
                nn.GELU(),
                nn.Dropout(dropout),
            ))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        B, T, C = x.shape
        x = x.permute(0, 2, 1).contiguous()      # (B, C, T)
        x = x.view(B * C, 1, T)                  # (B*C, 1, T)

        x = self.stem(x)
        for block in self.blocks:
            residual = x
            x = block(x)
            x = x + residual

        x = x.transpose(1, 2)                    # (B*C, T, hidden)
        hidden = x.size(-1)
        x = x.view(B, C, T, hidden)              # (B, C, T, hidden)
        x = x.permute(0, 2, 1, 3).contiguous()   # (B, T, C, hidden)
        return x

class CISharedHead(nn.Module):
    """
    CI 경로 공통 헤드
    입력: (B, T, C, hidden)
    출력: (B, OUT_LEN, C)
    """
    def __init__(self, hidden_channels=64, out_len=256, head_hidden_dim=128, dropout=0.1):
        super().__init__()
        self.out_len = out_len
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_channels, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, out_len),
            nn.Sigmoid(),
        )

    def forward(self, x):
        B, T, C, H = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()   # (B, C, H, T)
        x = x.view(B * C, H, T)                  # (B*C, H, T)
        x = self.global_pool(x)                  # (B*C, H, 1)
        x = self.head(x)                         # (B*C, OUT_LEN)
        x = x.view(B, C, self.out_len)           # (B, C, OUT_LEN)
        x = x.permute(0, 2, 1).contiguous()      # (B, OUT_LEN, C)
        return x

class CDEncoder(nn.Module):
    """
    CD 경로: 모든 변수를 함께 다변수로 처리
    입력: (B, T, C)
    출력: (B, T, hidden)
    """
    def __init__(self, in_dof=6, hidden_channels=128, num_blocks=5, kernel_size=31, dropout=0.1):
        super().__init__()

        if kernel_size % 2 == 0:
            raise ValueError("kernel_size는 홀수여야 합니다.")

        padding = kernel_size // 2
        self.stem = nn.Conv1d(in_dof, hidden_channels, kernel_size=7, padding=3, bias=True)

        blocks = []
        for _ in range(num_blocks):
            blocks.append(nn.Sequential(
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size=kernel_size,
                          padding=padding, groups=1, bias=True),
                ChannelLayerNorm(hidden_channels),
                nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1, bias=True),
                nn.GELU(),
                nn.Dropout(dropout),
            ))
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        x = x.transpose(1, 2)                   # (B, C, T)
        x = self.stem(x)                        # (B, hidden, T)
        for block in self.blocks:
            residual = x
            x = block(x)
            x = x + residual
        x = x.transpose(1, 2)                   # (B, T, hidden)
        return x

class CDHead(nn.Module):
    """
    CD 경로 헤드
    입력: (B, T, hidden)
    출력: (B, OUT_LEN, C)
    """
    def __init__(self, hidden_channels=128, out_len=256, out_dof=6, head_hidden_dim=256, dropout=0.1):
        super().__init__()
        self.out_len = out_len
        self.out_dof = out_dof
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_channels, head_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_hidden_dim, out_len * out_dof),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.transpose(1, 2)                 # (B, hidden, T)
        x = self.global_pool(x)               # (B, hidden, 1)
        x = self.head(x)                      # (B, out_len*out_dof)
        x = x.view(-1, self.out_len, self.out_dof)
        return x

class SRADeciderHybridRegressor(nn.Module):
    """
    SRA Decider 기반 동적 CI/CD 하이브리드 모델.
    스피어만 기반 점수로 CI와 CD 경로를 soft-gating 한다.
    """
    def __init__(
        self,
        in_dof=6,
        out_dof=6,
        in_len=256,
        out_len=256,
        ci_hidden_channels=64,
        ci_num_blocks=4,
        cd_hidden_channels=128,
        cd_num_blocks=5,
        kernel_size=31,
        dropout=0.1,
        ci_head_hidden_dim=128,
        cd_head_hidden_dim=256,
        sra_threshold=0.35,
        sra_temperature=0.05,
    ):
        super().__init__()

        self.in_dof = in_dof
        self.out_dof = out_dof
        self.in_len = in_len
        self.out_len = out_len
        self.sra_threshold = float(sra_threshold)
        self.sra_temperature = float(sra_temperature)

        self.ci_encoder = CISharedEncoder(
            hidden_channels=ci_hidden_channels,
            num_blocks=ci_num_blocks,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.ci_head = CISharedHead(
            hidden_channels=ci_hidden_channels,
            out_len=out_len,
            head_hidden_dim=ci_head_hidden_dim,
            dropout=dropout,
        )

        self.cd_encoder = CDEncoder(
            in_dof=in_dof,
            hidden_channels=cd_hidden_channels,
            num_blocks=cd_num_blocks,
            kernel_size=kernel_size,
            dropout=dropout,
        )
        self.cd_head = CDHead(
            hidden_channels=cd_hidden_channels,
            out_len=out_len,
            out_dof=out_dof,
            head_hidden_dim=cd_head_hidden_dim,
            dropout=dropout,
        )

    def compute_gate(self, src):
        sra_score = batch_spearman_mean_abs_offdiag(src)   # (B,)
        gate = torch.sigmoid((sra_score - self.sra_threshold) / self.sra_temperature)
        gate = gate.view(-1, 1, 1)                         # (B,1,1)
        return gate, sra_score

    def forward(self, src, return_gate=False):
        ci_feat = self.ci_encoder(src)                     # (B,T,C,Hci)
        ci_out = self.ci_head(ci_feat)                     # (B,OUT_LEN,C)

        cd_feat = self.cd_encoder(src)                     # (B,T,Hcd)
        cd_out = self.cd_head(cd_feat)                     # (B,OUT_LEN,C)

        gate, sra_score = self.compute_gate(src)
        out = gate * cd_out + (1.0 - gate) * ci_out

        if return_gate:
            return out, gate.squeeze(-1).squeeze(-1), sra_score
        return out

# ==============================
# 6. Train / Eval Utilities
# ==============================
class EarlyStopping:
    def __init__(self, patience=10, delta=0.0, path="models_SRADecider/best_model.pt"):
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

@torch.no_grad()
def evaluate_with_gate_stats(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    gate_list = []
    sra_list = []

    for src, trg in tqdm(loader, desc="Eval+Gate", leave=False):
        src = src.to(device)
        trg = trg.to(device)

        pred, gate, sra = model(src, return_gate=True)
        loss = criterion(pred, trg)

        total_loss += loss.item()
        gate_list.append(gate.detach().cpu())
        sra_list.append(sra.detach().cpu())

    gate_all = torch.cat(gate_list).numpy() if gate_list else np.array([])
    sra_all = torch.cat(sra_list).numpy() if sra_list else np.array([])
    return total_loss / max(1, len(loader)), gate_all, sra_all

# ==============================
# 7. 고정 파라미터 설정
# ==============================
MODEL_CONFIG = {
    "model_type": "SRADeciderHybrid",
    "in_dof": 6,
    "out_dof": 6,
    "in_len": IN_LEN,
    "out_len": OUT_LEN,
    "ci_hidden_channels": 64,
    "ci_num_blocks": 4,
    "cd_hidden_channels": 128,
    "cd_num_blocks": 5,
    "kernel_size": 31,
    "dropout": 0.1,
    "ci_head_hidden_dim": 128,
    "cd_head_hidden_dim": 256,
    "sra_threshold": 0.35,
    "sra_temperature": 0.05,
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
model = SRADeciderHybridRegressor(
    in_dof=MODEL_CONFIG["in_dof"],
    out_dof=MODEL_CONFIG["out_dof"],
    in_len=MODEL_CONFIG["in_len"],
    out_len=MODEL_CONFIG["out_len"],
    ci_hidden_channels=MODEL_CONFIG["ci_hidden_channels"],
    ci_num_blocks=MODEL_CONFIG["ci_num_blocks"],
    cd_hidden_channels=MODEL_CONFIG["cd_hidden_channels"],
    cd_num_blocks=MODEL_CONFIG["cd_num_blocks"],
    kernel_size=MODEL_CONFIG["kernel_size"],
    dropout=MODEL_CONFIG["dropout"],
    ci_head_hidden_dim=MODEL_CONFIG["ci_head_hidden_dim"],
    cd_head_hidden_dim=MODEL_CONFIG["cd_head_hidden_dim"],
    sra_threshold=MODEL_CONFIG["sra_threshold"],
    sra_temperature=MODEL_CONFIG["sra_temperature"],
)

if len(device_ids) >= 2:
    model = nn.DataParallel(model, device_ids=device_ids)

model = model.to(device)

optimizer = optim.Adam(model.parameters(), lr=TRAIN_CONFIG["learning_rate"])
criterion = nn.MSELoss()

model_save_path = "models_SRADecider/sra_decider_dataset_grid_best_full.pt"
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

best_model = SRADeciderHybridRegressor(
    in_dof=checkpoint["model_config"]["in_dof"],
    out_dof=checkpoint["model_config"]["out_dof"],
    in_len=checkpoint["model_config"]["in_len"],
    out_len=checkpoint["model_config"]["out_len"],
    ci_hidden_channels=checkpoint["model_config"]["ci_hidden_channels"],
    ci_num_blocks=checkpoint["model_config"]["ci_num_blocks"],
    cd_hidden_channels=checkpoint["model_config"]["cd_hidden_channels"],
    cd_num_blocks=checkpoint["model_config"]["cd_num_blocks"],
    kernel_size=checkpoint["model_config"]["kernel_size"],
    dropout=checkpoint["model_config"]["dropout"],
    ci_head_hidden_dim=checkpoint["model_config"]["ci_head_hidden_dim"],
    cd_head_hidden_dim=checkpoint["model_config"]["cd_head_hidden_dim"],
    sra_threshold=checkpoint["model_config"]["sra_threshold"],
    sra_temperature=checkpoint["model_config"]["sra_temperature"],
)

if len(device_ids) >= 2:
    best_model = nn.DataParallel(best_model, device_ids=device_ids)

best_model = best_model.to(device)

state = checkpoint["model_state_dict"]
if isinstance(best_model, nn.DataParallel):
    best_model.module.load_state_dict(state)
else:
    best_model.load_state_dict(state)

test_loss, test_gate_all, test_sra_all = evaluate_with_gate_stats(best_model, test_loader, criterion)
print(f"Best Model Test MSE: {test_loss:.8f}")
if len(test_gate_all) > 0:
    print(f"Test gate mean (CD 비중): {test_gate_all.mean():.6f}")
    print(f"Test SRA mean: {test_sra_all.mean():.6f}")

# ==============================
# 12. 학습 곡선 및 게이트 통계 저장
# ==============================
hist_df = pd.DataFrame(history)
hist_df.to_csv("results/sra_decider_dataset_grid_history.csv", index=False)

plt.figure(figsize=(8, 5))
plt.plot(history["epoch"], history["train_loss"], label="Train Loss")
plt.plot(history["epoch"], history["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.title("SRA Decider Training History")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("results/sra_decider_dataset_grid_history.png", dpi=150)
plt.show()

if len(test_gate_all) > 0:
    gate_stats_df = pd.DataFrame({
        "gate_cd_weight": test_gate_all,
        "sra_score": test_sra_all,
    })
    gate_stats_df.to_csv("results/sra_decider_test_gate_stats.csv", index=False)
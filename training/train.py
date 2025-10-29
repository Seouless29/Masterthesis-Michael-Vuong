import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import sys, os, math
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from model.model_uniform import UniformGaborCNN

from torch.optim import SGD
from tqdm.auto import tqdm

# configure path
BASE_DIR = Path(__file__).resolve().parent          
DATA = (BASE_DIR / ".." / "data_virus").resolve()   


# load data
def maybe_load(path: Path):
    p = Path(path)
    if not p.exists():
        print(f"[WARN] Not found: {p}")
        return None
    return torch.load(str(p), map_location="cpu")

train_x = maybe_load(DATA / "train_x.pt")
train_y = maybe_load(DATA / "train_y.pt")
val_x   = maybe_load(DATA / "val_x.pt")
val_y   = maybe_load(DATA / "val_y.pt")
test_x  = maybe_load(DATA / "test_x.pt")
test_y  = maybe_load(DATA / "test_y.pt")

assert train_x is not None, f"Missing train_x.pt at {DATA / 'train_x.pt'}"
assert train_x.ndim == 4, f"Expected (N,C,H,W), got {train_x.shape}"

# alexnet uses normalize because it is already pretrained whereas 
# the other two models use standardize, because we are training from scratch
NORMALIZE_MODE = "standardize"   # options: "minmax" or "standardize"
if NORMALIZE_MODE == "minmax":
    tmin, tmax = train_x.min(), train_x.max()
    eps = 1e-12
    scale = (tmax - tmin).clamp_min(eps)

    def transform(x): return (x - tmin) / scale

elif NORMALIZE_MODE == "standardize":
    mean = train_x.mean()
    std  = train_x.std().clamp_min(1e-12)

    def transform(x): return (x - mean) / std

else:
    raise ValueError("NORMALIZE_MODE must be 'minmax' or 'standardize'")

train_x = transform(train_x).to(torch.float32)
if val_x  is not None: val_x  = transform(val_x).to(torch.float32)
if test_x is not None: test_x = transform(test_x).to(torch.float32)

print(f"[{NORMALIZE_MODE}] train range: {train_x.min().item():.3f}, {train_x.max().item():.3f}")
print(f"[{NORMALIZE_MODE}] train mean/std: {train_x.mean().item():.3f}, {train_x.std().item():.3f}")
print(f"[{NORMALIZE_MODE}] val range: {val_x.min().item():.3f}, {val_x.max().item():.3f}" if val_x is not None else "No validation set")
print(f"[{NORMALIZE_MODE}] test range: {test_x.min().item():.3f}, {test_x.max().item():.3f}" if test_x is not None else "No test set")

# create dataloaders
class TensorImages(Dataset):
    def __init__(self, images: torch.Tensor, labels: torch.Tensor | None = None):
        self.x = images
        self.y = labels
        if self.y is not None:
            assert len(self.x) == len(self.y), "images/labels length mismatch"
    def __len__(self): return self.x.shape[0]
    def __getitem__(self, i):
        xi = self.x[i]
        if self.y is None: return xi
        return xi, self.y[i].long()

train_ds = TensorImages(train_x, train_y)
val_ds   = TensorImages(val_x, val_y)   if val_x  is not None else None
test_ds  = TensorImages(test_x, test_y) if test_x is not None else None

def make_loader(ds, batch_size=64, shuffle=False):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, pin_memory=True, drop_last=False)

train_loader = make_loader(train_ds, batch_size=64, shuffle=True)
val_loader   = make_loader(val_ds, batch_size=128) if val_ds  else None
test_loader  = make_loader(test_ds, batch_size=128) if test_ds else None

# logs. literally just copies the tdqm output to a file
LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
log_path = os.path.join(LOG_DIR, "train_log.txt")
log_file = open(log_path, "w", buffering=1) 

def log(msg: str):
    print(msg)
    print(msg, file=log_file)

# training hyperparameters
criterion = nn.CrossEntropyLoss()
DEVICE   = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS   = 100
LR       = 0.01
WD_MAIN  = 1e-4
WD_GABOR = 0.0 # not explored
GABOR_UPDATE_EVERY = 1 # if set to none then it is once per epoch, else every N batches. so here it is set to update each batch. this also has not been explored extensively.
log(f"Using device: {DEVICE}")
SAVE_DIR = "checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

model = UniformGaborCNN().to(DEVICE)

# create two optimizer in order to control the gabor updates

gabor_params, main_params = [], []
for name, p in model.named_parameters():
    if not p.requires_grad: 
        continue
    if name.startswith("gabor.sigma") or name.startswith("gabor.frequency"):
        gabor_params.append(p)
    else:
        main_params.append(p)

opt_main  = SGD(main_params,  lr=LR, momentum=0.9, weight_decay=WD_MAIN)
opt_gabor = SGD(gabor_params, lr=LR, momentum=0.9, weight_decay=WD_GABOR)

def step_main(x, y):
    model.gabor.set_freeze(True)
    for p in gabor_params: p.requires_grad_(False)
    opt_main.zero_grad(set_to_none=True)
    logits = model(x); loss = criterion(logits, y)
    loss.backward(); opt_main.step()
    return logits, loss

def step_gabor(x, y):
    model.gabor.set_freeze(False)
    for p in gabor_params: p.requires_grad_(True)
    opt_main.zero_grad(set_to_none=True)
    opt_gabor.zero_grad(set_to_none=True)
    logits = model(x); loss = criterion(logits, y)
    loss.backward(); opt_gabor.step()
    model.gabor.set_freeze(True)
    for p in gabor_params: p.requires_grad_(False)
    return logits, loss

@torch.no_grad()
def evaluate(loader):
    model.eval()
    total_loss, total_correct, total = 0.0, 0, 0
    pbar = tqdm(loader, desc="eval", leave=False)  
    for x, y in pbar:
        x, y = x.to(DEVICE), y.to(DEVICE)
        logits = model(x); loss = criterion(logits, y)
        total_loss += loss.item() * y.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
        pbar.set_postfix(loss=total_loss/total, acc=total_correct/total)
    return total_loss/total, total_correct/total

best_val_acc = -math.inf

# main training loop
for epoch in range(1, EPOCHS+1):
    log(f"\nEpoch {epoch}/{EPOCHS}")
    model.train()
    total_loss, total_correct, total = 0.0, 0, 0
    pbar = tqdm(enumerate(train_loader, 1), total=len(train_loader), desc="train")  

    last_batch_for_gabor = None

    for b, (x,y) in pbar:
        x, y = x.to(DEVICE), y.to(DEVICE)
        if GABOR_UPDATE_EVERY is None:
            logits, loss = step_main(x, y)
            last_batch_for_gabor = (x, y)
        else:
            logits, loss = step_main(x, y)
            if b % GABOR_UPDATE_EVERY == 0:
                _ = step_gabor(x, y)

        total_loss += loss.item() * y.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
        total += y.size(0)
        pbar.set_postfix(loss=total_loss/total, acc=total_correct/total)

    # this is where we update the gabor parameters
    if GABOR_UPDATE_EVERY is None and last_batch_for_gabor:
        x, y = last_batch_for_gabor
        _ = step_gabor(x, y)

    # validation
    if val_loader is not None:
        val_loss, val_acc = evaluate(val_loader)
        log(f"  train | loss {total_loss/total:.4f}  acc {total_correct/total:.4f}")
        log(f"  val   | loss {val_loss:.4f}        acc {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = os.path.join(SAVE_DIR, f"best_epoch{epoch}.pt")
            torch.save({
                "epoch": epoch,
                "model": model.state_dict(),
                "opt_main": opt_main.state_dict(),
                "opt_gabor": opt_gabor.state_dict(),
                "val_acc": val_acc,
            }, ckpt_path)
            log(f"saved {ckpt_path}")
    else:
        log(f"  train | loss {total_loss/total:.4f}  acc {total_correct/total:.4f}")


# we save training accuracy and validation accuracy as well as the best performing model (which is in our case defined by highest validation accuracy)
log_file.close()
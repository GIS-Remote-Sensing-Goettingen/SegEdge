import os

from torch.optim.lr_scheduler import OneCycleLR

# PREVENT CPU HANGS
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
# PREVENT MEMORY FRAGMENTATION
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoImageProcessor, AutoModel
from tifffile import imread
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.mask import mask
from shapely.geometry import box
from tqdm import tqdm
import gc
import concurrent.futures
import math


# ==========================================
# 1. OPTIMIZER & UTILS (NEW: MUON & EARLY STOPPING)
# ==========================================

def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power of a matrix G.
    Used by Muon optimizer for orthogonalization.
    """
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps)  # ensure top singular value <= 1
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1):
        X = X.T
    return X.to(G.dtype)


class Muon(optim.Optimizer):
    """
    Muon - MomentUm Orthogonalized Optimizer.
    Currently SOTA for training large models/transformers.
    """

    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5, adamw_params=None, adamw_lr=1e-4,
                 adamw_betas=(0.9, 0.95), adamw_wd=0.01):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
                        adamw_params=adamw_params, adamw_lr=adamw_lr, adamw_betas=adamw_betas, adamw_wd=adamw_wd)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            # Handle AdamW params (biases, embeddings, 1D tensors)
            if group['adamw_params'] is not None:
                self._step_adamw(group)

            # Handle Muon params (2D+ weights)
            lr = group['lr']
            momentum = group['momentum']
            nesterov = group['nesterov']
            ns_steps = group['ns_steps']

            for p in group['params']:
                if p.grad is None: continue
                g = p.grad
                if g.ndim > 2:
                    g = g.view(g.size(0), -1)

                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)

                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)

                if nesterov:
                    g = g.add(buf, alpha=momentum)
                else:
                    g = buf

                # Orthogonalize update
                g = zeropower_via_newtonschulz5(g, steps=ns_steps)

                if g.shape != p.shape:
                    g = g.view_as(p)

                p.data.add_(g, alpha=-lr)

    def _step_adamw(self, group):
        lr = group['adamw_lr']
        beta1, beta2 = group['adamw_betas']
        wd = group['adamw_wd']
        eps = 1e-8

        for p in group['adamw_params']:
            if p.grad is None: continue
            g = p.grad
            state = self.state[p]

            if 'step' not in state:
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p)
                state['exp_avg_sq'] = torch.zeros_like(p)

            state['step'] += 1
            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']

            exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1 - beta2)

            denom = exp_avg_sq.sqrt().add_(eps)
            step_size = lr * math.sqrt(1 - beta2 ** state['step']) / (1 - beta1 ** state['step'])

            p.data.mul_(1 - lr * wd)  # Weight decay
            p.data.addcdiv_(exp_avg, denom, value=-step_size)


class EarlyStopping:
    """Stops training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=5, min_delta=0.001, path='checkpoint.pth'):
        self.patience = patience
        self.min_delta = min_delta
        self.path = path
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), self.path)
        print(f'Validation loss decreased. Saving model to {self.path}')


# ==========================================
# 2. MODEL ARCHITECTURE
# ==========================================
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[-2:] != skip.shape[-2:]:
            skip = F.interpolate(skip, size=x.shape[-2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class DinoUNet(nn.Module):
    def __init__(self, dinov3_feature_channels, unet_decoder_channels, num_classes):
        super(DinoUNet, self).__init__()
        self.dinov3_feature_channels = dinov3_feature_channels
        self.decoder_channels = unet_decoder_channels
        self.bottleneck = ConvBlock(dinov3_feature_channels[-1], self.decoder_channels[0])
        self.up_blocks = nn.ModuleList()

        skip_connection_channels = self.dinov3_feature_channels[:-1][::-1] + [3]
        in_channels = self.decoder_channels[0]

        for i, skip_channels in enumerate(skip_connection_channels):
            out_channels = self.decoder_channels[i + 1] if i + 1 < len(self.decoder_channels) else \
                self.decoder_channels[-1]
            self.up_blocks.append(UpBlock(in_channels, skip_channels, out_channels))
            in_channels = out_channels

        self.final_conv = nn.Conv2d(self.decoder_channels[-1], num_classes, kernel_size=1)

    def forward(self, image, dinov3_features):
        features_reversed = dinov3_features[::-1]
        bottleneck_input = features_reversed[0]
        x = self.bottleneck(bottleneck_input)
        skip_connections = [features_reversed[1], features_reversed[2], features_reversed[3], image]
        for i, up_block in enumerate(self.up_blocks):
            x = up_block(x, skip_connections[i])
        return self.final_conv(x)


# ==========================================
# 3. DATA PROCESSING & HELPERS
# ==========================================

def extract_multiscale_features(image_hw3, model, processor, device, layers, ps=14):
    inputs = processor(images=image_hw3, return_tensors="pt", do_resize=False, do_center_crop=False).to(device)
    R = getattr(model.config, "num_register_tokens", 0)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
        hidden_states = out.hidden_states
    _, _, Hproc, Wproc = inputs["pixel_values"].shape
    feature_maps = []
    for layer_idx in layers:
        layer_output = hidden_states[layer_idx]
        patch_tokens = layer_output[:, 1 + R:, :]
        Hp, Wp = Hproc // ps, Wproc // ps
        N, C = patch_tokens.shape[1], patch_tokens.shape[2]
        feats = patch_tokens.reshape(1, Hp, Wp, C).permute(0, 3, 1, 2)
        feature_maps.append(feats.squeeze(0).cpu())
    return feature_maps


def subset_label_to_image_bounds(img_path, lab_path):
    with rasterio.open(img_path) as src_img:
        img_bounds = src_img.bounds
        img_meta = src_img.meta.copy()
        img_crs = src_img.crs
        H, W = src_img.shape
    with rasterio.open(lab_path) as src_lab:
        if src_lab.crs == img_crs:
            geom = [box(*img_bounds).__geo_interface__]
            out_image, _ = mask(src_lab, geom, crop=True)
            if out_image.shape[1] != H or out_image.shape[2] != W:
                t_lbl = torch.from_numpy(out_image).float().unsqueeze(0)
                t_lbl = F.interpolate(t_lbl, size=(H, W), mode='nearest')
                labels_aligned = t_lbl.squeeze(0).squeeze(0).numpy()
            else:
                labels_aligned = out_image[0]
        else:
            new_meta = img_meta.copy()
            new_meta.update(dtype=src_lab.dtypes[0], count=1)
            with rasterio.io.MemoryFile() as mem:
                with mem.open(**new_meta) as dst:
                    reproject(
                        source=rasterio.band(src_lab, 1),
                        destination=rasterio.band(dst, 1),
                        src_transform=src_lab.transform,
                        src_crs=src_lab.crs,
                        dst_transform=img_meta["transform"],
                        dst_crs=img_crs,
                        dst_width=img_meta["width"],
                        dst_height=img_meta["height"],
                        resampling=Resampling.nearest,
                    )
                    labels_aligned = dst.read(1)
    return labels_aligned


def _check_single_file(file_path):
    try:
        torch.load(file_path, weights_only=False, map_location='cpu')
        return None
    except Exception:
        return file_path


def verify_and_clean_dataset_fast(output_dir, num_workers=None):
    print(f"--- FAST VERIFICATION ({output_dir}) ---")
    files = glob.glob(os.path.join(output_dir, "*.pt"))
    if not files:
        print("No files found.")
        return
    if num_workers is None: num_workers = os.cpu_count()

    corrupted_files = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_check_single_file, f) for f in files]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(files), desc="Verifying"):
            result = future.result()
            if result is not None:
                corrupted_files.append(result)

    if corrupted_files:
        print(f"Found {len(corrupted_files)} corrupted files. Deleting...")
        for f in corrupted_files:
            try:
                os.remove(f)
            except OSError:
                pass
    else:
        print("All files verified successfully.")
    print("------------------------------------------\n")


def prepare_data_tiles(img_dir, label_path, output_dir, model_name, layers, device, tile_size=512):
    print("--- PHASE 1: TILING & PRE-COMPUTING ---")
    os.makedirs(output_dir, exist_ok=True)
    existing = glob.glob(os.path.join(output_dir, "*.pt"))
    if len(existing) > 0: print(f"[INFO] Found {len(existing)} existing tiles.")

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval().to(device)
    image_paths = glob.glob(os.path.join(img_dir, "*.tif"))
    ps = 14 if 'vitl14' in model_name else 16

    for img_path in tqdm(image_paths, desc="Processing Large Images"):
        basename = os.path.splitext(os.path.basename(img_path))[0]
        torch.cuda.empty_cache()
        gc.collect()
        try:
            full_img = imread(img_path)
            full_label = subset_label_to_image_bounds(img_path, label_path)
        except Exception:
            continue

        H, W, _ = full_img.shape
        for y in range(0, H, tile_size):
            for x in range(0, W, tile_size):
                y_min, x_min = y, x
                y_max, x_max = y + tile_size, x + tile_size
                if y_max > H: y_min, y_max = H - tile_size, H
                if x_max > W: x_min, x_max = W - tile_size, W

                tile_name = f"{basename}_y{y_min}_x{x_min}.pt"
                save_path = os.path.join(output_dir, tile_name)
                if os.path.exists(save_path): continue

                img_crop = full_img[y_min:y_max, x_min:x_max, :]
                lbl_crop = full_label[y_min:y_max, x_min:x_max]
                if img_crop.max() == 0: continue
                if np.isnan(img_crop).any(): img_crop = np.nan_to_num(img_crop)

                try:
                    feats = extract_multiscale_features(img_crop, model, processor, device, layers, ps=ps)
                    payload = {
                        "image": torch.from_numpy(img_crop),
                        "features": [f.cpu() for f in feats],
                        "label": lbl_crop
                    }
                    temp_path = save_path + ".tmp"
                    torch.save(payload, temp_path)
                    os.rename(temp_path, save_path)
                    del feats, payload, img_crop, lbl_crop
                except RuntimeError as e:
                    if "CUDA" in str(e):
                        del img_crop
                        torch.cuda.empty_cache()
                        gc.collect()
                        if os.path.exists(temp_path): os.remove(temp_path)
                        continue
                    else:
                        raise e
    del model
    del processor
    torch.cuda.empty_cache()
    gc.collect()
    print("Phase 1 Complete.\n")


class PrecomputedDataset(Dataset):
    def __init__(self, processed_dir):
        self.processed_files = glob.glob(os.path.join(processed_dir, "*.pt"))
        if len(self.processed_files) == 0:
            raise ValueError(f"No .pt files found in {processed_dir}.")

    def __len__(self):
        return len(self.processed_files)

    def __getitem__(self, idx):
        try:
            data = torch.load(self.processed_files[idx], weights_only=False)
        except TypeError:
            data = torch.load(self.processed_files[idx])
        img = data['image'].permute(2, 0, 1).float() / 255.0
        features = data['features']
        label_raw = data['label']
        label_seg = torch.from_numpy(label_raw.astype(np.int64)).long()
        return img, features, label_seg


# ==========================================
# 5. ADVANCED TRAINING (Muon + OneCycle + EarlyStopping)
# ==========================================

def train_model(processed_dir, device):
    print("--- PHASE 2: TRAINING ---")

    # 1. Setup Paths
    weights_dir = "weights"
    os.makedirs(weights_dir, exist_ok=True)
    best_model_path = os.path.join(weights_dir, "dinounet_best.pth")

    # 2. Hyperparameters
    dinov3_layers = [5, 11, 17, 23]
    dinov3_channels = [1024] * len(dinov3_layers)
    unet_channels = [512, 256, 128, 64]

    BATCH_SIZE = 4
    EPOCHS = 30  # Increase epochs to allow early stopping to work
    MUON_LR = 0.02
    ADAMW_LR = 1e-2

    # 3. Data Split (Train 80% / Val 20%)
    dataset = PrecomputedDataset(processed_dir)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    print(f"Dataset Split: {train_size} Train, {val_size} Validation")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # 4. Initialize Model
    unet = DinoUNet(dinov3_channels, unet_channels, num_classes=2).to(device)

    # 5. Optimizer Setup (Split Params for Muon vs AdamW)
    # Muon takes >=2D parameters (matrices), AdamW takes 1D (biases, norms)
    muon_params = []
    adamw_params = []
    for name, p in unet.named_parameters():
        if p.ndim >= 2:
            muon_params.append(p)
        else:
            adamw_params.append(p)

    optimizer = Muon(
        params=muon_params,
        lr=MUON_LR,
        momentum=0.95,
        adamw_params=adamw_params,
        adamw_lr=ADAMW_LR
    )

    # 6. SOTA Scheduler: OneCycleLR
    # Note: OneCycleLR updates every STEP, not every EPOCH
    total_steps = len(train_loader) * EPOCHS
    scheduler = OneCycleLR(
        optimizer,
        max_lr=MUON_LR,
        total_steps=total_steps
    )
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda')

    # 7. Early Stopping
    early_stopping = EarlyStopping(patience=5, min_delta=0.005, path=best_model_path)

    # --- TRAIN LOOP ---
    for epoch in range(EPOCHS):
        # A. Training Phase
        unet.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]")

        for img, features, y in pbar:
            img = img.to(device)
            features = [f.to(device) for f in features]
            y = y.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                logits = unet(img, features)
                if logits.shape[-2:] != y.shape[-2:]:
                    y = y.unsqueeze(1).float()
                    y = F.interpolate(y, size=logits.shape[2:], mode='nearest')
                    y = y.squeeze(1).long()
                loss = criterion(logits, y)

            scaler.scale(loss).backward()

            # Muon requires unscaled grads in some implementations, but standard scaler.step() handles float16
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()  # Step every batch

            train_loss += loss.item()
            pbar.set_postfix({"Loss": loss.item(), "LR": scheduler.get_last_lr()[0]})

        avg_train_loss = train_loss / len(train_loader)

        # B. Validation Phase
        unet.eval()
        val_loss = 0
        with torch.no_grad():
            for img, features, y in tqdm(val_loader, desc="[Val]"):
                img = img.to(device)
                features = [f.to(device) for f in features]
                y = y.to(device)

                with torch.amp.autocast('cuda'):
                    logits = unet(img, features)
                    if logits.shape[-2:] != y.shape[-2:]:
                        y = y.unsqueeze(1).float()
                        y = F.interpolate(y, size=logits.shape[2:], mode='nearest')
                        y = y.squeeze(1).long()
                    loss = criterion(logits, y)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch + 1} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # C. Early Stopping Check
        # This will save the model if Val Loss improves
        early_stopping(avg_val_loss, unet)

        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break

    print("Training Complete. Best model is at:", best_model_path)


if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    RAW_IMAGES_DIR = "/home/mak/PycharmProjects/SegEdge/experiments/get_data_from_api/patches_mt/"
    LARGE_LABEL_PATH = "/run/media/mak/Partition of 1TB disk/SH_dataset/planet_labels_2022.tif"
    PROCESSED_DATA_DIR = "/mnt/OS/processed_tiles_1024/"

    MODEL_NAME = "facebook/dinov3-vitl16-pretrain-sat493m"
    LAYERS_TO_EXTRACT = [5, 11, 17, 23]

    #prepare_data_tiles(RAW_IMAGES_DIR, LARGE_LABEL_PATH, PROCESSED_DATA_DIR, MODEL_NAME, LAYERS_TO_EXTRACT, DEVICE,
    #                   tile_size=512)
    #verify_and_clean_dataset_fast(PROCESSED_DATA_DIR)
    train_model(PROCESSED_DATA_DIR, DEVICE)
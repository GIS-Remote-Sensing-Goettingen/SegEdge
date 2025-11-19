import os

# PREVENT CPU HANGS & MEMORY FRAG
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
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
from rasterio.mask import mask
from shapely.geometry import box
from tqdm import tqdm
import gc
import concurrent.futures
import math


# ==========================================
# 1. OPTIMIZER (MUON - SOTA FOR MATRIX UPDATES)
# ==========================================

def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    assert len(G.shape) == 2
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps)
    if G.size(0) > G.size(1): X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    if G.size(0) > G.size(1): X = X.T
    return X.to(G.dtype)


class Muon(optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5,
                 adamw_params=None, adamw_lr=1e-4, adamw_betas=(0.9, 0.95), adamw_wd=0.01):
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
                        adamw_params=adamw_params, adamw_lr=adamw_lr, adamw_betas=adamw_betas, adamw_wd=adamw_wd)
        super().__init__(params, defaults)

    def step(self):
        for group in self.param_groups:
            if group['adamw_params'] is not None: self._step_adamw(group)
            lr, mom, nest, ns = group['lr'], group['momentum'], group['nesterov'], group['ns_steps']
            for p in group['params']:
                if p.grad is None: continue
                g = p.grad
                if g.ndim > 2: g = g.view(g.size(0), -1)
                state = self.state[p]
                if 'momentum_buffer' not in state: state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(mom).add_(g)
                g = g.add(buf, alpha=mom) if nest else buf
                g = zeropower_via_newtonschulz5(g, steps=ns)
                if g.shape != p.shape: g = g.view_as(p)
                p.data.add_(g, alpha=-lr)

    def _step_adamw(self, group):
        lr, betas, wd = group['adamw_lr'], group['adamw_betas'], group['adamw_wd']
        beta1, beta2 = betas
        for p in group['adamw_params']:
            if p.grad is None: continue
            g = p.grad
            state = self.state[p]
            if 'step' not in state:
                state['step'] = 0
                state['exp_avg'], state['exp_avg_sq'] = torch.zeros_like(p), torch.zeros_like(p)
            state['step'] += 1
            exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
            exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1 - beta2)
            denom = exp_avg_sq.sqrt().add_(1e-8)
            step_size = lr * math.sqrt(1 - beta2 ** state['step']) / (1 - beta1 ** state['step'])
            p.data.mul_(1 - lr * wd).addcdiv_(exp_avg, denom, value=-step_size)


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001, path='checkpoint.pth'):
        self.patience, self.min_delta, self.path = patience, min_delta, path
        self.counter, self.best_loss, self.early_stop = 0, None, False

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save(model)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            print(f'EarlyStopping: {self.counter}/{self.patience}')
            if self.counter >= self.patience: self.early_stop = True
        else:
            self.best_loss = val_loss
            self.save(model)
            self.counter = 0

    def save(self, model):
        torch.save(model.state_dict(), self.path)


# ==========================================
# 2. ARCHITECTURE V2: DINO U-NET (FAPM + SPM)
# ==========================================

class SpatialPriorModule(nn.Module):
    """
    SPM: Processes the raw image to capture high-frequency edges
    that the ViT backbone misses.
    """

    def __init__(self, in_channels=3, base_channels=32):
        super().__init__()
        # Stage 1: H -> H/2
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        # Stage 2: H/2 -> H/4
        self.layer2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        c1 = self.stem(x)  # H/2
        c2 = self.layer2(c1)  # H/4
        return c1, c2


class FidelityAwareProjection(nn.Module):
    """
    FAPM: Compresses massive DINO features (1024ch) into Decoder features
    while preserving semantic fidelity using Channel Attention.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

        # Channel Attention (SE-Block style)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(out_channels, out_channels // 4, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels // 4, out_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_proj = self.bn(self.proj(x))
        b, c, _, _ = x_proj.size()
        y = self.avg_pool(x_proj).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x_proj * y.expand_as(x_proj)  # Scale-modulated features


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x): return self.conv(x)


class DinoUNetV2(nn.Module):
    def __init__(self, num_classes=2, dino_ch=1024):
        super().__init__()

        # 1. Spatial Prior Module (Handles Raw Image)
        self.spm = SpatialPriorModule(in_channels=3, base_channels=32)

        # 2. Fidelity Projectors (Handles DINO Features)
        # We assume we extract 4 layers from DINO
        self.fapm1 = FidelityAwareProjection(dino_ch, 512)  # Deepest
        self.fapm2 = FidelityAwareProjection(dino_ch, 256)
        self.fapm3 = FidelityAwareProjection(dino_ch, 128)
        self.fapm4 = FidelityAwareProjection(dino_ch, 64)  # Shallowest DINO

        # 3. Decoder Path
        # Bottleneck
        self.bottleneck = DoubleConv(512, 512)

        # UpBlocks
        self.up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv1 = DoubleConv(256 + 256, 256)  # + FAPM2

        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv2 = DoubleConv(128 + 128, 128)  # + FAPM3

        self.up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv3 = DoubleConv(64 + 64, 64)  # + FAPM4

        # Fusion with SPM (The new V2 part)
        # Up3 output is roughly H/16 (or H/14 depending on DINO).
        # We need to bridge gap to H/4 and H/2 from SPM.

        self.up4 = nn.ConvTranspose2d(64, 64, 2, stride=2)  # -> H/8
        self.up4_extra = nn.ConvTranspose2d(64, 64, 2, stride=2)  # -> H/4

        # Fusion at H/4 (SPM Layer 2)
        self.conv4 = DoubleConv(64 + 64, 64)

        # Fusion at H/2 (SPM Layer 1)
        self.up5 = nn.ConvTranspose2d(64, 32, 2, stride=2)  # -> H/2
        self.conv5 = DoubleConv(32 + 32, 32)

        # Final Upsample to H
        self.final_up = nn.ConvTranspose2d(32, 32, 2, stride=2)  # -> H
        self.final_conv = nn.Conv2d(32, num_classes, 1)

        # Deep Supervision Heads (Optional but good for training)
        self.ds_head1 = nn.Conv2d(64, num_classes, 1)

    def forward(self, img, dino_feats):
        # 1. Run SPM on Image
        spm_h2, spm_h4 = self.spm(img)

        # 2. Process DINO Features (Deepest to Shallowest)
        # dino_feats = [L5, L11, L17, L23] -> We reverse them
        d_deep = self.fapm1(dino_feats[3])  # L23
        d_mid2 = self.fapm2(dino_feats[2])  # L17
        d_mid1 = self.fapm3(dino_feats[1])  # L11
        d_shallow = self.fapm4(dino_feats[0])  # L5

        # 3. Decoder
        x = self.bottleneck(d_deep)

        # Block 1
        x = self.up1(x)
        x = self._concat(x, d_mid2)
        x = self.conv1(x)

        # Block 2
        x = self.up2(x)
        x = self._concat(x, d_mid1)
        x = self.conv2(x)

        # Block 3
        x = self.up3(x)
        x = self._concat(x, d_shallow)
        x = self.conv3(x)
        ds_out = self.ds_head1(x)  # Deep Supervision output

        # 4. SPM Fusion (Bridging the resolution gap)
        # x is currently H/14 or H/16. We need to get to H/4
        x = self.up4(x)
        if x.shape[-1] < spm_h4.shape[-1]:  # Extra upsample if needed
            x = self.up4_extra(x)

        x = self._concat(x, spm_h4)
        x = self.conv4(x)

        # Fusion at H/2
        x = self.up5(x)
        x = self._concat(x, spm_h2)
        x = self.conv5(x)

        # Final
        x = self.final_up(x)
        logits = self.final_conv(x)

        return logits, ds_out

    def _concat(self, x, skip):
        if x.shape[-2:] != skip.shape[-2:]:
            skip = F.interpolate(skip, size=x.shape[-2:], mode='bilinear', align_corners=False)
        return torch.cat([x, skip], dim=1)


# ==========================================
# 3. DATA & HELPERS (UNCHANGED BUT ROBUST)
# ==========================================
# ... (Keep extract_multiscale_features, subset_label_to_image_bounds,
#      _check_single_file, verify_and_clean_dataset_fast, prepare_data_tiles
#      EXACTLY as they were in the previous script. I will condense them
#      here for brevity, but you should paste them back in full.) ...

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


# (Paste subset_label_to_image_bounds here)
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


# (Paste dataset prep functions here)
def _check_single_file(file_path):
    try:
        torch.load(file_path, weights_only=False, map_location='cpu')
        return None
    except Exception:
        return file_path


def verify_and_clean_dataset_fast(output_dir, num_workers=None):
    print(f"--- FAST VERIFICATION ({output_dir}) ---")
    files = glob.glob(os.path.join(output_dir, "*.pt"))
    if not files: return
    if num_workers is None: num_workers = os.cpu_count()
    corrupted_files = []
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_check_single_file, f) for f in files]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(files), desc="Verifying"):
            result = future.result()
            if result is not None: corrupted_files.append(result)
    if corrupted_files:
        for f in corrupted_files:
            try:
                os.remove(f)
            except OSError:
                pass
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
                    payload = {"image": torch.from_numpy(img_crop), "features": [f.cpu() for f in feats],
                               "label": lbl_crop}
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
    del model;
    del processor;
    torch.cuda.empty_cache();
    gc.collect()


class PrecomputedDataset(Dataset):
    def __init__(self, processed_dir):
        self.processed_files = glob.glob(os.path.join(processed_dir, "*.pt"))
        if len(self.processed_files) == 0: raise ValueError(f"No .pt files found in {processed_dir}.")

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
# 4. TRAINING
# ==========================================

def train_model(processed_dir, device):
    print("--- PHASE 2: TRAINING (DINO U-NET V2) ---")
    weights_dir = "weights_v2"
    os.makedirs(weights_dir, exist_ok=True)
    best_model_path = os.path.join(weights_dir, "dinounet_v2_best.pth")

    BATCH_SIZE = 4
    EPOCHS = 40  # More epochs for V2 as it has more parameters
    MUON_LR = 0.02
    ADAMW_LR = 1e-4

    dataset = PrecomputedDataset(processed_dir)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # Initialize V2 Model
    unet = DinoUNetV2(num_classes=2, dino_ch=1024).to(device)

    muon_params, adamw_params = [], []
    for name, p in unet.named_parameters():
        if p.ndim >= 2:
            muon_params.append(p)
        else:
            adamw_params.append(p)

    optimizer = Muon(muon_params, lr=MUON_LR, momentum=0.95, adamw_params=adamw_params, adamw_lr=ADAMW_LR)

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[MUON_LR, ADAMW_LR], total_steps=len(train_loader) * EPOCHS,
        pct_start=0.3, div_factor=25.0, final_div_factor=1000.0
    )

    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler('cuda')
    early_stopping = EarlyStopping(patience=10, min_delta=0.005, path=best_model_path)

    for epoch in range(EPOCHS):
        unet.train()
        train_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]")

        for img, features, y in pbar:
            img = img.to(device)
            features = [f.to(device) for f in features]
            y = y.to(device)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                logits, ds_logits = unet(img, features)

                # Resize Targets for Main Output
                if logits.shape[-2:] != y.shape[-2:]:
                    y_main = y.unsqueeze(1).float()
                    y_main = F.interpolate(y_main, size=logits.shape[2:], mode='nearest').squeeze(1).long()
                else:
                    y_main = y

                # Resize Targets for Deep Supervision Head (H/4 ish)
                if ds_logits.shape[-2:] != y.shape[-2:]:
                    y_ds = y.unsqueeze(1).float()
                    y_ds = F.interpolate(y_ds, size=ds_logits.shape[2:], mode='nearest').squeeze(1).long()
                else:
                    y_ds = y

                # Combined Loss (Main + 0.4 * DeepSup)
                loss = criterion(logits, y_main) + 0.4 * criterion(ds_logits, y_ds)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            train_loss += loss.item()
            pbar.set_postfix({"Loss": loss.item()})

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        unet.eval()
        val_loss = 0
        with torch.no_grad():
            for img, features, y in tqdm(val_loader, desc="[Val]"):
                img = img.to(device)
                features = [f.to(device) for f in features]
                y = y.to(device)

                with torch.amp.autocast('cuda'):
                    logits, _ = unet(img, features)  # Ignore DS output during Val
                    if logits.shape[-2:] != y.shape[-2:]:
                        y = y.unsqueeze(1).float()
                        y = F.interpolate(y, size=logits.shape[2:], mode='nearest').squeeze(1).long()
                    loss = criterion(logits, y)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        print(f"Epoch {epoch + 1} | Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")

        early_stopping(avg_val_loss, unet)
        if early_stopping.early_stop:
            print("Early stopping triggered!")
            break

    print("Training Complete.")


if __name__ == "__main__":
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    RAW_IMAGES_DIR = "/home/mak/PycharmProjects/SegEdge/experiments/get_data_from_api/patches_mt/"
    LARGE_LABEL_PATH = "/run/media/mak/Partition of 1TB disk/SH_dataset/planet_labels_2022.tif"
    PROCESSED_DATA_DIR = "/mnt/OS/processed_tiles_1024/"
    MODEL_NAME = "facebook/dinov3-vitl16-pretrain-sat493m"
    LAYERS_TO_EXTRACT = [5, 11, 17, 23]

    prepare_data_tiles(RAW_IMAGES_DIR, LARGE_LABEL_PATH, PROCESSED_DATA_DIR, MODEL_NAME, LAYERS_TO_EXTRACT, DEVICE,
                       tile_size=512)
    verify_and_clean_dataset_fast(PROCESSED_DATA_DIR)
    train_model(PROCESSED_DATA_DIR, DEVICE)
"""
Data-handling utilities: tiling GeoTIFFs, caching features, validation, and
dataset loader.
"""

from __future__ import annotations

import concurrent.futures
import gc
import glob
import math
import os
import random
from typing import Iterable, List, Optional, Sequence, Tuple, TYPE_CHECKING

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.mask import mask
from rasterio.warp import reproject
from shapely.geometry import box
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from tifffile import imread
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModel

if TYPE_CHECKING:
    from utils.logging import VerbosityLogger


def extract_multiscale_features(
    image_hw3: np.ndarray,
    model,
    processor,
    device: torch.device,
    layers: Sequence[int],
    ps: int = 14,
) -> List[torch.Tensor]:
    """
    Run DINO backbone once on a tile and slice hidden states into feature maps.

    >>> import types
    >>> class DummyBatch(dict):
    ...     def to(self, device):
    ...         return self
    >>> class DummyProcessor:
    ...     def __call__(self, images, return_tensors=None, do_resize=None, do_center_crop=None):
    ...         batch = DummyBatch()
    ...         batch["pixel_values"] = torch.randn(1, 3, 14, 14)
    ...         return batch
    >>> class DummyModel(torch.nn.Module):
    ...     def __init__(self):
    ...         super().__init__()
    ...         self.config = types.SimpleNamespace(num_register_tokens=0)
    ...     def forward(self, **kwargs):
    ...         hidden = tuple(torch.randn(1, 197, 4) for _ in range(24))
    ...         return types.SimpleNamespace(hidden_states=hidden)
    >>> dummy_processor = DummyProcessor()
    >>> dummy_model = DummyModel()
    >>> feats = extract_multiscale_features(
    ...     np.random.rand(14, 14, 3).astype(np.float32),
    ...     dummy_model,
    ...     dummy_processor,
    ...     torch.device("cpu"),
    ...     layers=[0],
    ...     ps=14,
    ... )
    >>> len(feats)
    1
    """

    inputs = processor(
        images=image_hw3,
        return_tensors="pt",
        do_resize=False,
        do_center_crop=False,
    ).to(device)
    R = getattr(model.config, "num_register_tokens", 0)
    with torch.no_grad():
        out = model(**inputs, output_hidden_states=True)
        hidden_states = out.hidden_states
    _, _, Hproc, Wproc = inputs["pixel_values"].shape
    feature_maps = []
    for layer_idx in layers:
        layer_output = hidden_states[layer_idx]
        patch_tokens = layer_output[:, 1 + R :, :]
        Hp, Wp = Hproc // ps, Wproc // ps
        feats = patch_tokens.reshape(1, Hp, Wp, -1).permute(0, 3, 1, 2)
        feature_maps.append(feats.squeeze(0).cpu())
    return feature_maps


def subset_label_to_image_bounds(img_path: str, lab_path: str) -> np.ndarray:
    """
    Crop or reproject the label raster so it aligns with the image tile.

    >>> subset_label_to_image_bounds("image.tif", "labels.tif")  # doctest: +SKIP
    array(...)
    """

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
                t_lbl = F.interpolate(t_lbl, size=(H, W), mode="nearest")
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


def _check_single_file(file_path: str) -> str | None:
    """
    Validate that a cached tile can be read.

    >>> import tempfile
    >>> tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
    >>> torch.save({"x": torch.tensor([1])}, tmp.name)
    >>> _check_single_file(tmp.name)
    """

    try:
        torch.load(file_path, weights_only=False, map_location="cpu")
        return None
    except Exception:
        return file_path


def verify_and_clean_dataset_fast(
    output_dir: str,
    num_workers: int | None = None,
    logger: Optional["VerbosityLogger"] = None,
) -> None:
    """
    Spawn workers to make sure each cached tile is readable; delete corrupt ones.

    >>> verify_and_clean_dataset_fast("/tmp", num_workers=1)  # doctest: +SKIP
    """

    files = glob.glob(os.path.join(output_dir, "*.pt"))
    if not files:
        if logger:
            logger.info("No cached tiles found for verification.")
        return
    if num_workers is None:
        num_workers = os.cpu_count() or 1
    corrupted_files = []
    if logger:
        logger.info(f"Verifying {len(files)} cached tiles.")
    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(_check_single_file, f) for f in files]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(files), desc="Verifying"):
            result = future.result()
            if result is not None:
                corrupted_files.append(result)
    for f in corrupted_files:
        try:
            os.remove(f)
            if logger:
                logger.error(f"Removed corrupted tile {f}")
        except OSError:
            if logger:
                logger.error(f"Failed to remove corrupted tile {f}")


def prepare_data_tiles(
    img_dir: str,
    label_path: str,
    output_dir: str,
    model_name: str,
    layers: Sequence[int],
    device: torch.device,
    tile_size: int = 512,
    logger: Optional["VerbosityLogger"] = None,
) -> None:
    """
    Tile raw GeoTIFFs, align labels, and pre-compute DINO feature tensors.

    >>> # Light-touch doctest ensures function signature works by calling with
    >>> # a fake directory (no images). Should exit early with no errors.
    >>> import tempfile
    >>> tmp_imgs = tempfile.mkdtemp()
    >>> tmp_out = tempfile.mkdtemp()
    >>> prepare_data_tiles(
    ...     img_dir=tmp_imgs,
    ...     label_path="/tmp/nonexistent.tif",
    ...     output_dir=tmp_out,
    ...     model_name="facebook/dinov3-vitl16-pretrain-sat493m",
    ...     layers=[5],
    ...     device=torch.device("cpu"),
    ... )  # doctest: +SKIP
    """

    def _log_info(message: str) -> None:
        if logger:
            logger.info(message)
        else:
            print(message)

    def _log_debug(message: str) -> None:
        if logger:
            logger.debug(message)

    _log_info("--- PHASE 1: TILING & PRE-COMPUTING ---")
    os.makedirs(output_dir, exist_ok=True)
    existing = glob.glob(os.path.join(output_dir, "*.pt"))
    if existing:
        _log_info(f"[INFO] Found {len(existing)} existing tiles.")
    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).eval().to(device)
    image_paths = glob.glob(os.path.join(img_dir, "*.tif"))
    ps = 14 if "vitl14" in model_name else 16
    for img_path in tqdm(image_paths, desc="Processing Large Images"):
        basename = os.path.splitext(os.path.basename(img_path))[0]
        _log_info(f"Processing image {basename}")
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
                if y_max > H:
                    y_min, y_max = H - tile_size, H
                if x_max > W:
                    x_min, x_max = W - tile_size, W
                tile_name = f"{basename}_y{y_min}_x{x_min}.pt"
                save_path = os.path.join(output_dir, tile_name)
                if os.path.exists(save_path):
                    _log_debug(f"Tile already exists: {tile_name}")
                    continue
                img_crop = full_img[y_min:y_max, x_min:x_max, :]
                lbl_crop = full_label[y_min:y_max, x_min:x_max]
                if img_crop.max() == 0:
                    _log_debug(f"Skipping zero tile {tile_name}")
                    continue
                if np.isnan(img_crop).any():
                    img_crop = np.nan_to_num(img_crop)
                    _log_debug(f"NaNs detected and replaced for tile {tile_name}")
                try:
                    feats = extract_multiscale_features(
                        img_crop,
                        model,
                        processor,
                        device,
                        layers,
                        ps=ps,
                    )
                    payload = {
                        "image": torch.from_numpy(img_crop),
                        "features": [f.cpu() for f in feats],
                        "label": lbl_crop,
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
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                        continue
                    raise e
    del model
    del processor
    torch.cuda.empty_cache()
    gc.collect()
    _log_info("Phase 1 Complete.")


class PrecomputedDataset(Dataset):
    """
    Lazy dataset that loads cached tiles on demand.
    """

    def __init__(
        self,
        processed_dir: str,
        augmentation_cfg: Optional[dict] = None,
        file_subset: Optional[List[str]] = None,
    ) -> None:
        """
        Index every cached tile path.

        >>> import tempfile
        >>> tmpdir = tempfile.mkdtemp()
        >>> sample = os.path.join(tmpdir, "sample.pt")
        >>> torch.save({"image": torch.zeros(4, 4, 3), "features": [torch.zeros(1, 1, 1)], "label": np.zeros((4, 4))}, sample)
        >>> ds = PrecomputedDataset(tmpdir)
        >>> len(ds)
        1
        """

        if file_subset is not None:
            self.processed_files = file_subset
        else:
            self.processed_files = sorted(glob.glob(os.path.join(processed_dir, "*.pt")))
        if not self.processed_files:
            raise ValueError(f"No .pt files found in {processed_dir}.")
        self.augmentation_cfg = augmentation_cfg or {}

    def __len__(self) -> int:
        """
        Number of cached tiles.

        >>> ds = PrecomputedDataset.__new__(PrecomputedDataset)
        >>> ds.processed_files = [1, 2, 3]
        >>> len(ds)
        3
        """

        return len(self.processed_files)

    def __getitem__(self, idx: int):
        """
        Load the tile, normalize RGB image, and return label tensor.

        >>> import tempfile
        >>> tmpdir = tempfile.mkdtemp()
        >>> sample = os.path.join(tmpdir, "sample.pt")
        >>> torch.save({
        ...     "image": torch.zeros(4, 4, 3),
        ...     "features": [torch.zeros(3, 3, 3) for _ in range(4)],
        ...     "label": np.zeros((4, 4)),
        ... }, sample)
        >>> ds = PrecomputedDataset(tmpdir)
        >>> img, feats, label = ds[0]
        >>> img.shape[0]
        3
        """

        try:
            data = torch.load(self.processed_files[idx], weights_only=False)
        except TypeError:
            data = torch.load(self.processed_files[idx])
        img = data["image"].permute(2, 0, 1).float() / 255.0
        features = data["features"]
        label_raw = data["label"]
        label_seg = torch.from_numpy(label_raw.astype(np.int64)).long()
        img, features, label_seg = self._apply_augmentations(img, features, label_seg)
        return img, features, label_seg

    def _apply_augmentations(
        self,
        img: torch.Tensor,
        features: List[torch.Tensor],
        label: torch.Tensor,
    ) -> tuple[torch.Tensor, List[torch.Tensor], torch.Tensor]:
        cfg = self.augmentation_cfg
        if not cfg or not cfg.get("enable", False):
            return img, features, label
        feats = [f.clone() for f in features]
        # Random rotation (multiples of 90 degrees)
        if cfg.get("rotate90", False):
            k = random.randint(0, 3)
            if k:
                img = torch.rot90(img, k, dims=(1, 2))
                label = torch.rot90(label, k, dims=(0, 1))
                feats = [torch.rot90(f, k, dims=(1, 2)) for f in feats]
        # Horizontal flip
        if cfg.get("hflip", False) and random.random() < 0.5:
            img = torch.flip(img, dims=(2,))
            label = torch.flip(label, dims=(1,))
            feats = [torch.flip(f, dims=(2,)) for f in feats]
        # Vertical flip
        if cfg.get("vflip", False) and random.random() < 0.5:
            img = torch.flip(img, dims=(1,))
            label = torch.flip(label, dims=(0,))
            feats = [torch.flip(f, dims=(1,)) for f in feats]
        return img, feats, label

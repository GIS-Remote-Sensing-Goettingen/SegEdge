# justification:
# - same logic as before (WMS → GeoTIFF patches)
# - adds robust logging, timing, and progress feedback
# - uses ThreadPoolExecutor for concurrency (I/O-bound)

import requests
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
from io import BytesIO
from PIL import Image
import numpy as np
import pyproj
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
import time
from datetime import timedelta

# Optional: tqdm for progress bar (pip install tqdm)
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------
url = "https://dienste.gdi-sh.de/WMS_SH_DOP20col_OpenGBD"
layer = "sh_dop20_rgb"
crs_epsg = 25832
tile_size = 1000          # meters (1x1 km)
width, height = 5000, 5000
max_workers = 20           # tune based on network/server limits
timeout_s = 90
out_dir = Path("patches_mt")
out_dir.mkdir(exist_ok=True)

# Bounding box (lon/lat)
bbox_ll = (10.407, 53.90, 10.52, 54.00)

# ---------------------------------------------------------------------
# LOGGING SETUP
# ---------------------------------------------------------------------
logfile = out_dir / "download.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    handlers=[
        logging.FileHandler(logfile, mode="w"),
        logging.StreamHandler()
    ]
)

logging.info("=== WMS tile downloader started ===")
logging.info(f"Saving output to {out_dir.resolve()}")
logging.info(f"Max workers: {max_workers}")

# ---------------------------------------------------------------------
# CRS conversion
# ---------------------------------------------------------------------
proj = pyproj.Transformer.from_crs("EPSG:4326", f"EPSG:{crs_epsg}", always_xy=True)
min_x, min_y = proj.transform(bbox_ll[0], bbox_ll[1])
max_x, max_y = proj.transform(bbox_ll[2], bbox_ll[3])
x_steps = np.arange(min_x, max_x, tile_size)
y_steps = np.arange(min_y, max_y, tile_size)

logging.info(f"Transformed bounding box: {min_x:.1f}, {min_y:.1f} to {max_x:.1f}, {max_y:.1f}")
logging.info(f"Total tiles: {len(x_steps) * len(y_steps)}")

# ---------------------------------------------------------------------
# Tile download function
# ---------------------------------------------------------------------
def fetch_and_save_tile(x0, y0):
    bbox = (x0, y0, x0 + tile_size, y0 + tile_size)
    params = {
        "SERVICE": "WMS",
        "VERSION": "1.3.0",
        "REQUEST": "GetMap",
        "LAYERS": layer,
        "STYLES": "default",
        "CRS": f"EPSG:{crs_epsg}",
        "BBOX": ",".join(map(str, bbox)),
        "WIDTH": width,
        "HEIGHT": height,
        "FORMAT": "image/png"
    }

    out_file = out_dir / f"dop20_patch_{int(x0)}_{int(y0)}.tif"
    if out_file.exists():
        return f"SKIP {out_file.name}"

    try:
        t0 = time.time()
        r = requests.get(url, params=params, timeout=timeout_s)
        dt = time.time() - t0

        if r.status_code != 200:
            return f"FAIL {x0:.0f},{y0:.0f} [{r.status_code}] ({dt:.1f}s)"

        img = Image.open(BytesIO(r.content))
        arr = np.array(img)
        transform = from_bounds(*bbox, width=width, height=height)
        crs = CRS.from_epsg(crs_epsg)

        with rasterio.open(
            out_file, "w",
            driver="GTiff",
            height=arr.shape[0],
            width=arr.shape[1],
            count=3,
            dtype=arr.dtype,
            crs=crs,
            transform=transform
        ) as dst:
            for i in range(3):
                dst.write(arr[:, :, i], i + 1)

        return f"DONE {out_file.name} ({dt:.1f}s)"

    except Exception as e:
        return f"ERROR {x0:.0f},{y0:.0f}: {e}"

# ---------------------------------------------------------------------
# Multithreaded execution with progress and timing
# ---------------------------------------------------------------------
start = time.time()
tasks = []
total = len(x_steps) * len(y_steps)

executor = ThreadPoolExecutor(max_workers=max_workers)
for x0 in x_steps:
    for y0 in y_steps:
        tasks.append(executor.submit(fetch_and_save_tile, x0, y0))

if tqdm:
    iterator = tqdm(as_completed(tasks), total=total, desc="Downloading", ncols=100)
else:
    iterator = as_completed(tasks)

completed, failed = 0, 0
for fut in iterator:
    msg = fut.result()
    if "FAIL" in msg or "ERROR" in msg:
        failed += 1
        logging.warning(msg)
    else:
        completed += 1
        logging.info(msg)

executor.shutdown(wait=True)
elapsed = timedelta(seconds=int(time.time() - start))
logging.info(f"=== Finished: {completed} success, {failed} failed, total {elapsed} ===")

# justification (overall): Fast, reliable WMS→GeoTIFF tiling at ~0.20 m/px.
# - Pooled HTTP session + retries: fewer TCP handshakes, auto backoff on 5xx.
# - EPSG:25832 + 1 km tiles + 5000 px: preserves ~0.20 m/px, avoids server resampling.
# - ThreadPoolExecutor: I/O-bound concurrency, bounded to respect the server.
# - Logging + tqdm: progress visibility and post-mortem debugging.
import math, time, logging
from datetime import timedelta
from pathlib import Path
from io import BytesIO
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pyproj
import requests
from requests.adapters import HTTPAdapter, Retry
from PIL import Image
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

# -------------------- CONFIG --------------------------------------------------
WMS_URL = "https://dienste.gdi-sh.de/WMS_SH_DOP20col_OpenGBD"
LAYER   = "sh_dop20_rgb"
CRS_EPSG   = 25832
TILE_M     = 1000                # 1 km × 1 km patches
GSD_M      = 0.2                 # 20 cm GSD target
WIDTH      = HEIGHT = int(round(TILE_M / GSD_M))  # 1000 / 0.2 = 5000 px
MAX_WORKERS = 12                  # tune: 4–8 is usually safe
TIMEOUT_S   = 90
OUT_DIR     = Path("/mnt/ceph-hdd/projects/mthesis_davide_mattioli/patches_mt")
OUT_DIR.mkdir(exist_ok=True)
LOG_FILE    = OUT_DIR / "download.log"

# AOI in lon/lat (min_lon, min_lat, max_lon, max_lat)
BBOX_LL = (10.407, 53.90, 10.52, 54.00)

# -------------------- LOGGING -------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode="w"), logging.StreamHandler()],
)
logging.info("=== DOP20 WMS 1 km tiler (≈0.20 m/px) ===")
logging.info(f"Output dir: {OUT_DIR.resolve()}  Max workers: {MAX_WORKERS}")

# -------------------- HTTP SESSION (pooling + retries) -----------------------
# justification: Connection pooling reduces latency; Retry(backoff) handles 502/503/504 gracefully.
def make_session() -> requests.Session:
    sess = requests.Session()
    sess.headers.update({
        "User-Agent": "dop20-tiler/1.0",
        "Accept-Encoding": "gzip, deflate",
        "Connection": "keep-alive",
    })
    retry = Retry(
        total=10,               # 1 initial + 4 retries
        backoff_factor=0.7,    # 0.7, 1.4, 2.8, 5.6 s (plus jitter internal to urllib3)
        status_forcelist=(502, 503, 504),
        allowed_methods=frozenset(["GET"]),
        raise_on_status=False,
    )
    adapter = HTTPAdapter(pool_connections=64, pool_maxsize=64, max_retries=retry)
    sess.mount("https://", adapter)
    sess.mount("http://", adapter)
    return sess

SESSION = make_session()

# -------------------- CRS & GRID ---------------------------------------------
# justification: EPSG:25832 in meters → exact 1 km steps; preserves ~0.20 m/px.
proj_4326_to_25832 = pyproj.Transformer.from_crs("EPSG:4326", f"EPSG:{CRS_EPSG}", always_xy=True)
min_x, min_y = proj_4326_to_25832.transform(BBOX_LL[0], BBOX_LL[1])
max_x, max_y = proj_4326_to_25832.transform(BBOX_LL[2], BBOX_LL[3])

# Snap to 1 km grid so each BBOX is cleanly aligned (no fractional km causing uneven px size)
gx0 = math.floor(min_x / TILE_M) * TILE_M
gy0 = math.floor(min_y / TILE_M) * TILE_M
gx1 = math.ceil (max_x / TILE_M) * TILE_M
gy1 = math.ceil (max_y / TILE_M) * TILE_M

xs = np.arange(gx0, gx1, TILE_M)
ys = np.arange(gy0, gy1, TILE_M)
logging.info(f"AOI 25832: ({min_x:.1f},{min_y:.1f})–({max_x:.1f},{max_y:.1f}); "
             f"grid: {len(xs)} × {len(ys)} = {len(xs)*len(ys)} tiles; px={WIDTH}x{HEIGHT}")

# -------------------- WORKER --------------------------------------------------
def fetch_and_write_tile(x0: float, y0: float) -> str:
    # justification: per-tile exact bbox in meters, matching 1 km; WIDTH/HEIGHT = 5000 for ~0.20 m/px.
    bbox = (x0, y0, x0 + TILE_M, y0 + TILE_M)
    params = {
        "SERVICE": "WMS",
        "VERSION": "1.3.0",
        "REQUEST": "GetMap",
        "LAYERS": LAYER,
        "STYLES": "",                 # default style
        "CRS": f"EPSG:{CRS_EPSG}",
        "BBOX": ",".join(map(str, bbox)),
        "WIDTH": WIDTH,
        "HEIGHT": HEIGHT,
        "FORMAT": "image/png",        # lossless for ML; fast enough
        # Optional: "DPI": "96"  # not required; some servers accept DPI hints
    }
    out_path = OUT_DIR / f"dop20_{int(x0)}_{int(y0)}_1km_20cm.tif"
    if out_path.exists():
        return f"SKIP {out_path.name}"

    t0 = time.time()
    r = SESSION.get(WMS_URL, params=params, timeout=TIMEOUT_S)
    # Accept 200 only with image/*; WMS may return XML ServiceException with 200 status.
    if r.status_code != 200 or not r.headers.get("Content-Type", "").startswith("image/"):
        dt = time.time() - t0
        # allow retry machinery to handle 5xx; for 200+XML, this is likely a hard error (e.g., exceeded limits)
        detail = r.text[:220].replace("\n", " ")
        return f"FAIL {int(x0)},{int(y0)} [{r.status_code}] ct={r.headers.get('Content-Type')} ({dt:.1f}s) {detail}"

    # justification: PNG→numpy; write GeoTIFF with correct transform & CRS for analysis-ready data.
    img = Image.open(BytesIO(r.content))
    arr = np.array(img)
    transform = from_bounds(*bbox, width=WIDTH, height=HEIGHT)
    crs = CRS.from_epsg(CRS_EPSG)

    with rasterio.open(
        out_path, "w",
        driver="GTiff",
        height=arr.shape[0],
        width=arr.shape[1],
        count=3,
        dtype=arr.dtype,
        crs=crs,
        transform=transform,
        tiled=True,          # better IO
        compress="DEFLATE",  # lossless; adjust if you prefer uncompressed (larger)
        predictor=2
    ) as dst:
        for b in range(3):
            dst.write(arr[:, :, b], b + 1)

    dt = time.time() - t0
    return f"DONE {out_path.name} ({dt:.1f}s)"

# -------------------- EXECUTION ----------------------------------------------
start = time.time()
tasks = []
with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
    for x0 in xs:
        for y0 in ys:
            tasks.append(ex.submit(fetch_and_write_tile, float(x0), float(y0)))

    iterator = tqdm(as_completed(tasks), total=len(tasks), desc="Downloading", ncols=100) if tqdm else as_completed(tasks)
    ok, bad = 0, 0
    for fut in iterator:
        msg = fut.result()
        if "DONE" in msg or "SKIP" in msg:
            ok += 1
            logging.info(msg)
        else:
            bad += 1
            logging.warning(msg)

elapsed = timedelta(seconds=int(time.time() - start))
logging.info(f"=== Finished: {ok} success/skip, {bad} failed, elapsed {elapsed} ===")

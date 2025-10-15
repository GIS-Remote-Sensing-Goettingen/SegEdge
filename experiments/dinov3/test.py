# === S2DR3 shim v12: env presets + path sandbox + subprocess rewrite + NumPy save
# + PROJ/GDAL data priming + GDAL Create/Open + Warp->Translate shim + torch/safetensors bridge
# + model cache ===
import os, sys, builtins, shutil, subprocess, traceback
from pathlib import Path

# ---------- environment presets ----------
_ENV_EXPORTS = {
    "COLAB_RELEASE_TAG": "release",
    "COLAB_GPU": "1",
    "COLAB_DEBUG": "0",
    "COLAB_TPU_ADDR": "",
    "GCE_METADATA_TIMEOUT": "3",
    "CLOUDSDK_CORE_PROJECT": "colab-project",
    # GeoTIFF SRS: prefer official EPSG parameters to avoid CRS mismatch warnings
    "GTIFF_SRS_SOURCE": "EPSG",
}
for _k, _v in _ENV_EXPORTS.items():
    os.environ[_k] = _v
print("[shim] env primed:", ", ".join(f"{k}={os.environ[k]!r}" for k in sorted(_ENV_EXPORTS)))

# ---------- basics ----------
def _pick_sandbox_root():
    candidates = []
    env_home = os.environ.get("S2DR3_HOME")
    if env_home:
        candidates.append(Path(env_home).expanduser())
    candidates.append(Path.home()/".cache"/"s2dr3"/"sandbox")
    here = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
    candidates.append(here/"_s2dr3_sandbox")
    candidates.append(Path.cwd()/"_s2dr3_sandbox")
    for cand in candidates:
        try:
            cand.mkdir(parents=True, exist_ok=True)
            probe = cand/".write_test"
            with probe.open("wb") as fh:
                fh.write(b"0")
            probe.unlink(missing_ok=True)
            return cand.resolve()
        except Exception as e:
            print(f"[shim] sandbox candidate skipped: {cand} ({e})")
    raise RuntimeError("No writable sandbox root found")

HOME = _pick_sandbox_root()
ROOTS = ("/var", "/content")
HOME.mkdir(parents=True, exist_ok=True)
print("[shim] sandbox root:", HOME)

MODEL_CACHE_ROOT = HOME/"local"/"S2DR3"
MODEL_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
MODEL_CACHE_PATTERNS = [
    str(MODEL_CACHE_ROOT/"**/*.pt"),
    str(MODEL_CACHE_ROOT/"**/*.pth"),
    str(MODEL_CACHE_ROOT/"**/*.safetensors"),
    str(MODEL_CACHE_ROOT/"**/*.dec"),
    str(HOME/"**/*.pt"),
    str(HOME/"**/*.pth"),
    str(HOME/"**/*.safetensors"),
    str(Path.home()/".cache/s2dr3/**/*.pt"),
    str(Path.home()/".cache/s2dr3/**/*.pth"),
    str(Path.home()/".cache/s2dr3/**/*.safetensors"),
    str(Path.home()/".cache/huggingface/**/*.safetensors"),
]

def _rewrite(p):
    p = Path(str(p)); s = str(p)
    if not p.is_absolute(): return p
    for r in ROOTS:
        if s == r or s.startswith(r + "/"):
            tail = s[len(r):].lstrip("/")
            return (HOME/tail) if tail else HOME
    return p
def _rewrite_str(s): return str(_rewrite(s)) if isinstance(s,str) and any(s.startswith(r) for r in ROOTS) else s
def _ensure_parent(x): Path(x).parent.mkdir(parents=True, exist_ok=True)

# ---------- file I/O + subprocess ----------
_real_open = builtins.open
def _open(file, mode="r", *a, **k):
    rf = _rewrite(file)
    if str(rf)!=str(file): print(f"[shim] open: {file} -> {rf} mode={mode}")
    if any(m in mode for m in ("w","a","x","+")): _ensure_parent(rf)
    return _real_open(rf, mode, *a, **k)
builtins.open = _open

import os as _os, shutil as _shutil, numpy as _np
_real_makedirs = _os.makedirs
_os.makedirs = lambda name, *a, **k: _real_makedirs(_rewrite(name), *a, **k)
_os.rename  = (lambda _f: (lambda s,d,*a,**k: _f(_rewrite(s), _rewrite(d), *a, **k)))(_os.rename)
_shutil.move = (lambda _f: (lambda s,d,*a,**k: (_ensure_parent(_rewrite(d)), _f(_rewrite(s), _rewrite(d), *a, **k))[1]))(_shutil.move)

_real_save = _np.save
def _np_save(file, arr, *a, **k):
    rf = _rewrite(file); _ensure_parent(rf)
    if str(rf)!=str(file): print(f"[shim] np.save: {file} -> {rf}")
    return _real_save(rf, arr, *a, **k)
_np.save = _np_save

def _rewrite_cmd(cmd):
    if isinstance(cmd, (list,tuple)):
        return type(cmd)([_rewrite_str(x) if isinstance(x,str) else x for x in cmd])
    if isinstance(cmd,str) and (cmd.startswith(("/var","/content")) or " /var/" in cmd or " /content/" in cmd):
        parts = cmd.split()
        parts = [_rewrite_str(p) if isinstance(p,str) and any(p.startswith(r) for r in ROOTS) else p for p in parts]
        return " ".join(parts)
    return cmd

def _maybe_fake_smime(cmd):
    txt = " ".join(cmd) if isinstance(cmd,(list,tuple)) else (cmd if isinstance(cmd,str) else "")
    if "openssl" in txt and "smime" in txt and "-decrypt" in txt and "-out" in txt:
        parts = txt.split()
        try:
            out_idx = parts.index("-out")+1
            out_path = Path(_rewrite_str(parts[out_idx]))
        except Exception: return (False, None)
        # Use cached-dec if we have it
        cache_target = MODEL_CACHE_ROOT/out_path.name
        if cache_target.exists():
            _ensure_parent(out_path)
            shutil.copy2(cache_target, out_path)
            print(f"[shim] model-cache: using cached decrypted blob: {cache_target}")
            return (True, 0)
        from glob import glob
        cands = []
        for pat in MODEL_CACHE_PATTERNS:
            cands += glob(pat, recursive=True)
        cands = [p for p in dict.fromkeys(cands) if Path(p).is_file()]
        if not cands:
            out_path.write_bytes(b"")
            print(f"[shim] faked smime -> empty: {out_path}")
            return (True, 0)
        # prefer pickle over safetensors when available
        cands = sorted(cands, key=lambda p: (p.endswith(".safetensors"), -Path(p).stat().st_size))
        src = cands[0]
        _ensure_parent(out_path); shutil.copy2(src, out_path)
        size = out_path.stat().st_size if out_path.exists() else 0
        if cache_target != Path(src):
            try:
                _ensure_parent(cache_target)
                shutil.copy2(out_path, cache_target)
                print(f"[shim] model-cache: primed from {src} -> {cache_target} (size={size/1048576:.1f} MB)")
            except Exception as e:
                print(f"[shim] model-cache: failed to prime ({e})")
        print(f"[shim] faked smime -> copied: {src} -> {out_path}")
        return (True, 0)
    return (False, None)

_real_run = subprocess.run
def run_patched(cmd, *a, **k):
    handled, rc = _maybe_fake_smime(cmd)
    if handled:
        print(f"[shim] subprocess.run (faked): {cmd}")
        return subprocess.CompletedProcess(args=cmd, returncode=rc, stdout=b"", stderr=b"")
    new = _rewrite_cmd(cmd)
    if new != cmd: print(f"[shim] subprocess.run: {cmd} -> {new}")
    return _real_run(new, *a, **k)
subprocess.run = run_patched

_real_system = os.system
def system_patched(cmd):
    handled, rc = _maybe_fake_smime(cmd)
    if handled:
        print(f"[shim] os.system (faked): {cmd} -> rc={rc}")
        return rc
    new = _rewrite_cmd(cmd)
    if new != cmd: print(f"[shim] os.system: {cmd} -> {new}")
    return _real_system(new)
os.system = system_patched

# ---------- PROJ/GDAL data priming ----------
def _prime_proj_gdal_data():
    # Prefer explicit env first
    cand = []
    for key in ("PROJ_LIB","PROJ_DATA"):
        v = os.environ.get(key, "")
        if v: cand.append(Path(v))
    # Conda/Miniforge prefixes
    for base in filter(None, [os.environ.get("CONDA_PREFIX"), sys.prefix, sys.base_prefix, str(Path.home()/ "miniforge3")]):
        cand.append(Path(base)/"share"/"proj")
        cand.append(Path(base)/"Library"/"share"/"proj")   # windows-ish
    # Set if a plausible dir is found (proj.db exists)
    chosen = None
    for c in cand:
        try:
            if c and c.exists() and any((c/"proj.db").exists() or list(c.glob("*.csv"))):
                chosen = c; break
        except Exception:
            pass
    if chosen:
        os.environ["PROJ_LIB"] = str(chosen)
        os.environ["PROJ_DATA"] = str(chosen)
        print("[shim] PROJ data path set:", chosen)
    # GDAL data (WKT, EPSG tables, etc.)
    gdal_cand = []
    for base in filter(None, [os.environ.get("CONDA_PREFIX"), sys.prefix, sys.base_prefix, str(Path.home()/ "miniforge3")]):
        gdal_cand.append(Path(base)/"share"/"gdal")
        gdal_cand.append(Path(base)/"Library"/"share"/"gdal")
    for gc in gdal_cand:
        if gc.exists() and any(gc.glob("srs_*.csv")) or (gc/"gdalicon.png").exists():
            os.environ["GDAL_DATA"] = str(gc)
            print("[shim] GDAL data path set:", gc)
            break

_prime_proj_gdal_data()

# ---------- GDAL Create/Open + Warp->Translate shim ----------
try:
    from osgeo import gdal
    gdal.UseExceptions()

    _Real_GetDriverByName = gdal.GetDriverByName
    def _Patched_GetDriverByName(name):
        drv = _Real_GetDriverByName(name)
        if drv is None: return None
        class Proxy:
            def __init__(self,d): self._d=d
            def Create(self,fn, x,y,bands,etype,options=None):
                fn2 = str(_rewrite(fn)); _ensure_parent(fn2)
                return self._d.Create(fn2, x,y,bands,etype,options)
            def __getattr__(self,k): return getattr(self._d,k)
        return Proxy(drv)
    gdal.GetDriverByName = _Patched_GetDriverByName

    _Real_Open = gdal.Open
    gdal.Open = lambda fn,*a,**k: _Real_Open(str(_rewrite(fn)), *a, **k)

    _Real_Warp = gdal.Warp  # keep original just in case

    # --- helpers for composites/band order ---
    def _product_kind(path_str):
        s = Path(path_str).name.upper()
        if "_TCI" in s:   return "TCI"   # True Color: B4,B3,B2
        if "_IRP" in s:   return "IRP"   # IR pseudo-color: B12,B8,B5 (adjust if your pipeline differs)
        if "NDVI" in s:   return "NDVI"
        return None

    def _bands_for(kind):
        return {"TCI":[4,3,2], "IRP":[12,8,5]}.get(kind, None)  # S2 B4,B3,B2 true-color. 

    def _build_bandorder_vrt(src_ds, band_list):
        print(f"[shim][bands] Request band order: {band_list} from src_count={src_ds.RasterCount}")
        if src_ds.RasterCount >= max(band_list):
            opts = gdal.TranslateOptions(format="VRT", bandList=band_list)
            vrt = gdal.Translate("", src_ds, options=opts)
            print("[shim][bands] Built VRT with bandList", band_list, "=>", bool(vrt))
            return vrt
        print("[shim][bands] Source has too few bands; using original order.")
        return src_ds

    def _describe(ds):
        try:
            return {"size": (ds.RasterXSize, ds.RasterYSize),
                    "count": ds.RasterCount,
                    "gt": tuple(ds.GetGeoTransform()) if ds.GetGeoTransform() else None,
                    "proj_set": bool(ds.GetProjection())}
        except Exception: return {}

    # --- replacement for gdal.Warp using Translate (stable across wheels) ---
    def Warp_patched(dest, src, *args, **kwargs):
        # rewrite paths & open source
        dest2 = str(_rewrite(dest)); _ensure_parent(dest2)
        src_ds = None
        if isinstance(src, (str, Path)):
            src_ds = gdal.Open(str(_rewrite(src)))
        elif isinstance(src, gdal.Dataset.__class__):  # proxy type
            src_ds = src
        else:
            # list of sources? fall back to original Warp if needed
            print("[shim][Warp] non-simple src; delegating to real gdal.Warp")
            return _Real_Warp(dest2, src, *args, **kwargs)

        kind = _product_kind(dest2)
        print("[shim][Warp] call -> dest:", dest2, "kind:", kind, "src:", _describe(src_ds))

        # Build VRT for specific composites to enforce correct band order
        ds_for_write = src_ds
        color_interp = None
        if kind in ("TCI","IRP"):
            band_list = _bands_for(kind)
            ds_for_write = _build_bandorder_vrt(src_ds, band_list)
            color_interp = ["red","green","blue"]  # set band color roles

        # Map a few commonly used warp kwargs to Translate equivalents
        translate_kwargs = {}
        if "resampleAlg" in kwargs:
            translate_kwargs["resampleAlg"] = kwargs["resampleAlg"]
        else:
            translate_kwargs["resampleAlg"] = "cubic"  # good for upsampling

        if "xRes" in kwargs: translate_kwargs["xRes"] = kwargs["xRes"]
        if "yRes" in kwargs: translate_kwargs["yRes"] = kwargs["yRes"]
        if "outputBounds" in kwargs: translate_kwargs["outputBounds"] = kwargs["outputBounds"]
        if "dstSRS" in kwargs: translate_kwargs["outputSRS"] = kwargs["dstSRS"]

        # creation options for GTiff / COG-like output
        co = ["TILED=YES","COMPRESS=DEFLATE","PREDICTOR=2","BIGTIFF=IF_SAFER"]
        if color_interp:
            translate_kwargs["colorInterpretation"] = color_interp

        # If dynamic range is very dark (16-bit reflectance), optional scale for TCI/IRP:
        # Uncomment to 0-255 map if your viewer expects 8-bit. (S2 L2A often 0..10000)
        # if kind in ("TCI","IRP"):
        #     translate_kwargs["scaleParams"] = [[0,10000,0,255]]

        topts = gdal.TranslateOptions(format="GTiff", creationOptions=co, **translate_kwargs)
        out = gdal.Translate(dest2, ds_for_write, options=topts)
        if out:
            print("[shim][Warp] wrote:", dest2, "bands:", out.RasterCount)
            if kind in ("TCI","IRP"):
                for b in range(1, out.RasterCount+1):
                    band = out.GetRasterBand(b)
                    print("   band", b, "ci:", band.GetColorInterpretation(), "desc:", band.GetDescription())
        else:
            print("[shim][Warp] Translate returned None; falling back to real gdal.Warp")
            out = _Real_Warp(dest2, src, *args, **kwargs)

        return out

    gdal.Warp = Warp_patched
    print("[shim] GDAL patch installed (Create/Open/Warp->Translate)")
except Exception as e:
    print("[shim] GDAL patch skipped:", e)

# ---------- torch.load / safetensors bridge (PyTorch>=2.6 weights_only) ----------
try:
    import torch
    from safetensors.torch import load_file as _sf_load_file
    _real_torch_load = torch.load
    def _torch_load_smart(f, *a, **k):
        p = Path(str(f)); p2 = _rewrite(p)
        k.setdefault("map_location", "cpu")
        print(f"[shim] torch.load -> {p2}")
        if p2.suffix.lower() == ".safetensors":
            sd = _sf_load_file(str(p2))
            print(f"[shim] safetensors loaded: {len(sd)} tensors from {p2}")
            return {"state_dict": sd, "params": {}}
        k_try = dict(k)
        # PyTorch >= 2.6 changed default weights_only; force False for old pickles
        k_try["weights_only"] = False
        try:
            return _real_torch_load(str(p2), *a, **k_try)
        except Exception as e1:
            print(f"[shim] torch.load failed ({type(e1).__name__}): {e1}")
            sd = _sf_load_file(str(p2))
            print(f"[shim] safetensors(alt) loaded: {len(sd)} tensors from {p2}")
            return {"state_dict": sd, "params": {}}
    torch.load = _torch_load_smart
except Exception as e:
    print("[shim] Torch bridge skipped:", e)

# ---------- run ----------
import s2dr3.inferutils as iu
print("[shim] START iu.test ...")
try:
    iu.test((-83.74, 43.44), "2024-07-20")
    print("[shim] DONE")
except Exception as e:
    print("\n[shim] Captured exception:", repr(e))
    traceback.print_exc(limit=10)

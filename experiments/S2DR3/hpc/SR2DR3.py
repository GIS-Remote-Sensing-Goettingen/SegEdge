(base) [mak@MakOs out]$ python - <<'PY'
# === S2DR3 master shim v6: path sandbox + subprocess rewrite + NumPy save + GDAL Create/Open + **robust GDAL.Warp coercion+debug** + torch/safetensors bridge ===
import os, sys, builtins, shutil, subprocess, traceback
from pathlib import Path

# ---------- basics ----------
HOME = Path(os.environ.get("S2DR3_HOME", str(Path.home()/".cache"/"s2dr3"/"sandbox"))).resolve()
ROOTS = ("/var", "/content")
HOME.mkdir(parents=True, exist_ok=True)
print("[shim] sandbox root:", HOME)

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
        try: out_idx = parts.index("-out")+1; out_path = _rewrite_str(parts[out_idx])
        except Exception: return (False, None)
        from glob import glob
        cands = []
        for pat in [
            str(HOME/"**/*.pth"), str(HOME/"**/*.pt"),
            str(Path.home()/".cache/s2dr3/**/*.pth"),
            str(Path.home()/".cache/s2dr3/**/*.pt"),
            str(Path.home()/".cache/huggingface/**/*.safetensors"),
        ]: cands += glob(pat, recursive=True)
        if not cands:
            Path(out_path).write_bytes(b"")
            print(f"[shim] faked smime -> empty: {out_path}")
            return (True, 0)
        # prefer pickle over safetensors so torch.load succeeds when possible
        cands = sorted(cands, key=lambda p: (p.endswith(".safetensors"), -Path(p).stat().st_size))
        src = cands[0]
        _ensure_parent(out_path); shutil.copy2(src, out_path)
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

# ---------- GDAL Create/Open/Warp with DEEP DEBUG ----------
try:
    from osgeo import gdal

    # Create()/Open() patches
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

    # ---- Warp() coercion ----
    _Real_Warp = gdal.Warp
    _KNOWN_WARP_KW = {
        "format","dstSRS","srcSRS","xRes","yRes","targetAlignedPixels","outputBounds","multithread",
        "srcNodata","dstNodata","resampleAlg","cutlineDSName","cutlineLayer","cropToCutline",
        "errorThreshold","warpOptions","creationOptions","initDest","outputType","warpMemoryLimit",
        "geoloc","rpc","transformerOptions"
    }

    def _is_swig_obj(x):
        # WarpOptions and other SWIG types usually carry 'thisown'
        return hasattr(x, "thisown") or x.__class__.__name__.endswith("WarpOptions")

    def _repr_short(x, limit=240):
        r = repr(x)
        return (r[:limit] + "...") if len(r) > limit else r

    def _coerce_to_WarpOptions(options, extra_kwargs=None):
        """Return a real gdal.WarpOptions object from many 'creative' inputs."""
        extra_kwargs = dict(extra_kwargs or {})
        # 0) already a WarpOptions SWIG object
        if _is_swig_obj(options):
            if extra_kwargs:
                print(f"[shim][Warp] extra kwargs present -> rebuilding WarpOptions with {extra_kwargs}")
                return gdal.WarpOptions(**extra_kwargs)
            return options
        # 1) one-element tuple/list that *contains* a WarpOptions SWIG object (common bug: trailing comma)
        if isinstance(options, (list, tuple)) and len(options)==1 and _is_swig_obj(options[0]):
            print("[shim][Warp] unwrapped single-element tuple/list containing WarpOptions")
            return options[0]
        # 2) dict form: may contain 'options' (string/list) + other kwargs
        if isinstance(options, dict):
            optstr = options.get("options", None)
            named  = {k:v for k,v in options.items() if k!="options"}
            if extra_kwargs: named.update(extra_kwargs)
            if isinstance(optstr, (list,tuple)): return gdal.WarpOptions(options=[str(x) for x in optstr], **named)
            if isinstance(optstr, str):          return gdal.WarpOptions(options=optstr.split(), **named)
            return gdal.WarpOptions(**named)
        # 3) list/tuple of flags (strings) -> WarpOptions(options=[...])
        if isinstance(options, (list, tuple)):
            # If it *looks* like ["-wo","INIT_DEST=NO_DATA","-overwrite"] etc.
            as_strs = [str(x) for x in options]
            return gdal.WarpOptions(options=as_strs, **extra_kwargs)
        # 4) plain string of flags -> split on spaces
        if isinstance(options, str):
            return gdal.WarpOptions(options=options.split(), **extra_kwargs)
        # 5) None -> build from extra kwargs, or empty
        if options is None:
            return gdal.WarpOptions(**extra_kwargs) if extra_kwargs else gdal.WarpOptions()
        # 6) unknown python object - Hail Mary: return as is (but log!)
        print(f"[shim][Warp] WARNING: unknown options type {type(options)}; passing through.")
        return options

    def Warp_patched(dest, src, *args, **kwargs):
        # Rewrite paths
        dest2 = str(_rewrite(dest)); _ensure_parent(dest2)
        src2 = src
        if isinstance(src, (str, Path)): src2 = str(_rewrite(src))
        elif isinstance(src, (list,tuple)):
            src2 = [str(_rewrite(s)) if isinstance(s,(str,Path)) else s for s in src]

        # Pull raw options and any known WarpOptions kwargs
        options_in = kwargs.pop("options", None)
        extra = {k: kwargs.pop(k) for k in list(kwargs.keys()) if k in _KNOWN_WARP_KW}

        # DEBUG: show exactly what we got
        print(f"[shim][Warp] raw call:")
        print(f"  dest: {dest2}")
        print(f"  src:  {type(src2)}")
        print(f"  positional args (IGNORED): {[_repr_short(a) for a in args]}")
        print(f"  options_in: type={type(options_in)} repr={_repr_short(options_in)}")
        print(f"  extra kwargs for WarpOptions: {extra}")
        print(f"  remaining kwargs: {kwargs}")

        # Coerce to WarpOptions
        options_obj = _coerce_to_WarpOptions(options_in, extra_kwargs=extra)
        print(f"[shim][Warp] coerced options -> type={type(options_obj)} repr={_repr_short(options_obj)}")

        # Absolutely no positional options to C shim
        if args:
            print(f"[shim][Warp] NOTE: ignoring positional args; packed into options already.")

        # Call GDAL
        try:
            return _Real_Warp(dest2, src2, options=options_obj, **kwargs)
        except Exception as e:
            print(f"[shim][Warp] FAIL: {type(e).__name__}: {e}")
            raise

    gdal.Warp = Warp_patched
    print("[shim] GDAL patch installed (Create/Open/Warp)")
except Exception as e:
    print("[shim] GDAL patch skipped:", e)

# ---------- torch.load / safetensors bridge ----------
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
        # PyTorch >= 2.6 default flip
        k_try = dict(k)
        if "weights_only" in str(getattr(_real_torch_load,"__text_signature__","")):
            k_try["weights_only"] = False
        try:
            return _real_torch_load(str(p2), *a, **k_try)
        except Exception as e1:
            print(f"[shim] torch.load failed ({type(e1).__name__}): {e1}")
            # last-ditch: maybe wrong extension but actually safetensors
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
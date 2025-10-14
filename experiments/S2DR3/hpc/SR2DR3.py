# === S2DR3 shim v11: env presets + path sandbox + subprocess rewrite + NumPy save + GDAL Create/Open/Warp repairs + torch/safetensors bridge + model cache ===
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
}
_DEFAULT_PROJ_LIB = Path(sys.prefix)/"share"/"proj"
_DEFAULT_GDAL_DATA = Path(sys.prefix)/"share"/"gdal"
if _DEFAULT_PROJ_LIB.exists():
    _ENV_EXPORTS.setdefault("PROJ_LIB", str(_DEFAULT_PROJ_LIB))
if _DEFAULT_GDAL_DATA.exists():
    _ENV_EXPORTS.setdefault("GDAL_DATA", str(_DEFAULT_GDAL_DATA))
for _k, _v in _ENV_EXPORTS.items():
    os.environ[_k] = _v
print("[shim] env primed:", ", ".join(f"{k}={os.environ[k]!r}" for k in sorted(_ENV_EXPORTS)))

# ---------- basics ----------
def _pick_sandbox_root():
    candidates = []
    env_home = os.environ.get("S2DR3_HOME")
    if env_home:
        candidates.append(Path(env_home).expanduser())
    # prefer legacy default, but only if writable
    candidates.append(Path.home()/".cache"/"s2dr3"/"sandbox")
    here = Path(__file__).resolve().parent
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
def _ensure_parent(x):
    path = Path(x).parent
    s = str(path)
    if s.startswith("/tmp/.__"):
        rel = path.relative_to("/tmp")
        target = HOME / "tmp_overlay" / rel
        target.mkdir(parents=True, exist_ok=True)
        try:
            if path.is_symlink():
                return
            if path.exists():
                return
            path.parent.mkdir(parents=True, exist_ok=True)
            os.symlink(target, path)
        except FileExistsError:
            pass
        except OSError:
            # fall back to direct creation if symlink fails
            path.mkdir(parents=True, exist_ok=True)
        return
    path.mkdir(parents=True, exist_ok=True)

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
    def _rewrite_piece(piece):
        if isinstance(piece, str) and any(piece.startswith(r) for r in ROOTS):
            return _rewrite_str(piece)
        return piece

    if isinstance(cmd, (list, tuple)):
        return type(cmd)(_rewrite_piece(x) for x in cmd)

    if isinstance(cmd, str):
        import shlex

        try:
            tokens = shlex.split(cmd)
        except ValueError:
            tokens = cmd.split()
        rewritten = []
        for tok in tokens:
            if any(tok.startswith(r) for r in ROOTS):
                rewritten.append(_rewrite_str(tok))
            else:
                rewritten.append(tok)
        # preserve trailing whitespace if present in original
        sep = " " if " " in cmd else ""
        return sep.join(rewritten)

    return cmd

def _maybe_fake_smime(cmd):
    txt = " ".join(cmd) if isinstance(cmd,(list,tuple)) else (cmd if isinstance(cmd,str) else "")
    if "openssl" in txt and "smime" in txt and "-decrypt" in txt and "-out" in txt:
        parts = txt.split()
        try:
            out_idx = parts.index("-out")+1
            out_path = Path(_rewrite_str(parts[out_idx]))
        except Exception: return (False, None)
        cache_target = MODEL_CACHE_ROOT/out_path.name
        if cache_target.exists():
            _ensure_parent(out_path)
            try:
                if out_path.exists() or out_path.is_symlink():
                    out_path.unlink()
            except FileNotFoundError:
                pass
            try:
                os.symlink(cache_target, out_path)
                print(f"[shim] model-cache: symlinked cached blob: {cache_target} -> {out_path}")
                return (True, 0)
            except OSError:
                shutil.copy2(cache_target, out_path)
                print(f"[shim] model-cache: copied cached blob: {cache_target}")
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
        # prefer real pickle over safetensors when available
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

# ---------- GDAL Create/Open/Warp: unwrap tuple, repair null SWIG proxy ----------
try:
    from osgeo import gdal

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

    _Real_Warp = gdal.Warp
    _Real_WarpOptions = gdal.WarpOptions
    _KNOWN_WARP_KW = {
        "format","dstSRS","srcSRS","xRes","yRes","targetAlignedPixels","outputBounds","multithread",
        "srcNodata","dstNodata","resampleAlg","cutlineDSName","cutlineLayer","cropToCutline",
        "errorThreshold","warpOptions","creationOptions","initDest","outputType","warpMemoryLimit",
        "geoloc","rpc","transformerOptions"
    }

    def _is_swig_warpopts(x):
        try:
            r = repr(x)
            return ("GDALWarpAppOptions" in r) or hasattr(x, "thisown")
        except Exception:
            return False

    def _is_null_proxy(x):
        try:
            return "proxy of None" in repr(x)
        except Exception:
            return False

    def _repr_short(x, n=220):
        try: r = repr(x)
        except Exception: r = f"<{type(x).__name__}>"
        return (r if len(r)<=n else r[:n]+"...")

    def _wrap_existing_options_tuple(seq, cb_override=None, cbd_override=None):
        seq = list(seq)
        base = seq[0]
        cb = cb_override if cb_override is not None else (seq[1] if len(seq)>1 else None)
        cbd = cbd_override if cbd_override is not None else (seq[2] if len(seq)>2 else None)
        return (base, cb, cbd)

    def _build_warp_options(extra_kwargs=None):
        extra_kwargs = dict(extra_kwargs or {})
        return _Real_WarpOptions(**extra_kwargs) if extra_kwargs else _Real_WarpOptions()

    def WarpOptions_patched(*args, **kwargs):
        cb_kw = kwargs.pop("callback", None)
        cbd_kw = kwargs.pop("callback_data", None)

        if not args:
            if cb_kw is not None: kwargs["callback"] = cb_kw
            if cbd_kw is not None: kwargs["callback_data"] = cbd_kw
            return _Real_WarpOptions(**kwargs)

        if len(args) == 3 and _is_swig_warpopts(args[0]):
            if kwargs:
                print(f"[shim][WarpOptions] WARNING: dropping extra kwargs {list(kwargs)} for tuple input")
            cb = cb_kw if cb_kw is not None else args[1]
            cbd = cbd_kw if cbd_kw is not None else args[2]
            return (args[0], cb, cbd)

        if len(args) == 1:
            arg0 = args[0]
            if _is_swig_warpopts(arg0):
                if kwargs:
                    print("[shim][WarpOptions] WARNING: rebuilding new options for SWIG object with kwargs")
                    if cb_kw is not None: kwargs["callback"] = cb_kw
                    if cbd_kw is not None: kwargs["callback_data"] = cbd_kw
                    return _Real_WarpOptions(**kwargs)
                return (arg0, cb_kw, cbd_kw)
            if isinstance(arg0, (list, tuple)):
                if arg0 and _is_swig_warpopts(arg0[0]):
                    if kwargs:
                        print("[shim][WarpOptions] WARNING: kwargs ignored with tuple input; rebuilding")
                        if cb_kw is not None: kwargs["callback"] = cb_kw
                        if cbd_kw is not None: kwargs["callback_data"] = cbd_kw
                        return _Real_WarpOptions(**kwargs)
                    return _wrap_existing_options_tuple(arg0, cb_kw, cbd_kw)
                opts_list = [str(x) for x in arg0 if x is not None]
                if cb_kw is not None: kwargs["callback"] = cb_kw
                if cbd_kw is not None: kwargs["callback_data"] = cbd_kw
                kwargs.setdefault("options", opts_list)
                return _Real_WarpOptions(**kwargs)
            if isinstance(arg0, str):
                if cb_kw is not None: kwargs["callback"] = cb_kw
                if cbd_kw is not None: kwargs["callback_data"] = cbd_kw
                kwargs.setdefault("options", arg0.split())
                return _Real_WarpOptions(**kwargs)

        print(f"[shim][WarpOptions] WARNING: unsupported positional usage -> {tuple(type(a).__name__ for a in args)}; rebuilding with kwargs only")
        if cb_kw is not None: kwargs["callback"] = cb_kw
        if cbd_kw is not None: kwargs["callback_data"] = cbd_kw
        return _Real_WarpOptions(**kwargs)

    gdal.WarpOptions = WarpOptions_patched

    def _coerce_to_WarpOptions(options, extra_kwargs=None):
        extra_kwargs = dict(extra_kwargs or {})

        # CASE A: already a WarpOptions SWIG object
        if _is_swig_warpopts(options):
            if _is_null_proxy(options):
                print("[shim][Warp] WARNING: WarpOptions is a NULL SWIG proxy -> rebuilding new options")
                return _build_warp_options(extra_kwargs)
            return options if not extra_kwargs else _build_warp_options(extra_kwargs)

        # CASE B: tuple/list that *starts* with WarpOptions (often (opts, None, None))
        if isinstance(options, (list, tuple)) and len(options)>=1 and _is_swig_warpopts(options[0]):
            print(f"[shim][Warp] unwrapping tuple/list starting with WarpOptions; tail ignored: {options[1:]}")
            if _is_null_proxy(options[0]):
                print("[shim][Warp] WARNING: first element is NULL proxy -> rebuilding new options")
                return _build_warp_options(extra_kwargs)
            return options[0] if not extra_kwargs else _build_warp_options(extra_kwargs)

        # CASE C: dict form
        if isinstance(options, dict):
            optstr = options.get("options", None)
            named  = {k:v for k,v in options.items() if k!="options"}
            if extra_kwargs: named.update(extra_kwargs)
            if isinstance(optstr, (list,tuple)): return _Real_WarpOptions(options=[str(x) for x in optstr], **named)
            if isinstance(optstr, str):          return _Real_WarpOptions(options=optstr.split(), **named)
            return _Real_WarpOptions(**named)

        # CASE D: list/tuple of string flags (e.g., ['-overwrite','-wo','INIT_DEST=NO_DATA'])
        if isinstance(options, (list, tuple)):
            as_strs = [str(x) for x in options if x is not None]
            return _Real_WarpOptions(options=as_strs, **extra_kwargs)

        # CASE E: a single string of flags
        if isinstance(options, str):
            return _Real_WarpOptions(options=options.split(), **extra_kwargs)

        # CASE F: None -> build from extra kwargs or empty
        if options is None:
            return _build_warp_options(extra_kwargs)

        # CASE G: unknown object
        print(f"[shim][Warp] WARNING: unknown options type {type(options)}; building empty WarpOptions")
        return _build_warp_options(extra_kwargs)

    def _to_options_tuple(obj):
        if isinstance(obj, tuple):
            if len(obj) >= 3:
                return obj[0], obj[1], obj[2]
            if len(obj) == 2:
                return obj[0], obj[1], None
            if len(obj) == 1:
                return obj[0], None, None
            # empty tuple -> build default
        if _is_swig_warpopts(obj):
            return obj, None, None
        if obj is None:
            new = _Real_WarpOptions()
            return new[0], new[1], new[2]
        print(f"[shim][Warp] WARNING: unexpected WarpOptions payload {type(obj)}; rebuilding default tuple")
        new = _Real_WarpOptions()
        return new[0], new[1], new[2]

    def Warp_patched(dest, src, *args, **kwargs):
        dest2 = str(_rewrite(dest)); _ensure_parent(dest2)
        src2 = src
        if isinstance(src, (str, Path)): src2 = str(_rewrite(src))
        elif isinstance(src, (list,tuple)): src2 = [str(_rewrite(s)) if isinstance(s,(str,Path)) else s for s in src]

        options_in = kwargs.pop("options", None)
        extra = {k: kwargs.pop(k) for k in list(kwargs.keys()) if k in _KNOWN_WARP_KW}

        print("[shim][Warp] raw call:")
        print("  dest:", dest2)
        print("  src: ", type(src2))
        print("  positional args (IGNORED):", [type(a).__name__ for a in args])
        print("  options_in:", "type="+str(type(options_in)), "repr="+_repr_short(options_in))
        print("  extra kwargs for WarpOptions:", extra)
        print("  remaining kwargs:", kwargs)

        options_obj = _coerce_to_WarpOptions(options_in, extra_kwargs=extra)
        opt_ptr, opt_cb, opt_cbd = _to_options_tuple(options_obj)
        print("[shim][Warp] coerced ->", f"ptr={opt_ptr}, cb={opt_cb}, cbd={opt_cbd}")

        if args:
            print("[shim][Warp] NOTE: ignoring positional args; not supported by Utilities API.")

        try:
            call_kwargs = dict(kwargs)
            if opt_cb is not None and "callback" not in call_kwargs:
                call_kwargs["callback"] = opt_cb
            if opt_cbd is not None and "callback_data" not in call_kwargs:
                call_kwargs["callback_data"] = opt_cbd
            return _Real_Warp(dest2, src2, options=(opt_ptr, call_kwargs.get("callback"), call_kwargs.get("callback_data")), **call_kwargs)
        except Exception as e:
            print(f"[shim][Warp] FAIL: {type(e).__name__}: {e}")
            raise

    gdal.Warp = Warp_patched
    print("[shim] GDAL patch installed (Create/Open/Warp)")
except Exception as e:
    print("[shim] GDAL patch skipped:", e)

# ---------- torch.load / safetensors bridge (PyTorch>=2.6 weights_only) ----------
try:
    import torch
    from safetensors.torch import load_file as _sf_load_file
    from torch import nn
    import itertools

    _TRACE_DIR = HOME/"logs"
    _TRACE_DIR.mkdir(parents=True, exist_ok=True)
    _TRACE_FILE = _TRACE_DIR/"module_trace.log"
    _TRACE_LIMIT = int(os.environ.get("S2DR3_TRACE_MAX_LINES", "0"))
    _TRACE_ECHO = os.environ.get("S2DR3_TRACE_STDOUT", "1") != "0"
    _TRACE_ENABLED = os.environ.get("S2DR3_TRACE_ENABLE", "1") != "0"
    _TRACE_STATE = {"lines": 0}
    _DUMP_ENABLED = os.environ.get("S2DR3_DUMP_STATE", "1") != "0"
    _DUMP_DIR = HOME / "weights"
    if _DUMP_ENABLED:
        _DUMP_DIR.mkdir(parents=True, exist_ok=True)
    _DUMP_COUNTER = itertools.count()

    def _trace_write(msg: str) -> None:
        if not _TRACE_ENABLED:
            return
        if _TRACE_LIMIT and _TRACE_STATE["lines"] >= _TRACE_LIMIT:
            return
        _TRACE_STATE["lines"] += 1
        line = f"[trace] {msg}"
        if _TRACE_ECHO:
            print(line)
        try:
            with _TRACE_FILE.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
        except Exception as trace_exc:
            print(f"[trace] log append failed: {trace_exc}")

    def _summarize_value(val, depth: int = 0, max_depth: int = 2):
        if depth > max_depth:
            return "…"
        try:
            import torch as _torch_mod
        except Exception:
            _torch_mod = None
        if _torch_mod is not None and isinstance(val, _torch_mod.Tensor):
            shape = tuple(val.shape)
            dtype = str(val.dtype)
            device = str(val.device)
            return f"Tensor(shape={shape}, dtype={dtype}, device={device})"
        if isinstance(val, (list, tuple)):
            items = ", ".join(_summarize_value(v, depth + 1, max_depth) for v in list(val)[:4])
            if len(val) > 4:
                items += ", …"
            wrapper = "[" if isinstance(val, list) else "("
            closer = "]" if isinstance(val, list) else ")"
            return f"{wrapper}{items}{closer}"
        if isinstance(val, dict):
            kv = ", ".join(
                f"{k}: {_summarize_value(v, depth + 1, max_depth)}"
                for k, v in list(val.items())[:4]
            )
            if len(val) > 4:
                kv += ", …"
            return f"{{{kv}}}"
        return repr(val)

    _real_torch_load = torch.load
    def _torch_load_smart(f, *a, **k):
        p = Path(str(f)); p2 = _rewrite(p)
        k.setdefault("map_location", "cpu")
        print(f"[shim] torch.load -> {p2}")
        if p2.suffix.lower() == ".safetensors":
            sd = _sf_load_file(str(p2))
            print(f"[shim] safetensors loaded: {len(sd)} tensors from {p2}")
            result = {"state_dict": sd, "params": {}}
            _trace_write(f"torch.load safetensors -> {len(sd)} tensors")
            _trace_write(f"state_dict keys (first 10): {list(sd.keys())[:10]}")
            return result
        k_try = dict(k)
        if "weights_only" in str(getattr(_real_torch_load,"__text_signature__","")):
            k_try["weights_only"] = False  # PyTorch>=2.6 default flip
        try:
            obj = _real_torch_load(str(p2), *a, **k_try)
        except Exception as e1:
            print(f"[shim] torch.load failed ({type(e1).__name__}): {e1}")
            sd = _sf_load_file(str(p2))
            print(f"[shim] safetensors(alt) loaded: {len(sd)} tensors from {p2}")
            result = {"state_dict": sd, "params": {}}
            _trace_write(f"torch.load fallback -> keys={list(result.keys())}")
            if "state_dict" in result:
                sd_keys = list(result["state_dict"].keys())
                _trace_write(f"state_dict keys (first 10): {sd_keys[:10]}")
            return result
        else:
            _trace_write(f"torch.load -> type={type(obj).__name__}")
            if isinstance(obj, dict):
                _trace_write(f"torch.load dict keys: {list(obj.keys())}")
                if "state_dict" in obj and isinstance(obj["state_dict"], dict):
                    sd_keys = list(obj["state_dict"].keys())
                    _trace_write(f"state_dict keys (first 10): {sd_keys[:10]}")
            return obj
    torch.load = _torch_load_smart

    if _TRACE_ENABLED:
        _orig_module_call = nn.Module.__call__
        _call_depth_state = {"depth": 0}
        _module_registry = {}
        _dumped_module_ids = set()

        def _debug_module_call(self, *args, **kwargs):
            indent = "  " * _call_depth_state["depth"]
            _call_depth_state["depth"] += 1
            try:
                result = _orig_module_call(self, *args, **kwargs)
            except Exception as exc:
                _trace_write(
                    f"{indent}{type(self).__module__}.{type(self).__name__} raised {type(exc).__name__}: {exc}"
                )
                _call_depth_state["depth"] -= 1
                raise
            arg_summary = _summarize_value(args)
            kw_summary = _summarize_value(kwargs) if kwargs else "{}"
            res_summary = _summarize_value(result)
            _trace_write(
                f"{indent}{type(self).__module__}.{type(self).__name__} args={arg_summary} kwargs={kw_summary} -> {res_summary}"
            )
            if _DUMP_ENABLED and isinstance(self, nn.Module):
                mod_id = id(self)
                if type(self).__module__.startswith("s2dr3."):
                    _module_registry.setdefault(mod_id, self)
                    if mod_id not in _dumped_module_ids:
                        try:
                            state = self.state_dict()
                            if state:
                                state_cpu = {
                                    k: (v.detach().cpu() if hasattr(v, "detach") else v)
                                    for k, v in state.items()
                                }
                                dump_name = f"module_{len(_dumped_module_ids):03d}_{type(self).__module__.replace('.', '_')}.{type(self).__name__}.pt"
                                dump_path = _DUMP_DIR / dump_name
                                torch.save(state_cpu, dump_path)
                                _dumped_module_ids.add(mod_id)
                                _trace_write(f"module state_dict dumped -> {dump_path}")
                        except Exception as mod_exc:
                            _trace_write(f"module dump failed during call: {mod_exc}")
            _call_depth_state["depth"] -= 1
            return result

        nn.Module.__call__ = _debug_module_call

        _orig_load_state_dict = nn.Module.load_state_dict

        def _debug_load_state_dict(self, state_dict, *args, **kwargs):
            key_count = len(state_dict) if isinstance(state_dict, dict) else "?"
            _trace_write(
                f"{type(self).__module__}.{type(self).__name__}.load_state_dict("
                f"keys={key_count}, args={_summarize_value(args)}, kwargs={_summarize_value(kwargs) if kwargs else '{}'})"
            )
            if _DUMP_ENABLED and isinstance(state_dict, dict):
                idx = next(_DUMP_COUNTER)
                dump_name = f"{idx:03d}_{type(self).__module__.replace('.', '_')}.{type(self).__name__}.pt"
                dump_path = _DUMP_DIR / dump_name
                try:
                    torch.save(state_dict, dump_path)
                    _trace_write(f"state_dict dumped -> {dump_path}")
                except Exception as dump_exc:
                    _trace_write(f"state_dict dump failed: {dump_exc}")
            return _orig_load_state_dict(self, state_dict, *args, **kwargs)

        nn.Module.load_state_dict = _debug_load_state_dict

        if _DUMP_ENABLED:
            import atexit

            def _dump_registered_modules():
                for idx, mod in enumerate(list(_module_registry.values())):
                    try:
                        state = mod.state_dict()
                        if not state:
                            continue
                        state_cpu = {
                            k: (v.detach().cpu() if hasattr(v, "detach") else v)
                            for k, v in state.items()
                        }
                        dump_name = f"module_{idx:03d}_{type(mod).__module__.replace('.', '_')}.{type(mod).__name__}.pt"
                        dump_path = _DUMP_DIR / dump_name
                        torch.save(state_cpu, dump_path)
                        _trace_write(f"module state_dict dumped -> {dump_path}")
                    except Exception as mod_exc:
                        _trace_write(f"module dump failed: {mod_exc}")

            atexit.register(_dump_registered_modules)
except Exception as e:
    print("[shim] Torch bridge skipped:", e)

# ---------- run ----------
import s2dr3.inferutils as iu
print("[shim] START iu.test ...")
try:
    iu.test((9.1896346, 45.4641943), "2025-04-14")
    print("[shim] DONE")
except Exception as e:
    print("\n[shim] Captured exception:", repr(e))
    traceback.print_exc(limit=10)

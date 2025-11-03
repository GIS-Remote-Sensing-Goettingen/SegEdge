#!/usr/bin/env python
import argparse
from pathlib import Path
import torch
from omegaconf import OmegaConf
import opensr_model
import opensr_utils

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-tif", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--factor", type=int, default=4)
    ap.add_argument("--window-size", type=int, nargs=2, default=(128,128))
    ap.add_argument("--overlap", type=int, default=12)
    ap.add_argument("--eliminate-border-px", type=int, default=2)
    ap.add_argument("--gpus", type=int, default=0)
    ap.add_argument("--save-preview", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device:", device)
    cfg = OmegaConf.load(Path("config_10m.yaml"))

    model = opensr_model.SRLatentDiffusion(cfg, device=device)
    model.load_pretrained(cfg.ckpt_version)
    assert model.training is False

    _ = opensr_utils.large_file_processing(
        root=args.input_tif,
        model=model,
        window_size=tuple(args.window_size),
        factor=args.factor,
        overlap=args.overlap,
        eliminate_border_px=args.eliminate_border_px,
        device=device,
        gpus=args.gpus,
        save_preview=args.save_preview,
        debug=False,
    )

if __name__ == "__main__":
    main()

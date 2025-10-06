import rasterio, numpy as np, torch
from PIL import Image
from transformers import pipeline

src = "./images/1084-1393.tif"; out = "fields_mask.tif"

with rasterio.open(src) as ds:
    rgb = ds.read([1,2,3]).astype("uint8"); H, W = ds.height, ds.width
    transform, crs = ds.transform, ds.crs
pil = Image.fromarray(np.moveaxis(rgb, 0, -1))
pil.thumbnail((2048, 2048))
gen = pipeline("mask-generation", model="facebook/sam2-hiera-large", device=0, dtype=torch.float32)
res = gen(pil, points_per_batch=8)                      # smaller batch → less VRAM
masks = [ (m.cpu().numpy() if hasattr(m, "cpu") else np.asarray(m)) > 0 for m in res["masks"] ]
union = (np.any(np.stack(masks, 0), 0) if masks else np.zeros(pil.size[::-1], bool)).astype("uint8")
union = np.array(Image.fromarray(union*255).resize((W, H), Image.NEAREST)) // 255
with rasterio.open(out, "w", driver="GTiff", height=H, width=W, count=1, dtype="uint8",
                   transform=transform, crs=crs) as dst:
    dst.write(union, 1)
print(f"Saved {out} with {len(masks)} masks")
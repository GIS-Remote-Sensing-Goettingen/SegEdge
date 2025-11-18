#!/usr/bin/env python
import argparse, os, re
from math import floor
import numpy as np
import cubo
import rioxarray  # activates .rio accessor
from pyproj import CRS

def guess_utm_epsg(lat: float, lon: float) -> int:
    """
    Fallback: approximate UTM EPSG from latitude/longitude.

    We assume WGS84 and compute the standard 6-degree zone. Values near poles
    clamp to the maximum UTM-supported latitude (~84°N/S).
    """
    # Clamp latitude to valid UTM range
    lat = max(min(lat, 84.0), -80.0)
    zone = int(floor((lon + 180.0) / 6.0)) + 1
    if lat >= 0:
        epsg = 32600 + zone
    else:
        epsg = 32700 + zone
    return epsg


def cubo_to_rgbnir_geotiff(lat, lon, start_date, end_date,
                           edge_size, resolution, image_index,
                           out_path, as_uint16=True, nodata=0):
    da = cubo.create(
        lat=lat, lon=lon,
        collection="sentinel-2-l2a",
        bands=["B04", "B03", "B02", "B08"],
        start_date=start_date, end_date=end_date,
        edge_size=edge_size, resolution=resolution,
    )
    if "time" in da.dims:
        da = da.isel(time=image_index)
    da = da.transpose("band", "y", "x")

    epsg_text = str(da.attrs.get("epsg", "") or da.coords.get("epsg", ""))
    m = re.search(r"(\d{4,5})", epsg_text)
    if m:
        epsg_code = int(m.group(1))
    else:
        epsg_code = guess_utm_epsg(lat, lon)
        print(f"[WARN] Could not parse EPSG from {epsg_text!r}; using fallback {epsg_code}.")
    da = da.rio.write_crs(int(epsg_code), inplace=False)

    arr = da
    if as_uint16:
        a = arr.data
        if a.dtype.kind == "f" and np.nanmax(a) <= 1.1:
            a = np.clip(a * 10000.0, 0, 10000).astype("uint16")
            arr = arr.copy(data=a)
        else:
            arr = arr.astype("uint16")
    arr = arr.rio.write_nodata(nodata, encoded=True, inplace=False)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    arr.rio.to_raster(
        out_path, compress="deflate", tiled=True,
        blockxsize=512, blockysize=512, BIGTIFF="YES"
    )
    return os.path.abspath(out_path)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lat", type=float, required=True)
    ap.add_argument("--lon", type=float, required=True)
    ap.add_argument("--start-date", required=True)
    ap.add_argument("--end-date", required=True)
    ap.add_argument("--edge-size", type=int, default=512)
    ap.add_argument("--resolution", type=int, default=10)
    ap.add_argument("--image-index", type=int, default=0)
    ap.add_argument("--out-path", required=True)
    args = ap.parse_args()
    tif = cubo_to_rgbnir_geotiff(
        lat=args.lat, lon=args.lon,
        start_date=args.start_date, end_date=args.end_date,
        edge_size=args.edge_size, resolution=args.resolution,
        image_index=args.image_index,
        out_path=args.out_path, as_uint16=True, nodata=0,
    )
    print("Wrote:", tif)

if __name__ == "__main__":
    main()

from pathlib import Path


if __name__ == "__main__":
    # Import Model Package
    import opensr_model

    import torch
    from omegaconf import OmegaConf

    import cubo
    import numpy as np




    # Set Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)
    config = OmegaConf.load(Path("config_10m.yaml"))

    # 0.0 Create Model
    model = opensr_model.SRLatentDiffusion(config, device=device)  # create model
    model.load_pretrained(config.ckpt_version)  # download checkpint
    assert model.training == False, "Model has to be in eval mode."

    # Define Coordinates and Time
    LATITUDE =51.5413
    LONGITUDE = 9.9158
    START_DATE = "2025-04-14"  # Set date (be mindful of cloud cover)
    END_DATE = "2025-04-30"


    import opensr_utils


    def cubo_to_rgbnir_geotiff(
            lat, lon, start_date, end_date,
            edge_size, resolution=10,
            image_index=0, reducer=None,
            out_path="input_LR_image.tif",
            as_uint16=True, nodata=0,
    ):
        import os, re
        import numpy as np
        import cubo
        import xarray as xr
        import rioxarray  # activates .rio accessor

        # 1) Create cube: RGBNIR in S2 L2A
        da = cubo.create(
            lat=lat, lon=lon,
            collection="sentinel-2-l2a",
            bands=["B04", "B03", "B02", "B08"],  # R,G, B, NIR
            start_date=start_date, end_date=end_date,
            edge_size=edge_size, resolution=resolution,
        )  # dims usually: (time, band, y, x)

        # 2) Select time slice
        if "time" in da.dims:
            da = da.isel(time=image_index) if reducer is None else da.reduce(np.nanmedian, dim="time")

        # 3) Reorder to [band, y, x]
        da = da.transpose("band", "y", "x")

        # 4) Extract EPSG from attrs/coord, write CF-compliant CRS
        epsg_text = (str(da.attrs.get("epsg", "")) or str(getattr(da.coords.get("epsg", None), "item", lambda: "")()))
        m = re.search(r"(\d{4,5})", epsg_text)  # e.g., 'EPSG:32632' -> 32632
        assert m, f"Could not parse EPSG from '{epsg_text}'"
        da = da.rio.write_crs(int(m.group(1)), inplace=False)  # CF-compliant CRS

        # (optional) ensure spatial dims are recognized as x/y
        # da = da.rio.set_spatial_dims(x_dim="x", y_dim="y", inplace=False)

        # 5) Type & scaling
        arr = da
        if as_uint16:
            a = arr.data
            if a.dtype.kind == "f" and (np.nanmax(a) <= 1.1):
                a = np.clip(a * 10000.0, 0, 10000).astype("uint16")
                arr = arr.copy(data=a)
            else:
                arr = arr.astype("uint16")

        arr = arr.rio.write_nodata(nodata, encoded=True, inplace=False)

        # 6) Write GeoTIFF (transform inferred from x/y coords)
        arr.rio.to_raster(
            out_path,
            compress="deflate", tiled=True,
            blockxsize=512, blockysize=512,
            BIGTIFF="YES"
        )
        return os.path.abspath(out_path)


    lr_tif = cubo_to_rgbnir_geotiff(
        lat=LATITUDE, lon=LONGITUDE,
        start_date=START_DATE, end_date=END_DATE,
        edge_size=512,  # 128 in your case
        resolution=10,
        image_index=0,
        out_path="./input_LR_image.tif",
        as_uint16=True, nodata=0,
    )

    sr_job = opensr_utils.large_file_processing(
        root=lr_tif,  # use the exact same path you just wrote
        model=model,
        window_size=(128, 128),
        factor=4,
        overlap=12,
        eliminate_border_px=2,
        device=device,
        gpus=0,
    )


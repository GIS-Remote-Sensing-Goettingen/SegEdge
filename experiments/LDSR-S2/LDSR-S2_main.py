from pathlib import Path


def get_s2_scene(LATITUDE,LONGITUDE,START_DATE,END_DATE):
  # In the START_DATE & END_DATE range, there can be multiple images. This index is used to select one of them.
  IMAGE_INDEX = 0

  # Create a Sentinel-2 L2A data cube for a specific location and date range
  da = cubo.create(
      lat=LATITUDE, # set coor's
      lon=LONGITUDE,
      collection="sentinel-2-l2a", # select product
      bands=["B04", "B03", "B02", "B08"], # Define RGB-NIR
      start_date=START_DATE, # set dates
      end_date=END_DATE,
      edge_size=128, # get only patch size which model expects
      resolution=10) # get native 10m resolution

  # preprocess image: normalize and unsqueeze
  original_s2_numpy = (da[IMAGE_INDEX].compute().to_numpy()).astype("float32")
  low_resolution = torch.from_numpy(original_s2_numpy).float().to(device)
  low_resolution = low_resolution / 10_000  # bring to 0..1
  low_resolution = low_resolution.unsqueeze(0)  # bring to Bx4x128x128
  return low_resolution

if __name__ == "__main__":
    # Import Model Package
    import opensr_model
    import time
    # other inmports
    import torch
    from omegaconf import OmegaConf
    from importlib.resources import files
    from io import StringIO
    import requests
    from IPython.display import Image, display
    import cubo
    import numpy as np



    def save_sr_image(sr, out_file='sr image.png'):
        import matplotlib.pyplot as plt
        # apply same ops as the original function
        sr = sr.cpu() * 3.5
        sr = sr.clamp(0.001, 0.9999)
        img = sr[0, :3].permute(1, 2, 0).numpy()  # HxWxC
        plt.imsave(out_file, img)


    # Set Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(device)




    config = OmegaConf.load(Path("/home/mak/PycharmProjects/SegEdge/experiments/LDSR-S2/config_10m.yaml"))

    # 0.0 Create Model
    model = opensr_model.SRLatentDiffusion(config, device=device)  # create model
    model.load_pretrained(config.ckpt_version)  # download checkpint
    assert model.training == False, "Model has to be in eval mode."

    # Define Coordinates and Time
    LATITUDE =51.583974
    LONGITUDE = 11.596080
    START_DATE = "2025-08-18"  # Set date (be mindful of cloud cover)
    END_DATE = "2025-08-20"

    # Get image - Shape needs to be Bx4x128x128
    lr = get_s2_scene(LATITUDE, LONGITUDE, START_DATE, END_DATE)

    times = time.time()
    sr = model.forward(lr, sampling_steps=200)

    sr

    print("Inference Time (s): ", time.time() - times)

    from opensr_model.utils import plot_example

    plot_example(lr, sr, out_file="example.png")
    save_sr_image(sr, out_file="sr_image.png")
    display(Image(filename="example.png"))

    # Run Encertainty Map Generation
    # Due to the many samples needed, this might take a moment.
    uncertainty_map = model.uncertainty_map(lr, n_variations=20, sampling_steps=100)

    # Inspect Plot Result
    from opensr_model.utils import plot_uncertainty, plot_example

    plot_uncertainty(uncertainty_map, out_file="uncertainty_map.png", normalize=True)
    display(Image(filename="uncertainty_map.png"))



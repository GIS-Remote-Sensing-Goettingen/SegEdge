# **Bibliography Repository**

## 🛰️ Super-Resolution for Remote Sensing

* **Satlas SR (Allen Institute)**
  *World-scale Super-resolution of Satellite Imagery*
  ESRGAN-based temporal SR for Sentinel-2; trained with NAIP-S2 dataset.
  [arXiv:2311.18082](https://arxiv.org/pdf/2311.18082)

* **NAIP-S2 Dataset**
  *NAIP-S2: A Paired Dataset for Training SR Models on Sentinel-2 Data*
  Basis for Satlas training.
  [arXiv:2211.15660](https://arxiv.org/abs/2211.15660)

* **DSen2**
  *Super-Resolution of Sentinel-2 Images: Learning a Globally Applicable Model*
  CNN-based, globally trained on Sentinel-2.
  [arXiv:1803.04271](https://arxiv.org/abs/1803.04271)

* **SEN2SR (ESA OpenSR)**
  ESA official CNN-based SR; 2.5 m GSD, radiometric consistency.
  [GitHub](https://github.com/ESAOpenSR/opensr)

* **S2DR3**
  DevelopmentSeed; FCN upscaling all 12 Sentinel-2 bands to 1 m.
  [Medium overview](https://medium.com/@dan.akhtman/sentinel-2-deep-resolution-fc8f300b1834)

* **SEN4X**
  *Beyond Pretty Pictures: Combined Single- and Multi-Image Super-Resolution for Sentinel-2*
  [arXiv:2505.24799](https://arxiv.org/pdf/2505.24799)

* **Swin2-MoSE**
  *Transformer-based Multi-Expert SR for Remote Sensing*
  [arXiv:2404.18924](https://arxiv.org/pdf/2404.18924)

* **StarSRGAN**
  *Lightweight Real-Time SR with GAN Architecture*
  [arXiv:2307.16169](https://arxiv.org/abs/2307.16169)

* **GAN Variants for SR**
  *Satellite Image Super-Resolution Using GANs*
  [ResearchGate link](https://www.researchgate.net/publication/392194671)

---

## 🌱 Segmentation Models

* **DeepLab v3+**
  Atrous spatial pyramid pooling, solid baseline.
  [arXiv:1802.02611](https://arxiv.org/abs/1802.02611)

* **ResUNet-a**
  U-Net with residual atrous blocks for Sentinel-2.
  [GitHub](https://github.com/JanMarcelKezmann/ResUNet-a)

* **D-LinkNet**
  Thin-object extraction (roads, hedgerows).
  [arXiv:1804.08711](https://arxiv.org/abs/1804.08711)

* **SCNN (Spatial CNN)**
  Lane/linear structure detection with spatial convolutions.
  [arXiv:1712.06080](https://arxiv.org/abs/1712.06080)

* **SegFormer**
  Transformer-based segmentation, efficient.
  [HuggingFace model](https://huggingface.co/nvidia/segformer-b2-finetuned-ade-512-512)

* **Swin-UNet**
  Hybrid U-Net + Swin Transformer.
  [GitHub](https://github.com/HuCaoFighting/SwinUNet-main)

* **Mask2Former**
  Latest unified transformer segmentation model.
  [GitHub](https://github.com/facebookresearch/Mask2Former)

* **U-TAE**
  Temporal Attention Encoder for multi-temporal Sentinel-2.
  [GitHub](https://github.com/VSainteuf/utae)

---

## 🧩 SAM-based & Foundation Models

* **SAM 2 (Meta AI)**
  *Segment Anything 2*
  [arXiv:2404.19902](https://arxiv.org/abs/2404.19902)

* **RS2-SAM 2**
  *RS2-SAM2: Segment Anything for Referring Remote Sensing Image Segmentation*
  [arXiv:2404.18732](https://arxiv.org/abs/2404.18732)

* **SAMWS**
  Weak supervision + SAM2 for crop mapping.
  [GitHub](https://github.com/chrieke/sam-ws)

* **RemoteSAM**
  Fine-tuned SAM on 300k EO chips.
  [GitHub](https://github.com/whr946321/RemoteSAM)

---

## 🧪 UAV Multi-Angle / BRDF Studies

* **Roosjen et al. 2017**
  *Mapping Reflectance Anisotropy of a Potato Canopy Using UAV*
  UAV hyperspectral, RPV fitting, canopy structure links.
  Remote Sens. 2017, 9, 417
  [MDPI link](https://doi.org/10.3390/rs9050417)&#x20;

* **Cao et al. 2023**
  *The Method of Multi-Angle Remote Sensing Observation Based on UAV and BRDF Validation*
  Compares M\_Walthall, RPV, RTLSR; RPV best for rough surfaces.
  Remote Sens. 2023, 15, 5000
  [MDPI link](https://doi.org/10.3390/rs15205000)&#x20;

---

## 📊 Datasets & Benchmarks

* **ESA OpenSR Benchmark**
  Evaluation suite for Sentinel-2 SR.
  [GitHub](https://github.com/ESAOpenSR/opensr-test)

* **WorldStrat**
  Sentinel-2 & Pléiades paired dataset for SR.
  [GitHub](https://github.com/satellite-image-deep-learning/datasets)

* **NAIP-S2 Dataset**
  Sentinel-2 ↔ NAIP for SR model training.
  [arXiv:2211.15660](https://arxiv.org/abs/2211.15660)

---

This stripped-down format should save you from drowning in markdown tables. If you actually want to generate BibTeX for all of these (so your reference manager doesn’t hate you), I can crank that out next. Want me to?

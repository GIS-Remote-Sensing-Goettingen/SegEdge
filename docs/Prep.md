Here's your **updated master thesis sources markdown**, with all relevant models, papers, tools, and benchmark datasets mentioned throughout our conversation, including those you provided and those I referenced:

---

# **Master Thesis Sources**

## 📚 Models

### 🔼 Super-Resolution Models
Of course, here is the Markdown table for the super-resolution models with the requested information:

### 🔼 Super-Resolution Models

| Model | Architecture Family | Best-Fit Use Case | Multispectral Support* | Strengths for Hedgerows / Thin Lines | Open-source? |
|---|---|---|---|---|---|
| **Satlas SR (Allen Institute)** | ESRGAN-based | Global-scale super-resolution of Sentinel-2 for environmental monitoring and change detection. | Yes (trained on 12-band Sentinel-2 data). | Enhanced detail from temporal stacking can improve definition of linear features over time. | Yes |
| **SEN2SR (ESA OpenSR)** | CNN-based (includes lightweight SWIN and latent diffusion models). | Super-resolving Sentinel-2 10m and 20m bands to 2.5m, with a focus on radiometric and spatial consistency. | Yes (designed for Sentinel-2 bands). | Aims to preserve spectral information, which is crucial for distinguishing vegetation types in hedgerows. | Yes |
| **S2DR3** | Fully Convolutional Network (similar to ESRGAN). | Upscaling all 12 Sentinel-2 bands to 1m for applications like precision agriculture and environmental monitoring. | Yes (optimised for all 12 Sentinel-2 bands). | Preserves spectral fidelity, aiding in the identification of fine-scale vegetation patterns. | Inference code available. |
| **DSen2** | Deep Neural Network. | Globally applicable super-resolution of Sentinel-2 images. | Yes (works with Sentinel-2 data). | By improving the overall resolution, it can help in delineating narrow features like hedgerows. | Yes |
| **SEN4X** | Not specified. | Combined single- and multi-image super-resolution for Sentinel-2. | Not specified. | Not specified. | Not specified. |
| **Swin2-MoSE** | Transformer-based (enhanced Swin2SR). | Single-image super-resolution for remote sensing, with applications in semantic segmentation. | Yes (demonstrated on multispectral datasets like Sen2Venus). | The transformer architecture can capture global context, potentially improving the continuity of long, thin features. | Yes |
| **GAN Variants** | Generative Adversarial Network (GAN). | General satellite image super-resolution, with a focus on creating visually realistic high-resolution images. | Can be adapted for multispectral data, though many common variants are RGB-focused. | GANs can generate sharp textures, which might enhance the appearance of vegetation in hedgerows. | Varies by implementation. |
| **StarSRGAN** | GAN-based with Multi-scale and Attention U-Net Discriminators. | Real-world blind super-resolution, aiming to improve upon models like Real-ESRGAN with better perceptual quality. | Not explicitly designed for multispectral, focuses on visual quality. | The attention mechanism could potentially focus on and better reconstruct fine details like thin lines. | Yes |

#### **Satlas SR (Allen Institute)**

* **Overview & Code**:
  [https://github.com/allenai/satlas-super-resolution](https://github.com/allenai/satlas-super-resolution)
  ESRGAN-based temporal super-resolution (SR) model for Sentinel-2; includes full pipeline.
* **Pretrained Models**:
  [https://github.com/allenai/satlaspretrain\_models/tree/main](https://github.com/allenai/satlaspretrain_models/tree/main)
* **Model Paper**:
  *World-scale Super-resolution of Satellite Imagery*
  [https://arxiv.org/pdf/2311.18082](https://arxiv.org/pdf/2311.18082)
* **Dataset Paper**:
  *NAIP-S2: A Paired Dataset for Training SR Models on Sentinel-2 Data*
  [https://arxiv.org/abs/2211.15660](https://arxiv.org/abs/2211.15660)

---

#### **SEN2SR (ESA OpenSR)**

* **GitHub & Docs**:
  [https://github.com/ESAOpenSR/opensr](https://github.com/ESAOpenSR/opensr)
  Pretrained CNN models for 2×/4× SR of Sentinel-2 bands to 2.5 m GSD.
* **Benchmark Tool**:
  [https://github.com/ESAOpenSR/opensr-test](https://github.com/ESAOpenSR/opensr-test)

---

#### **S2DR3**

* **Overview**:
  Fully convolutional SR model trained on aerial-to-Sentinel-2 pairs; outputs all 12 bands at 1 m GSD.
* **Author Post**:
  [https://medium.com/@dan.akhtman/sentinel-2-deep-resolution-fc8f300b1834](https://medium.com/@dan.akhtman/sentinel-2-deep-resolution-fc8f300b1834)
* **Google Colab (inference)**:
  [https://colab.research.google.com/github/developmentseed/s2dr3/blob/main/notebooks/inference.ipynb](https://colab.research.google.com/github/developmentseed/s2dr3/blob/main/notebooks/inference.ipynb)

---

#### **DSen2**

* **Paper**:
  *Super-Resolution of Sentinel-2 Images: Learning a Globally Applicable Model*
  [https://arxiv.org/abs/1803.04271](https://arxiv.org/abs/1803.04271)
* **Codebase**:
  [https://github.com/lanha/DSen2](https://github.com/lanha/DSen2)

---

#### **SEN4X**

* **Paper**:
  *Beyond Pretty Pictures: Combined Single- and Multi-Image Super-Resolution for Sentinel-2*
  [https://arxiv.org/pdf/2505.24799](https://arxiv.org/pdf/2505.24799)

---

#### **Swin2-MoSE**

* **Paper**:
  *Transformer-based Multi-Expert SR for Remote Sensing*
  [https://arxiv.org/pdf/2404.18924](https://arxiv.org/pdf/2404.18924)

---

#### **GAN Variants**

* **Paper**:
  *Satellite Image Super-Resolution Using GANs*
  [https://www.researchgate.net/publication/392194671](https://www.researchgate.net/publication/392194671)

#### **StarSRGAN**

* **Paper**:
  *Lightweight Real-Time SR with GAN Architecture*
  [https://arxiv.org/abs/2307.16169](https://arxiv.org/abs/2307.16169)

---

### 🧠 Segmentation Models




| Model                                    | Architecture Family                                     | Best-Fit Use Case                                                       | Multispectral Support\*                              | Strengths for Hedgerows / Thin Lines                                                             | Open-source?             |
| ---------------------------------------- | ------------------------------------------------------- | ----------------------------------------------------------------------- | ---------------------------------------------------- | ------------------------------------------------------------------------------------------------ | ------------------------ |
| **DeepLab v3+** (ResNet-50/101 backbone) | Encoder–decoder CNN with Atrous Spatial Pyramid Pooling | General land-cover & object segmentation on SR Sentinel-2               | ✔️ (replace first conv layer for N bands)            | Multi-scale context from ASPP recovers long, narrow objects; strong baselines & ImageNet weights | Yes (many PyTorch repos) |
| **ResUNet-a**                            | U-Net + residual + atrous blocks                        | Multispectral remote-sensing segmentation where class imbalance is high | ✔️ Native (authors trained on 13-band Sentinel-2)    | Dilated residual blocks keep resolution while enlarging receptive field—good edge sharpness      | Yes                      |
| **D-LinkNet**                            | Dilated encoder + LinkNet decoder                       | Road, levee, hedgerow extraction (thin linear features)                 | ✔️ (adapt first conv layer)                          | Spatial dilations + skip links preserve continuity of 1-pixel-wide lines; lightweight            | Yes                      |
| **SAM 2** (Meta AI)                      | Vision Transformer with Mask Decoder & Streaming Memory | Prompt-based, zero-shot segmentation; interactive mapping               | RGB-only by default (needs adapters for extra bands) | Memory module enforces temporal/patch consistency; excels at interactive tracing                 | Yes                      |
| **RemoteSAM**                            | SAM v1/v2 fine-tuned on EO chips (LoRA adapters)        | Zero-/few-shot segmentation on multi-sensor data                        | ✔️ (models released for 4- & 13-band configs)        | Keeps full SAM promptability but understands spectral textures; ready weights                    | Yes                      |
| **SAM-Instruct**                         | Instruction-tuned SAM 2 variant                         | Language-guided or prompt-rich EO segmentation                          | RGB-only (multispectral branch in progress)          | Natural-language prompts; good for quick QA of hedge maps                                        | Yes                      |



#### **DeepLab v3+**

* Strong performer on linear features in Sentinel-2 (reported >75% F1 on hedgerows).
* GitHub (PyTorch):
  [https://github.com/VainF/DeepLabV3Plus-Pytorch](https://github.com/VainF/DeepLabV3Plus-Pytorch)

---

#### **U-Net & ResUNet-a Variants**

* **UNet++ (Keras)**:
  [https://github.com/MrGiovanni/UNetPlusPlus](https://github.com/MrGiovanni/UNetPlusPlus)
* **ResUNet-a for multispectral data**:
  [https://github.com/JanMarcelKezmann/ResUNet-a](https://github.com/JanMarcelKezmann/ResUNet-a)

---

#### **D-LinkNet**

* Designed for road and thin-object extraction; performs well on long vegetative lines.
  [https://github.com/ydjwgy/D-LinkNet](https://github.com/ydjwgy/D-LinkNet)

---

#### **SCNN (Spatial CNN)**

* Original paper:
  *Spatial As Deep: Spatial CNN for Traffic Lane Detection*
  [https://arxiv.org/abs/1712.06080](https://arxiv.org/abs/1712.06080)

---

#### **SegFormer**

* Transformer-based, high-efficiency segmentation on remote sensing imagery.
  Hugging Face model:
  [https://huggingface.co/nvidia/segformer-b2-finetuned-ade-512-512](https://huggingface.co/nvidia/segformer-b2-finetuned-ade-512-512)

---

#### **Swin-UNet**

* Hierarchical transformer + U-Net hybrid.
  [https://github.com/HuCaoFighting/SwinUNet-main](https://github.com/HuCaoFighting/SwinUNet-main)

---

#### **Mask2Former**

* Latest transformer-based unified segmentation architecture.
  GitHub:
  [https://github.com/facebookresearch/Mask2Former](https://github.com/facebookresearch/Mask2Former)

---

#### **U-TAE / Temporal Models**

* Temporal segmentation models using revisit stacks (3D CNN or attention encoder).
  [https://github.com/VSainteuf/utae](https://github.com/VSainteuf/utae)

---

## 🛰️ SAM-based & Foundation Models

#### **SAM 2 (Meta AI)**

* Paper:
  [https://arxiv.org/abs/2404.19902](https://arxiv.org/abs/2404.19902)
* GitHub (official):
  [https://github.com/facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything)
* Ultralytics SAM2 adaptation:
  [https://github.com/ultralytics/ultralytics/tree/main/ultralytics/models/sam](https://github.com/ultralytics/ultralytics/tree/main/ultralytics/models/sam)

---

#### **RS2-SAM 2**

* Paper only (no code):
  *RS2-SAM2: Segment Anything for Referring Remote Sensing Image Segmentation*
  [https://arxiv.org/abs/2404.18732](https://arxiv.org/abs/2404.18732)

---

#### **SAMWS**

* Crop-type segmentation with SAM 2 + weak supervision.
  [https://github.com/chrieke/sam-ws](https://github.com/chrieke/sam-ws)

---

#### **RemoteSAM**

* Foundation model fine-tuned on 300k+ EO chips for zero-shot segmentation.
  [https://github.com/whr946321/RemoteSAM](https://github.com/whr946321/RemoteSAM)

---

## 🧪 Datasets & Benchmarks

### 📊 ESA OpenSR Benchmark

* ESA’s official SR benchmarking and evaluation suite.
  [https://github.com/ESAOpenSR/opensr-test](https://github.com/ESAOpenSR/opensr-test)

---

### 🌍 WorldStrat Dataset

* Sentinel-2 & Pléiades aligned image pairs for SR training/testing.
  [https://github.com/satellite-image-deep-learning/datasets](https://github.com/satellite-image-deep-learning/datasets)

---

### 🗂️ NAIP-S2 Dataset

* For supervised SR (Sentinel-2 → NAIP high-res RGB).
* Dataset Paper:
  [https://arxiv.org/abs/2211.15660](https://arxiv.org/abs/2211.15660)

---

Would you like a version of this exported as PDF, or turned into a citation-ready BibTeX file?

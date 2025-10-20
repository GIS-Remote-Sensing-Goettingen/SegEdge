# Research Questions

This study investigates how different data representations, model architectures, and analytical strategies influence the segmentation of **linear woody features (LWFs)** from Sentinel-2 imagery.  
The questions are organized into four thematic groups that together define the logical structure of the research:  
(A) Data Representation and Enhancement,  
(B) Model Architectures and Feature Learning,  
(C) Temporal and Spatial Generalization, and  
(D) Analytical and Interpretive Methods.

---

## A. Data Representation and Enhancement
This study investigates how different data representations, model architectures, and analytical strategies influence the segmentation of **linear woody features (LWFs)** from Sentinel-2 imagery.  
**RQ1. Super-resolution for segmentation**  
*How does the application of satellite image super-resolution models influence the accuracy and reliability of segmenting linear woody features compared to segmentation on native Sentinel-2 imagery?*  

**RQ2. Comparative performance analysis**  
*What are the quantitative and qualitative performance differences between segmentation models trained on (a) super-resolved Sentinel-2 data, (b) native Sentinel-2 data, and (c) conventional image upsampling or enhancement techniques?*  

**RQ3. Super-resolution model ablation**  
*How do different super-resolution models (e.g., S2DR, ESRGAN, SwinIR) influence downstream segmentation performance, and what trade-offs arise in terms of computational efficiency, spectral fidelity, and spatial detail?*  

---

## B. Model Architectures and Feature Learning

**RQ4. Transformer-based segmentation**  
*To what extent do transformer-based architectures improve the detection and delineation of linear woody features compared to convolutional or hybrid models?*  

**RQ5. Backbone comparison**  
*What are the effects of different self-supervised backbones—such as DINOv3, DINOv2, and SatCLIP—on segmentation accuracy, feature quality, and generalization of linear woody features across heterogeneous landscapes?*  

**RQ6. Representation of LWFs in multidimensional feature spaces**  
*How can linear woody features be characterized and distinguished within high-dimensional feature spaces derived from self-supervised or pretrained backbones?*  

---

## C. Temporal and Spatial Generalization

**RQ7. Temporal vs single-date imagery**  
*Does integrating temporal sequences of Sentinel-2 imagery enhance the segmentation of linear woody features compared to models trained on single-date observations, and under what seasonal or phenological conditions?*  

**RQ8. Regional generalization and transferability**  
*How does segmentation model performance vary across ecological or biogeographical regions, and what factors (e.g., vegetation structure, climate, landscape configuration) drive differences in model generalization and transferability?*  

---

## D. Analytical and Interpretive Methods

**RQ9. Feature-space analysis**  
*What insights can principal component analysis (PCA) of intermediate feature maps provide about the separability, redundancy, and discriminative power of representations associated with linear woody features?*  


**RQ10. SAM2 vs feature-space segmentation methods**  
*How does segmentation of linear woody features using a foundation model such as SAM2 compare to unsupervised or semi-supervised methods based on PCA clustering or neighborhood-based segmentation, in terms of accuracy, robustness, and data dependency?*  

**RQ11. End-to-end segmentation pipeline**  
*How can an end-to-end processing pipeline—from super-resolution and feature extraction to segmentation and evaluation—be optimized for accurate and scalable mapping of linear woody features in heterogeneous landscapes?*  


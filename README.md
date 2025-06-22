# **RESTORING OCCLUDED OBJECTS: FROM DETECTION TO 3D RECONSTRUCTION**

## Table of Contents
1. [Introduction](#overview)
2. [Motivation & Applications](#motivation--applications)
3. [Model Components](#model-components)
4. [Dataset Preparation](#dataset-preparation)
5. [Architectural Details](#architectural-details)
6. [Workflow Pipeline](#workflow-pipeline)
7. [3D Reconstruction and Mesh Remeshing](#3d-reconstruction-and-mesh-remeshing)
8. [Evaluation & Key Takeaways](#evaluation--key-takeaways)

---

## Introduction

**"Restoring Occluded Objects: From Detection to 3D Reconstruction"** is a deep learning-based vision pipeline designed to detect, reconstruct, and reimagine occluded regions in RGB images. This multi-stage system integrates object detection, segmentation, inpainting, super-resolution, and 3D mesh generation into a unified framework.
At its core, the project leverages a convolutional autoencoder to predict and restore missing pixel regions, supported by a YOLO-based detection system and transformer-based segmentation model. High-fidelity detail is preserved and enhanced using Real-ESRGAN for super-resolution, while the final stage transforms 2D predictions into structured 3D mesh representations using Instant Meshes.
Implemented in PyTorch, this pipeline is built to handle challenging occlusion scenarios in real-world images—bridging the gap between 2D perception and 3D spatial understanding.

---

## Motivation & Applications
Imagine AR characters hiding behind real furniture, robots identifying partially hidden objects, or 3D sports analytics in occluded scenarios. This project enables machines to perceive occluded objects with human-like understanding.

### Applications:
- Surveillance and security in crowded scenes  
- Safer indoor navigation for robots  
- Autonomous driving under occlusion  
- Enhanced AR/VR immersion  
- Sports analytics for occluded athletes  
- Sorting occluded crops and clutter navigation  

---

## Model Components
1. **YOLOv11s** – Real-time object detection
2. **SAM (Segment Anything Model)** – Transformer-based segmentation
3. **Convolutional Autoencoder** – Pixel restoration/inpainting
4. **Real-ESRGAN** – Super-resolution enhancement
5. **Instant Meshes** – Mesh generation for 3D reconstruction

---

## Dataset Preparation
- Format: RGB images, YOLO-format labels, and visible/occluded masks
- Generated from COCO-style annotations
- Split into `train`, `val`, and `test`
- `dataset.yaml` for YOLO training compatibility

---

## Architectural Details

### 1. YOLOv11s (You Only Look Once)

**Explanation & Role:**  
YOLOv11s detects objects by generating bounding boxes. It’s fast and accurate—ideal for real-time inference. It initiates the pipeline by locating occluded object regions.

**Architecture:**
- Backbone: CNN for feature extraction
- Neck: Feature Pyramid Network (FPN)
- Head: Outputs boxes, confidence, class labels

**Activation Functions:**
- `SiLU`: Efficient for deep CNNs
- `Sigmoid`: For object confidence scaling

**Loss Functions:**
- `λ_box`: Bounding box regression
- `λ_obj`: Objectness confidence
- `λ_cls`: Classification accuracy

---

### 2. SAM (Segment Anything Model)

**Explanation & Role:**  
SAM segments the regions identified by YOLO using bounding box prompts. It provides pixel-level binary masks.

**Architecture:**
- ViT Encoder: Hierarchical feature extraction
- Prompt Encoder: Box or point prompts
- Mask Decoder: High-res binary masks

**Activation Functions:**
- `SiLU`: Used in transformer layers

**Loss Functions:**
- `Binary Cross Entropy (BCE)`
- `Dice Loss`
- `IoU Loss`

---

### 3. Convolutional Autoencoder

**Explanation & Role:**  
Fills in the occluded parts using learned features. It compresses, understands, and reconstructs the missing regions.

**Architecture:**
- Encoder: Convolutional layers with `ReLU`
- Bottleneck: Fully connected MLP
- Decoder: Transposed Conv layers

**Activation Functions:**
- `ReLU`: Non-linear feature learning
- `Sigmoid`: Pixel range normalization

**Loss Functions:**
- `Mean Squared Error (MSE)`: Best for reconstruction accuracy

---

### 4. Real-ESRGAN

**Explanation & Role:**  
Performs perceptual and pixel-level super-resolution before and after inpainting.

**Architecture:**
- Generator: Residual-in-Residual Dense Blocks
- Discriminator: GAN-based classifier

**Activation Functions:**
- `ReLU / Leaky ReLU`: Sharpens image features
- `Sigmoid`: Scales output pixels

**Loss Functions:**
- `Adversarial Loss`: Realism via discriminator
- `Perceptual Loss`: Feature-based accuracy
- `Pixel Loss (MSE/L1)`: Base fidelity retention

---

## Workflow Pipeline

1. **Object Detection** → YOLOv11s locates occluded objects  
2. **Segmentation** → SAM generates object masks  
3. **Super-Resolution (1)** → Real-ESRGAN enhances mask quality  
4. **Inpainting** → Autoencoder fills occlusions  
5. **Super-Resolution (2)** → Real-ESRGAN final enhancement  
6. **Optional** → Save masks, bounding boxes, all intermediates

---

## 3D Reconstruction and Mesh Remeshing

**Instant Meshes** transforms 2D output into quad-dominant 3D meshes:
- Uses **directional & position fields**
- **Quadrangulation** ensures smooth topology
- **Adaptive remeshing** adds detail where necessary

Ideal for simulation, animation, and AR applications.

---

## Evaluation & Key Takeaways

### Performance Metrics:
- `YOLO`: mAP (mean Average Precision)
- `Autoencoder`: PSNR, MSE
- `Real-ESRGAN`: SSIM, perceptual fidelity

### Key Insights:
- Autoencoder restores context effectively
- SAM produces high-quality masks with minimal input
- Real-ESRGAN enhances both semantic and pixel clarity
- Modular system design allows easy adaptation

---

## Tech Stack
- Python, PyTorch, OpenCV, PIL
- Ultralytics YOLO, Segment Anything, Real-ESRGAN
- Kaggle/Jupyter Environment

---

## Contributing
Pull requests and feedback are welcome. Let’s make machines see through occlusions together.

---

## License
This project is under the MIT License.

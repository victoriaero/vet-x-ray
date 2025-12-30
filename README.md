# Veterinary Thoracic X-ray Anomaly Detection

This project investigates the feasibility of **transferring knowledge from human chest X-ray models to veterinary thoracic radiographs**, with the goal of identifying **pulmonary opacities** in small animals (dogs and cats).  
Given the anatomical similarities between human and animal thoraxes, we explore whether models pretrained on large-scale human datasets can be adapted to the veterinary domain through **fine-tuning with a limited number of labeled animal exams**.

The project is a collaboration between the **Department of Computer Science** and the **School of Veterinary Medicine**, combining machine learning techniques with expert clinical annotation.

## Motivation

Pulmonary opacities are a common radiographic finding in veterinary medicine and may indicate conditions such as pneumonia, edema, neoplasia, or pleural effusion. However, annotated veterinary imaging datasets are scarce, expensive, and time-consuming to produce, which motivates the use of **transfer learning** and **pretrained medical vision models** originally developed for humans.

## Dataset Characteristics and Challenges

Veterinary thoracic radiography presents several challenges that differ from the human setting. Veterinary exams often include multiple views, such as:
  - Ventro-dorsal (VD)
  - Latero-lateral right (LLD)
  - Latero-lateral left (LLE)

Veterinary thoracic radiography also presents heterogeneity in exam composition and image quality. Not all patients share the same set of projections. Some exams include ventro-dorsal and latero-lateral views, others include all three standard projections, and in some cases images are acquired multiple times. In addition, variations in positioning, exposure, motion artifacts, and acquisition protocols strongly affect the visibility of pulmonary opacities. To reduce this variability and maintain closer similarity with human chest X-ray datasets, we initially opted to work with a single projection, focusing exclusively on ventro-dorsal (VD) images.

## Annotation Protocol

Approximately 200 veterinary thoracic exams were manually reviewed and labeled by domain experts. Images were originally classified into four categories: ausencia (no pulmonary opacity), pleura (opacity restricted to the pleural space), intrapulmonar (opacity within the lung parenchyma), and ambos (pulmonary opacity combined with other thoracic abnormalities). Although this labeling captures clinically relevant distinctions, the initial modeling objective was simplified to a binary task, evaluating whether a model can detect the presence of any pulmonary opacity regardless of its specific type.

```python
CLASS_MAP = {
    "ausencia": 0,
    "ambos": 1,
    "pleura": 1,
    "intrapulmonar": 1
}
```

## Project Structure

The repository is organized around three main components. The first component consists of report processing and annotation support tools, including PDF-to-text extraction, basic text analysis, and few-shot LLM-based classification used solely to assist veterinarians during the annotation process. These tools were not used as ground truth and served only as organizational aids.

The second component is a supervised learning pipeline for training and evaluating convolutional neural networks on veterinary radiographs. This pipeline includes patient-level dataset splitting to avoid data leakage, support for multiple backbone architectures such as ResNet, DenseNet, TorchXRayVision, and EVA-X, strategies to mitigate class imbalance, and standard evaluation metrics including accuracy, precision, recall, F1-score, ROC curves, and confusion matrices.

The third component focuses on evaluating pretrained and self-supervised models. In this setting, representations learned from large-scale human medical imaging datasets are reused, either with frozen encoders or through partial fine-tuning, and lightweight classification heads are trained on veterinary data to assess the transferability of these features.

## Current Status

The project is still in an active phase of experimentation. Although several pretrained models and fine-tuning strategies have been evaluated, the results obtained so far have not yet been satisfactory for robust pulmonary opacity detection in veterinary radiographs. Current findings suggest that limitations such as the restricted number of annotated exams, high variability in image quality, and domain differences between human and animal thoracic imaging play a significant role. At this stage, the project remains ongoing and awaits the acquisition and expert annotation of additional images to support more conclusive experiments.

# MultiDentNet: A Unified Deep Learning Framework for Automated Diagnosis of Multiple Dental Conditions and Oral Cancer

## Overview

Early and accurate diagnosis of multiple dental conditions and oral cancer remains a critical unmet need in clinical practice, where traditional visual and radiographic examinations are subjective and siloed by disease type. This study introduces MultiDentNet, a unified deep learning framework designed to automatically classify five common dental conditions caries, gingivitis, tooth discoloration, ulcers, and Hypodontia as well as oral cancer from intraoral images. To the best of our knowledge, this is the first comprehensive study to classify all five dental conditions simultaneously alongside oral cancer on these datasets; previous efforts have addressed at most one or two classes. To address the gap in multi-condition diagnostic tools, we augment a DenseNet-121 backbone with hierarchical self-attention modules that capture both local lesion details and global context. Evaluated on a curated dataset of 6,791 training, 1,701 validation, and 947 test images for dental conditions, and an additional dataset of 940 images for oral cancer (490 cancer and 450 non-cancer), MultiDentNet achieves 99.37\% overall accuracy, 99.37\% precision, 99.37\% recall, an F1-Score of 0.98, AUC of 1.00, and a log loss of 0.023 for dental conditions. For oral cancer, the model achieves 94.68\% accuracy, 97.62\% precision, 91.11\% recall, and an F1-Score of 0.94. The model delivers notably high accuracy for underrepresented categories, reaching an F1-score of 0.94 for Hypodontia and 0.95 for oral cancer, even without any fine-tuning. Interpretability analyses, including t-SNE feature clustering, one-vs-rest ROC and precision-recall curves, and confusion matrix visualizations, demonstrate distinct class separation and minimal misclassification, highlighting both robustness and clinical relevance. By integrating advanced attention mechanisms, class-aware loss functions, and rigorous interpretability tools, this work delivers a novel, scalable, and clinically applicable AI solution for comprehensive dental and oral cancer diagnostics, setting the stage for real-world decision support systems and future extensions to broader oral health conditions.




## Features

- **Comprehensive Classification:** Simultaneously detects five dental conditions and oral cancer in a unified model.
- **High Performance:** Achieves 99.37% accuracy on dental conditions and 94.68% on oral cancer datasets.
- **Interpretability:** Includes t-SNE visualizations, ROC/PR curves, and confusion matrices for transparent decision-making.
- **Robust Training:** Incorporates Focal Loss, mixed-precision training, and domain-specific augmentations for handling imbalanced data.

## Dataset

- **Dental Condition Dataset:** Contains 9,439 annotated intraoral images, split into 6,791 training, 1,701 validation, and 947 test samples. Available at [Kaggle](https://www.kaggle.com/datasets/salmansajid05/oral-diseases).
- **Oral Cancer Dataset:** Comprises 940 images (490 cancer, 450 non-cancer), split into 752 training, 94 validation, and 94 test samples. Available at [Kaggle](https://www.kaggle.com/datasets/shivan17299/oral-cancer-lips-and-tongue-images).
- **Augmentation:** Utilizes CLAHE, elastic deformations, and rotations to enhance diversity and generalization.

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/MultiDentNet.git
   cd MultiDentNet



torch==2.1.0

 torchvision==0.16.0  # Compatible with torch 2.1.0
 
 albumentations==1.3.1
 
 numpy==1.24.4
 
 matplotlib==3.7.4
 
 seaborn==0.13.0
 
 scikit-learn==1.3.2
 
 pillow==10.1.0

The datasets can be downloaded from this Kaggle address: https://www.kaggle.com/datasets/salmansajid05/oral-diseases, https://www.kaggle.com/datasets/zaidpy/new-oral-cancer/data.

![Example Image](ps0.png)
![Example Image](ps1.png)




## Summary of dental condition dataset

| **Category**          | **Description** |
|-----------------------|----------------|
| Caries               | Images showing tooth decay, cavities, or carious lesions. |
| Gingivitis           | Images displaying inflamed or infected gums. |
| Tooth discoloration  | Images showcasing tooth discoloration or staining. |
| Ulcers               | Images exhibiting oral ulcers or canker sores. |
| Hypodontia           | Images representing the condition of missing one or more teeth. |


![Example Image](Arch.png)

### Evaluation Summary  of dental conditions

| Metric          | Value   |
|-----------------|---------|
| Accuracy        | 0.9937  |
| Precision       | 0.9937  |
| Recall          | 0.9937  |
| F1-Score        | 0.9936  |
| Cohen's Kappa   | 0.9917  |
| MCC             | 0.9917  |
| Log Loss        | 0.0230  |



## Evaluation summary of oral cancer

| Metric          | Value   |
|-----------------|---------|
| Accuracy        | 0.9468  |
| Precision       | 0.9762  |
| Recall          | 0.9111  |
| F1-Score        | 0.9425  |
| Cohen's Kappa   | 0.8931  |
| MCC             | 0.8950  |
| Log Loss        | 0.2470  |


## Classification report for oral cancer

| **Category**   | **Precision** | **Recall** | **F1-Score** | **Support** |
|---------------|--------------|-----------|-------------|------------|
| Cancer        | 0.92         | 0.98      | 0.95        | 49         |
| Non-Cancer    | 0.98         | 0.91      | 0.94        | 45         |

### Classification report for dental conditions

| Category              | Precision | Recall | F1-Score | Support |
|-----------------------|-----------|--------|----------|---------|
| Caries                | 1.00      | 1.00   | 1.00     | 239     |
| Gingivitis            | 0.98      | 1.00   | 0.99     | 234     |
| Hypodontia            | 0.97      | 0.91   | 0.94     | 35      |
| Tooth Discoloration   | 1.00      | 0.99   | 0.99     | 184     |
| Ulcers                | 1.00      | 1.00   | 1.00     | 255     |


![Example Image](dental3.png)
![Example Image](output11.png)

![Example Image](Misclassified.png)
![Example Image](Confusion.png)
![Example Image](Confusion1.png)


![Example Image](LC.png)
![Example Image](LC1.png)
![Example Image](LC2.png)

![Example Image](Precision-Recall.png)
![Example Image](Precision-Recall1.png)

![Example Image](ROC.png)
![Example Image](ROC22.png)
![Example Image](tsne.png)
![Example Image](tsne1.png)

# Conclusions

In this work, we have presented MultiDentNet, a novel self-attention-enhanced DenseNet architecture designed for the simultaneous classification of five common dental conditions (caries, gingivitis, tooth discoloration, ulcers, and Hypodontia) as well as oral cancer from intraoral images. Our key findings include:

## Key Findings

- **High diagnostic performance:** On a held-out test set of 947 images for dental conditions, our model achieved 99.37% overall accuracy, 99.37% precision and recall, a macro-F1 score of 0.98, and a macro-AUC of 1.00. For oral cancer, the model achieved 94.68% accuracy, 97.62% precision, 91.11% recall, and an F1-score of 0.94 on a dataset of 131 images. These results demonstrate robust classification even for underrepresented classes, such as Hypodontia (F1 = 0.94) and oral cancer (F1 = 0.95 for Cancer class).

- **Effective imbalance handling:** By integrating Focal Loss and a domain-specific augmentation pipeline (including CLAHE, elastic deformations, and anisotropic rotations), we substantially mitigated the effects of severe class skew, ensuring strong performance across all conditions.

- **Enhanced interpretability:** Qualitative analyses such as t-SNE feature clustering, one-vs-rest ROC/PR curves, and confusion-matrix visualizations confirmed clear class separability and provided valuable insights into residual error modes, fostering transparency and clinician trust.

These results directly address our original objective of developing a unified, scalable diagnostic tool for comprehensive oral health assessment. By combining hierarchical self-attention with DenseNet's feature reuse, our framework effectively captures both fine-grained lesion details and global contextual cues, addressing limitations in prior studies that focused on single or dual conditions.

## Implications

Our approach offers several practical benefits for clinical practice:

- *Clinical decision support:* The high accuracy, confidence, and interpretability of MultiDentNet make it a promising candidate for real-time augmentation of dentist evaluations, potentially reducing diagnostic errors and enhancing patient outcomes.

- *Workflow efficiency:* An end-to-end system that concurrently screens for multiple dental conditions and oral cancer can streamline clinical workflows, reduce patient chair time, and minimize variability across clinicians.

- *Foundation for multimodal extension:* The self-attention modules and training strategies we propose can be readily adapted to incorporate additional imaging modalities, such as radiographs or 3D scans, further enhancing diagnostic capabilities.

## Limitations

While MultiDentNet demonstrates exceptional performance on the current datasets, several limitations warrant consideration:

- **Dataset diversity:** The dental condition images were sourced from a limited number of clinical centers, and the oral cancer dataset is relatively small (131 images). External validation on more diverse populations, imaging devices, and larger datasets is needed to confirm generalizability.

- **Two-dimensional images:** Intraoral photographs may not capture sub-surface pathology visible in radiographs. Future work should explore multimodal fusion to integrate radiographic data for a more comprehensive diagnostic tool.

- **Threshold optimization:** Although AUC values indicate excellent ranking capability, optimal decision thresholds for each class remain to be calibrated for specific clinical settings to maximize clinical utility.

## Future Work

Building on these findings, we plan to:

- Validate MultiDentNet on multi-institutional and mobile-device-captured image datasets to assess its robustness in tele-dentistry contexts and ensure applicability across diverse populations.

- Integrate radiographic inputs (e.g., bitewing X-rays, CBCT) via a multimodal architecture to improve the detection of sub-surface lesions and enhance diagnostic accuracy.

- Investigate active-learning strategies to reduce annotation burden and continually adapt the model to evolving clinical practices, ensuring long-term relevance and performance.

## Summary

MultiDentNet provides a comprehensive, interpretable, and high-performance solution for multi-condition dental and oral cancer diagnostics. By addressing the limitations of prior research and offering a scalable framework for clinical deployment, we believe this work will catalyze further advancements in AI-driven oral healthcare and accelerate the adoption of intelligent diagnostic tools in everyday practice.

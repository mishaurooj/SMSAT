# SMSAT Dataset and Models

This is the official repository for the **SMSAT (Spiritual Meditation, Music, Silence Acoustic Time Series)** dataset, trained models, code, and results, accompanying the paper:

> **"SMSAT: An Acoustic Dataset and Multi-Feature Deep Contrastive Learning Framework for Affective and Physiological Modeling of Spiritual Meditation"**  
> (IEEE Transactions on Affective Computing, 2025)  

📄 [Paper PDF]([./paper/SMSAT_Paper.pdf](https://arxiv.org/abs/2505.00839))  
📊 [Dataset on Kaggle](https://www.kaggle.com/datasets/crdkhan/qmsat-dataset)

---
## SMSAT Dataset

### Detailed flow graph of the proposed methodology
![Subject Distribution](./data/block.jpg)

### Data Collection and Acquisition Device
![Subject Distribution](./data/daq.png)


#### Time Domain 
![All in One](./data/all_classes_audio.jpeg)

#### Subject Distribution
![Subject Distribution](./data/subject_distribution.png)

#### Dataset  Distribution
![Distribution](./data/data-subplot.png)

---
## Dataset Validation

![Subject Distribution](./Dataset-Validation/signal_comparison_all_classes.png)

---
## PROPOSED SMSAT ATS ENCODER
![Subject Distribution](./SMSAT-Encoder-Results-Results/SMSAT_Encoder_page-0001.jpg)

### Dataset Augmentation

![Subject Distribution](./Dataset-Augmentation/all_in_one_figure.png)
---
## CALMNESS ANALYSIS MODEL (CAM)

### Architecture
![Subject Distribution](./CAM-model-Results/CAM_architecture.jpg)

### Spiritual Meditation
![Subject Distribution](./CAM-model-Results/activations_SpiritualMeditation.png)

### Normal Silence
![Subject Distribution](./CAM-model-Results/activations_NormalSilence.png)

### Music
![Subject Distribution](./CAM-model-Results/activations_Music.png)
---
## 🚀 Getting Started

## Dataset
The dataset is hosted on Kaggle: 👉 SMSAT Dataset on Kaggle https://www.kaggle.com/datasets/crdkhan/qmsat-dataset/data

---

## 📊 Results
SMSAT Encoder classification accuracy: 99.99%
CAM calmness analysis accuracy: 99.99%
Statistical analyses confirm that Spiritual Meditation ≈ Silence < Music in calmness effect.
Full ablation study, ANOVA, and visualizations are available in results/.

---

##
📄 Citation
If you use this dataset or models, please cite:
@article{SMSAT2025,
  title={SMSAT: An Acoustic Dataset and Multi-Feature Deep Contrastive Learning Framework for Affective and Physiological Modeling of Spiritual Meditation},
  author={Ahmad Suleman and Yazeed Alkhrijah and Misha Urooj Khan and Hareem Khan and Muhammad Abdullah Husnain Ali Faiz and Mohamad A. Alawad and Zeeshan Kaleem and Guan Gui},
  journal={IEEE Transactions on Affective Computing},
  year={2025}
}

---

##
📧 Contact
For questions, reach out: crdteamwork786@gmail.com

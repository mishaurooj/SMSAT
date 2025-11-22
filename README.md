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

# 🌟 WaveGAN Results

## 📊 Subject-wise Cross-Validation Performance

| Model / Classifier     | Accuracy (mean ± std) | F1-score (mean ± std) |
|------------------------|------------------------|-------------------------|
| **CAM (ours)**         | **98.4 ± 3.1**         | **97.9 ± 3.4**          |
| **SMSAT-Enc (ours)**   | 96.5 ± 3.8             | 96.2 ± 4.0              |
| wav2vec2.0 + Linear    | 81.7 ± 4.5             | 80.9 ± 4.2              |
| OpenL3 + SVM           | 78.5 ± 5.1             | 77.8 ± 5.3              |
| MFCC + SVM             | 69.3 ± 6.4             | 68.1 ± 6.9              |
| 1D CNN baseline        | 73.4 ± 5.7             | 72.6 ± 6.0              |

---

## 🧠 SHAP Explainability

### SHAP Beeswarm Plot  
`SMSAT/WaveGAN-Results/shap_summary_beeswarm.png`  
![SHAP Beeswarm](SMSAT/WaveGAN-Results/shap_summary_beeswarm.png)

### SHAP Bar Plot  
`SMSAT/WaveGAN-Results/shap_summary_bar.png`  
![SHAP Bar](SMSAT/WaveGAN-Results/shap_summary_bar.png)

---

## 🏗️ WaveGAN Architecture

| Generator | Discriminator |
|----------|---------------|
| ![Generator](SMSAT/WaveGAN-Results/Wavegan_generator.png) | ![Discriminator](SMSAT/WaveGAN-Results/Wavegan_discriminator.png) |

---

## 📉 Training Loss Curves

| NS | Music | SM |
|----|-------|-----|
| ![NS](SMSAT/WaveGAN-Results/NS_loss_curve.png) | ![M](SMSAT/WaveGAN-Results/M_loss_curve.png) | ![SM](SMSAT/WaveGAN-Results/SM_loss_curve.png) |

---

## 🎧 Classwise Generation Quality

| Music | Normal (Silence) | Spiritual Meditation |
|-------|------------------|----------------------|
| ![Music](SMSAT/WaveGAN-Results/quality_eval_music.png) | ![NS](SMSAT/WaveGAN-Results/quality_eval_Normal(Silence).png) | ![SM](SMSAT/WaveGAN-Results/quality_eval_SpiritualMeditation.png) |
---

## 🔎 Signal Comparison

`SMSAT/WaveGAN-Results/signal_comparison_all_classes.png`  
![Signal Comparison](SMSAT/WaveGAN-Results/signal_comparison_all_classes.png)

---

## 🧪 Example Generated Output

`SMSAT/WaveGAN-Results/gen1.png`  
![gen1](SMSAT/WaveGAN-Results/gen1.png)

---

## 📦 Generated Dataset Preview

`SMSAT/WaveGAN-Results/Generated-dataset.jpeg`  
![Dataset](SMSAT/WaveGAN-Results/Generated-dataset.jpeg)

---

## ✨ Summary

WaveGAN successfully produced class-consistent synthetic audio for **Music**, **Natural Silence**, and **Spiritual Meditation**, improving balance and supporting stable subject-wise evaluation.


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

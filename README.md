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

# 🌟 SMSAT WaveGAN Results

This repository presents the **WaveGAN-based augmentation results** for the **SMSAT dataset**, including training curves, generator/discriminator architectures, and quality evaluations for all three auditory classes.

## WaveGAN Architecture

| Generator | Discriminator |
|----------|---------------|
| ![Generator](Wavegan_generator.png) | ![Discriminator](Wavegan_discriminator.png) |

## Training Loss Curves

| NS Loss | Music Loss | SM Loss |
|--------|------------|---------|
| ![NS](NS_loss_curve.png) | ![M](M_loss_curve.png) | ![SM](SM_loss_curve.png) |

## Classwise Generation Samples

| Music | Normal (Silence) | Spiritual Meditation |
|-------|------------------|----------------------|
| ![Music](quality_eval_music.png) | ![NS](quality_eval_Normal(Silence).png) | ![SM](quality_eval_SpiritualMeditation.png) |

## Signal Comparison Across All Classes

![Signal Comparison](signal_comparison_all_classes.png)

## Example Generated Output

![Generated Sample](gen1.png)

## Generated Dataset Preview

![Dataset](Generated-dataset.jpeg)

## Summary

WaveGAN successfully produced class-consistent synthetic audio for **Music**, **Normal Silence**, and **Spiritual Meditation**, supporting balanced training and improved model stability.


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

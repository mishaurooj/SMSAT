# SMSAT Dataset and Models

This is the official repository for the **SMSAT (Spiritual Meditation, Music, Silence Acoustic Time Series)** dataset, trained models, code, and results, accompanying the paper:

> **"SMSAT: An Acoustic Dataset and Multi-Feature Deep Contrastive Learning Framework for Affective and Physiological Modeling of Spiritual Meditation"**  
> (IEEE Transactions on Affective Computing, 2025)  

📄 [Paper PDF](./paper/SMSAT_Paper.pdf)  
📊 [Dataset on Kaggle](https://www.kaggle.com/datasets/crdkhan/qmsat-dataset)

---

## 📂 Repository Structure
- `data/` – Raw ATS signals, dataset augmentation, and validation.  
- `models/` – Pretrained models (SMSAT Encoder, CAM).  
- `notebooks/` – Jupyter notebooks for reproducing experiments.  
- `results/` – Statistical tests, ablation studies, and performance results.  
- `code/` – Training & evaluation scripts with utilities.  
- `paper/` – Published paper (PDF).  

---

## 🚀 Getting Started
### Installation
```bash
git clone https://github.com/yourusername/SMSAT-dataset.git
cd SMSAT-dataset/code
pip install -r requirements.txt

---

## Dataset
The dataset is hosted on Kaggle: 👉 SMSAT Dataset on Kaggle https://www.kaggle.com/datasets/crdkhan/qmsat-dataset/data
---

## ⚡ Usage

Train the SMSAT Encoder:
python train_encoder.py --data ../data/ATS-data
Train the Calmness Analysis Model (CAM):
python train_cam.py --data ../data/ATS-data

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

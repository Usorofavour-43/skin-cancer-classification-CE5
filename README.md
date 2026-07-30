# Skin Lesion Classifier (Benign vs Malignant)

GET 324 - Laboratory Exercise 10 (Mini-Project)
Group: CE5

## Overview
A binary image classifier that distinguishes benign from malignant
dermatoscopic skin lesion images, deployed as a Streamlit web application.

Two architectures were trained and compared:
1. A custom CNN (three convolutional blocks, trained from scratch)
2. MobileNetV3Small via transfer learning (frozen feature extraction, then fine-tuned)

The best-performing model (MobileNetV3, frozen) was selected for deployment,
prioritizing the lowest false-negative rate over raw accuracy.

## Dataset
[Skin Cancer: Malignant vs Benign](https://www.kaggle.com/datasets/fanconic/skin-cancer-malignant-vs-benign) (Kaggle)

## Project Structure
- `app.py` - Streamlit application
- `train_model.py` - Training and evaluation pipeline
- `requirements.txt` - Python dependencies
- `models/` - Saved trained models

## How to Run Locally
1. Clone this repo
2. `pip install -r requirements.txt`
3. `streamlit run app.py`

## Team Members

| Name | Registration Number | GitHub Username |
|------|---------------------|------------------|
| Ukpuho Miracle Aniekan | 22/EG/CE/1402 | Mabu04 |
| Eze Agatha Oluebube | 22/EG/CE/1352 | Nik-ki25 |
| Okon Clement Emem | 22/EG/CE/1392 | Okonclement10 |
| Ebong Augustine Jerome | 22/EG/CE/1382 | Ebongaustine10 |
| George Felix Uduak | 22/EG/CE/1422 | fg9190293 |
| Ekemini Udoma Ekong | 22/EG/CE/1412 | blackstar2004 |
| Akaka Nsisong Victoria | 22/EG/CE/1372 | Naddiee |
| Usoro Favour Elijah | 22/EG/CE/1362 | Usorofavour-43 |

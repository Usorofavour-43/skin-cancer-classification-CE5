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

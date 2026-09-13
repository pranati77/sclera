import os
import zipfile
import requests
import cv2
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

DATASET_ZIP = "anemia_dataset.zip"
DATASET_DIR = "anemia_dataset"
GITHUB_RAW_URL = "https://github.com/turna1/Computer-Vision/raw/main/anemia%20dataset-20231113T154638Z-001.zip"

if not os.path.exists(DATASET_DIR):
    if not os.path.exists(DATASET_ZIP):
        print("Downloading dataset from GitHub...")
        r = requests.get(GITHUB_RAW_URL, stream=True)
        with open(DATASET_ZIP, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        print("Download complete.")
    
    print("Unzipping dataset...")
    with zipfile.ZipFile(DATASET_ZIP, 'r') as zip_ref:
        zip_ref.extractall(DATASET_DIR)

def extract_features(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    
    R = img_rgb[:, :, 0].astype(np.float32)
    G = img_rgb[:, :, 1].astype(np.float32)
    B = img_rgb[:, :, 2].astype(np.float32)
    
    eps = 1e-6
    ei = float(np.mean(np.log(R + eps) - np.log(G + eps)) * 100)
    a_star = float(np.mean(img_lab[:, :, 1]))
    nrr = float(np.mean(R / (R + G + B + eps)) * 100)
    return [ei, a_star, nrr]

X, y = [], []
print("Extracting features from dataset images...")

for root, _, files in os.walk(DATASET_DIR):
    for f in files:
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            full_path = os.path.join(root, f)
            lower_str = (root + "/" + f).lower()
            if "non" in lower_str or "normal" in lower_str or "_2_" in lower_str:
                label = 0
            elif "anemic" in lower_str or "anaemic" in lower_str or "_1_" in lower_str:
                label = 1
            else:
                continue
            
            img = cv2.imread(full_path)
            if img is not None:
                X.append(extract_features(img))
                y.append(label)

X, y = np.array(X), np.array(y)
print(f"Extracted {len(X)} samples ({np.sum(y == 1)} Anemic, {np.sum(y == 0)} Healthy).")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
clf = LogisticRegression(class_weight='balanced')
clf.fit(X_train, y_train)

preds = clf.predict(X_test)
print("\n--- Model Calibration Report ---")
print(classification_report(y_test, preds, target_names=["Healthy", "Anemic"]))

with open("anemia_classifier.pkl", "wb") as f:
    pickle.dump({"model": clf}, f)
print("Saved classifier to anemia_classifier.pkl successfully.")
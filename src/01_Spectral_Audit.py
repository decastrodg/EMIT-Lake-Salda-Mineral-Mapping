import pandas as pd
import numpy as np
import os
import re
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# --- CONFIGURATION (GITHUB READY) ---
# Calculates paths dynamically based on file location.
# Works on any machine (Windows, Mac, Linux).

# 1. Get the directory where this script is located (src/)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Go up one level to the Project Root
BASE_DIR = os.path.dirname(SCRIPT_DIR)

# 3. Define paths relative to the Root
SPECTRUM_DIR = os.path.join(BASE_DIR, 'data', 'spectra')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')

# Create figures folder if it doesn't exist
os.makedirs(FIGURES_DIR, exist_ok=True)

# Set random seed for exact reproducibility
np.random.seed(42)
SWIR_GRID = np.linspace(2100, 2450, 50)

class_mapping = {
   "Stromatolitic_Complex": ["ROI_Hydromagnesite_Shoreline_01.txt", "ROI_Hydromagnesite_Shoreline_02.txt", "magnesispcMagnesiteHydromagHS473B.txt"],
   "Ancient_Lacustrine_Sediment": ["ROI_Magnesite_Lacustrine_Sediment.txt"],
   "Transition_Soil": ["ROI_Mixed_Carbonate_Soil.txt", "dolomit2.txt"],
   "Lateritic_Soil": ["ROI_Lateritic_Oxide_Soil.txt", "chlorite6.txt"],
   "Ophiolitic_Basement": ["ROI_Serpentinite_Antigorite_01.txt", "ROI_Serpentinite_Antigorite_02.txt", "antigor6.txt", "antigorit3.txt", "antigorite4.txt"]
}

def read_envi_ascii(filepath):
   data = []
   if not os.path.exists(filepath):
       print(f"Warning: File not found: {filepath}")
       return None
   with open(filepath, 'r') as f:
       lines = f.readlines()
   start_data = False
   for line in lines:
       if re.match(r'^\s*[\d\.]+\s+[\d\.\-eE]+', line): start_data = True
       if start_data:
           try:
               parts = line.split()
               if len(parts) >= 2: data.append((float(parts[0]), float(parts[1])))
           except ValueError: continue
   df = pd.DataFrame(data, columns=['wl', 'refl'])
   if df.empty: return None
   if df['wl'].max() < 100: df['wl'] *= 1000
   return df.sort_values('wl')

# --- DATA LOADING & AUGMENTATION ---
print(f"Loading spectra from: {SPECTRUM_DIR}")

if not os.path.exists(SPECTRUM_DIR):
    raise FileNotFoundError(f"CRITICAL: The directory {SPECTRUM_DIR} does not exist. Please move your .txt files to 'data/spectra'.")

X_list, y_list = [], []
files_found = 0

for class_name, files in class_mapping.items():
   for f_name in files:
       full_path = os.path.join(SPECTRUM_DIR, f_name)
       df = read_envi_ascii(full_path)
       if df is not None:
           files_found += 1
           interp_refl = np.interp(SWIR_GRID, df['wl'], df['refl'])
           # Create 30 variations per spectrum (Noise Injection)
           for _ in range(30):
               noise = np.random.normal(0, 0.002, len(SWIR_GRID))
               X_list.append(interp_refl + noise)
               y_list.append(class_name)

if files_found == 0:
    raise ValueError("No spectrum files were loaded. Check if the files are actually inside 'data/spectra'.")

X = np.array(X_list)
le = LabelEncoder()
y_encoded = le.fit_transform(y_list)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# --- VALIDATION (SVM) ---
print("Training SVM Classifier...")
pca = PCA(n_components=5)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

svm = SVC(kernel='rbf', C=10)
svm.fit(X_train_pca, y_train)

# --- AUDIT (Gradient Boosting) ---
print("Auditing Feature Importance...")
boosting = GradientBoostingClassifier(n_estimators=100, random_state=42)
boosting.fit(X_train, y_train)

print("\n[SVM + PCA] Precision Report:")
print(classification_report(y_test, svm.predict(X_test_pca), target_names=le.classes_))

# --- PLOTTING ---
print("Generating Feature Importance Plot...")
plt.figure(figsize=(10, 5))
plt.bar(SWIR_GRID, boosting.feature_importances_, width=4, color='darkblue', alpha=0.8)

# Add highlight zones
plt.axvspan(2300, 2315, color='gray', alpha=0.1, label='Carbonate Absorption (~2.31 µm)')
plt.axvspan(2320, 2340, color='orange', alpha=0.1, label='Serpentine Absorption (~2.33 µm)')

plt.title("Diagnostic Wavelengths for Lake Salda Minerals (Model Audit)")
plt.xlabel("Wavelength (nm)")
plt.ylabel("Importance Score")
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.5)

# Save to figures folder
output_path = os.path.join(FIGURES_DIR, 'Figure5_Feature_Audit.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"SUCCESS: Plot saved to {output_path}")

plt.show()
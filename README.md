```markdown
# Lake Salda Carbonate-Serpentine Discrimination using EMIT SWIR Spectroscopy

Author: Douglas Bazo de Castro  
Date: January 2026  
Context: Technical Report for Scheller Lab (Stanford University) - Take Home Problem

## 📌 Executive Summary
This project tests whether NASA’s **EMIT** (Earth Surface Mineral Dust Source Investigation) imaging spectroscopy can reliably distinguish **Mg-carbonate shoreline deposits** from **serpentinized ultramafics** at Lake Salda (Turkey).

The analysis focuses on the spectrally congested **2.30–2.34 µm region**, where carbonate and serpentine absorption features overlap significantly. By utilizing a **non-linear Support Vector Machine (SVM)** on PCA-reduced data and auditing the model with **Gradient Boosting**, this workflow successfully maps the mineral gradients without relying on linear unmixing assumptions that fail in intimate mixture scenarios.

---

## 📊 Key Results

### 1. Final Mineralogical Map
The model successfully confined the **Stromatolitic Complex (Hydromagnesite)** to the shoreline interface, distinguishing it from the surrounding **Ophiolitic Basement (Serpentine)**.

![Mineral Map](figures/Figure4_Mineral_Map.png)
*Figure 1: Final mineralogical map derived from EMIT imagery (2.10–2.45 µm).*

### 2. Model Audit (Feature Importance)
To ensure the model was not overfitting to broadband albedo, a Gradient Boosting auditor confirmed that decision power is driven by the diagnostic absorption minima at **2.30 µm** (Carbonates) and **2.33 µm** (Serpentine).

![Feature Audit](figures/Figure5_Feature_Audit.png)
Figure 2: Feature importance scores confirming the physical validity of the classification model.

---

## 🚀 How to Run This Project

### Prerequisites
 Python 3.9+
 Libraries: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `xarray`, `netCDF4`

### Installation

1. Clone this repository:
   ```bash
   git clone [https://github.com/YOUR_USERNAME/EMIT-Lake-Salda-Analysis.git](https://github.com/YOUR_USERNAME/EMIT-Lake-Salda-Analysis.git)
   cd EMIT-Lake-Salda-Analysis

```

2. Install dependencies:
```bash
pip install -r requirements.txt

```



### Data Setup

Due to size constraints (2GB+), the EMIT hyperspectral scene is **not** included in this repository.

1. Download the Granule:
* Dataset: EMIT L2A Surface Reflectance
* Granule ID: `EMIT_L2A_RFL_001_20240623T082158_2417506_019.nc`
* Source: [NASA EarthData Search](https://search.earthdata.nasa.gov/) or [EMIT Data Portal](https://earth.jpl.nasa.gov/emit/data/data-portal/).


2. Place the file:
Move the `.nc` file into the `data/` folder of this project. The path should look like this:
`EMIT-Lake-Salda-Analysis/data/EMIT_L2A_RFL_001_20240623T082158_2417506_019.nc`

### Running the Analysis

* **Step 1: Spectral Audit & Model Training**
Runs the cross-validation and generates the Feature Importance plot.
```bash
python src/01_Spectral_Audit_SVM.py

```


* Step 2: Full Scene Mapping
Applies the trained model to the EMIT scene and generates the Mineral Map.
```bash
python src/02_Scene_Mapping.py

```



---

## 📄 Technical Report

### 1. Introduction

Imaging spectroscopy has become a workhorse for mineral mapping because diagnostic absorption features in the shortwave infrared (SWIR, ~2.0–2.5 µm) can be linked to specific molecular bonds. In ultramafic systems, distinguishing Mg-bearing carbonates from Mg-rich phyllosilicates (e.g., serpentine) is critical.

Lake Salda (SW Turkey) serves as a key test case because it combines ultramafic source lithologies with shoreline carbonate precipitates—a direct analog for Jezero Crater on Mars. The challenge lies in the 2.30–2.34 µm region, where band centers for hydromagnesite (~2.31 µm) and serpentine (~2.32 µm) differ by only ~10 nm. Standard methods like Spectral Angle Mapper (SAM) often fail here due to sensitivity to small spectral shifts and noise.

### 2. Methodology

My approach prioritizes robust classification over theoretical linear unmixing. Given that intimate mixtures at the shoreline involve non-linear scattering that violates simple additivity, I employed a **Support Vector Machine (SVM)** with a Radial Basis Function (RBF) kernel applied to a PCA-reduced feature space.

Workflow Highlights:

1. Continuum Removal: Applied to the 2.10–2.45 µm window to isolate absorption geometry.
2. Hybrid Library: Constructed from scene ROIs and USGS laboratory endmembers.
3. Data Augmentation: Gaussian noise injection (σ = 0.002) during training to force the model to learn stable features rather than overfitting noise.
4. Auditing: Used XGBoost to verify that the model relies on physical absorption bands (2.30/2.33 µm) rather than artifacts.

### 3. Results and Discussion

The resulting map (Figure 1) aligns the serpentine-related unit with ultramafic exposures (32.1% of area) and confines the Stromatolitic Complex to the shoreline (19.2%).

The key implication is not merely that a carbonate-like absorption exists, but that separability remains defensible when absorption centers are tightly spaced. By producing abundance gradients (transition soils) and validating feature importance, this study provides an **auditable separability test** for carbonate–serpentine discrimination under EMIT-like conditions.

### 4. Conclusion

I tested whether Mg-carbonate shoreline materials can be reliably separated from serpentinized ultramafics at Lake Salda. Using EMIT surface reflectance and a continuum-normalized, noise-augmented SVM workflow, I found that carbonate detections are spatially coherent and spectrally distinct. The inference remains credible because mixed-pixel conditions and small band-position shifts were explicitly stress-tested during model training.

---

For code inquiries, please open an issue in this repository.

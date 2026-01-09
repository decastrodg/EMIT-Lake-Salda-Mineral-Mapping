import xarray as xr
import numpy as np
import os
import re
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import matplotlib.font_manager as fm
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.patheffects as PathEffects

# --- 1. CONFIGURATION (GITHUB READY) ---

# 1. Get directories relative to this script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR) # Project Root

# 2. Define standard paths
SPECTRUM_DIR = os.path.join(BASE_DIR, 'data', 'spectra')
FIGURES_DIR = os.path.join(BASE_DIR, 'figures')
DATA_DIR = os.path.join(BASE_DIR, 'data')

# 3. Define the EMIT Scene Path
# NOTE: The large .nc file should be placed in 'data/', but ignored by git.
EMIT_FILENAME = 'EMIT_L2A_RFL_001_20240623T082158_2417506_019.nc'
RFL_FILE_PATH = os.path.join(DATA_DIR, EMIT_FILENAME)

# Create figures folder if needed
os.makedirs(FIGURES_DIR, exist_ok=True)

# Reproducibility
np.random.seed(42)

# Mineral Formulas
class_info = {
   "Stromatolitic_Complex": r"Hydromagnesite $Mg_5(CO_3)_4(OH)_2 \cdot 4H_2O$",
   "Ancient_Lacustrine_Sediment": r"Magnesite $MgCO_3$ (Dry Lakebed)",
   "Transition_Soil": r"Mixed Carbonates / Serpentine Soils",
   "Lateritic_Soil": r"Fe-Oxides (Hematite/Goethite)",
   "Ophiolitic_Basement": r"Serpentinite / Antigorite $(Mg,Fe)_3Si_2O_5(OH)_4$"
}

# --- 2. PHASE 1: INTEGRATED TRAINING ---
print("--- Phase 1: Training Model with Spectral Library ---")

def read_envi_ascii(filepath):
   data = []
   if not os.path.exists(filepath): 
       print(f"Warning: Spectrum file not found: {filepath}")
       return None
   with open(filepath, 'r') as f:
       for line in f:
           if re.match(r'^\s*[\d\.]+\s+[\d\.\-eE]+', line):
               parts = line.split()
               try:
                   data.append((float(parts[0]), float(parts[1])))
               except ValueError:
                   continue
   df = np.array(data)
   if df.size == 0: return None
   if df[:, 0].max() < 100: df[:, 0] *= 1000
   return df

SWIR_GRID = np.linspace(2100, 2450, 48)
X_list, y_list = [], []

training_map = {
   "Stromatolitic_Complex": ["ROI_Hydromagnesite_Shoreline_01.txt", "ROI_Hydromagnesite_Shoreline_02.txt", "magnesispcMagnesiteHydromagHS473B.txt"],
   "Ancient_Lacustrine_Sediment": ["ROI_Magnesite_Lacustrine_Sediment.txt"],
   "Transition_Soil": ["ROI_Mixed_Carbonate_Soil.txt", "dolomit2.txt"],
   "Lateritic_Soil": ["ROI_Lateritic_Oxide_Soil.txt", "chlorite6.txt"],
   "Ophiolitic_Basement": ["ROI_Serpentinite_Antigorite_01.txt", "ROI_Serpentinite_Antigorite_02.txt", "antigor6.txt", "antigorit3.txt", "antigorite4.txt"]
}

print(f"Loading spectra from {SPECTRUM_DIR}...")
if not os.path.exists(SPECTRUM_DIR):
    raise FileNotFoundError(f"CRITICAL: Directory not found: {SPECTRUM_DIR}")

for class_name, files in training_map.items():
   for f_name in files:
       spec = read_envi_ascii(os.path.join(SPECTRUM_DIR, f_name))
       if spec is not None:
           interp_refl = np.interp(SWIR_GRID, spec[:, 0], spec[:, 1])
           # Data Augmentation
           for _ in range(50):
               X_list.append(interp_refl + np.random.normal(0, 0.002, 48))
               y_list.append(class_name)

if len(X_list) == 0:
    raise ValueError("No spectra loaded. Check 'data/spectra' folder.")

le = LabelEncoder()
y_train = le.fit_transform(y_list)
pca = PCA(n_components=8)
X_pca = pca.fit_transform(np.array(X_list))

svm = SVC(kernel='rbf', C=10, gamma='scale')
svm.fit(X_pca, y_train)
print("Model trained successfully.")

# --- 3. PHASE 2: IMAGE PROCESSING ---
print(f"\n--- Phase 2: Loading EMIT Reflectance Scene ---")
print(f"Looking for file at: {RFL_FILE_PATH}")

if not os.path.exists(RFL_FILE_PATH):
    raise FileNotFoundError(f"CRITICAL: EMIT file not found at {RFL_FILE_PATH}. Please ensure the .nc file is in the 'data' folder.")

ds_meta = xr.open_dataset(RFL_FILE_PATH, group='sensor_band_parameters')
wavelengths = ds_meta['wavelengths'].values
ds_root = xr.open_dataset(RFL_FILE_PATH)
full_cube = ds_root['reflectance'].values

nir_idx = np.abs(wavelengths - 850).argmin()
is_water = (full_cube[:, :, nir_idx] < 0.05)
is_land = ~is_water

swir_idx = np.where((wavelengths >= 2100) & (wavelengths <= 2450))[0]
reflectance_swir = full_cube[:, :, swir_idx]

# --- 4. PHASE 3: CLASSIFICATION ---
print(f"Classifying {np.sum(is_land)} land pixels...")
rows, cols, bands = reflectance_swir.shape
final_map = np.full((rows, cols), -1, dtype=float)

land_pixels = reflectance_swir[is_land]
land_pca = pca.transform(land_pixels)
final_map[is_land] = svm.predict(land_pca)

# Statistics
pixel_area_km2 = (60 * 60) / 1_000_000
unique, counts = np.unique(final_map, return_counts=True)
stats = dict(zip(unique, counts))

print(f"\n{'Geological Unit':<35} | {'Area (km²)':<10} | {'%'}")
print("-" * 60)
for i, class_label in enumerate(['Water/Mask'] + list(le.classes_)):
   count = stats.get(i-1, 0)
   print(f"{class_label:<35} | {count*pixel_area_km2:<10.2f} | {(count/final_map.size)*100:.1f}%")

# --- 5. PHASE 4: VISUALIZATION ---
print("\n--- Phase 4: Generating Publication-Ready Map ---")
colors = ['royalblue', 'lightgray', 'red', 'purple', 'darkgreen', 'peru']
cmap_geo = ListedColormap(colors)

fig, ax = plt.subplots(figsize=(12, 12))

im = ax.imshow(final_map, cmap=cmap_geo, vmin=-1, vmax=len(le.classes_)-1)
ax.set_title("Mineralogical Mapping of Lake Salda, Turkey (EMIT Hyperspectral)", fontsize=16, pad=10)
ax.axis('off')

# DEFINE STROKE
stroke = [PathEffects.withStroke(linewidth=2, foreground="black")]

# --- 1. SCALE BAR ---
pixel_size_meters = 60
target_scale_km = 5
target_scale_meters = target_scale_km * 1000
pixels_for_scalebar = target_scale_meters / pixel_size_meters
fontprops = fm.FontProperties(size=12, weight='bold')

scalebar = AnchoredSizeBar(ax.transData,
                          pixels_for_scalebar,
                          f'{target_scale_km} km',
                          loc=3,
                          pad=1,
                          color='white',
                          frameon=False,
                          size_vertical=5,
                          fontproperties=fontprops)
ax.add_artist(scalebar)
scalebar.txt_label.set_path_effects(stroke)

# --- 2. NORTH ARROW ---
arrow_x, arrow_y = 0.05, 0.85

arrow = ax.annotate('', xy=(arrow_x, arrow_y + 0.08), xytext=(arrow_x, arrow_y),
           xycoords='axes fraction',
           arrowprops=dict(facecolor='white', edgecolor='white', width=4, headwidth=12))

arrow.arrow_patch.set_path_effects(stroke)

n_text = ax.text(arrow_x, arrow_y + 0.10, 'N', transform=ax.transAxes,
       ha='center', va='bottom', color='white', fontsize=16, fontweight='bold')
n_text.set_path_effects(stroke)

# --- 3. LEGEND ---
legend_elements = [Patch(facecolor='royalblue', label='Water/Deep Shadow (NIR Masked)')]
for i, class_name in enumerate(le.classes_):
   formula = class_info.get(class_name, "")
   clean_label = f"{class_name.replace('_',' ')}\n{formula}"
   legend_elements.append(Patch(facecolor=colors[i+1], label=clean_label))

ax.legend(handles=legend_elements,
         loc='upper center',
         bbox_to_anchor=(0.5, -0.02),
         title="Geological Legend & Unmixing Interpretation",
         fontsize=10, title_fontsize=12, ncol=3, frameon=False)

plt.tight_layout()

# SAVE TO FIGURES FOLDER
output_path = os.path.join(FIGURES_DIR, 'Figure4_Mineral_Map.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nSUCCESS: Map saved to: {output_path}")

plt.show()
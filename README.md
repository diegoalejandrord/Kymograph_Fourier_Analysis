# Circular Kymograph Extraction Pipeline AND Fourier Rings Velocity Detection

## Circular Kymograph Extraction Pipeline
This repository contains a Python pipeline for detecting ring-like structures in microscopy images and extracting circular kymographs from dynamic image stacks. The workflow combines morphological preprocessing, contour detection, and trajectory-based sampling to generate intensity maps along circular trajectories.
## 📌 Features
Circular trajectory generation: Produces clockwise pixel coordinates around detected rings.
Morphological preprocessing: Thresholding, hole filling, erosion, and connected component labeling.
Region filtering: Selects regions based on circularity and area thresholds.
Contour detection: Identifies ring boundaries using OpenCV.
Kymograph extraction: Samples pixel intensities along multiple radii across all frames in a stack.
Visualization: Saves labeled region maps with contours and indices for validation.
## 📂 Input Data
Static image (AVG_SS7_2_EC_mean2.tif): Used for thresholding and ring detection.
Dynamic stack (SS7_2_EC_mean2.tif): Multi-frame image stack from which kymographs are extracted.
Both files should be placed in the working directory or paths updated in the script.
## 🚀 Usage
Run the script directly:
bash
python circular_kymograph.py
Workflow
Preprocessing
Threshold static image to isolate rings.
Fill holes and erode edges.
Label connected components.
Region Filtering
Compute circularity
Keep regions with 0.7 < circ < 1.0 and area > 250.
Contour Detection
Extract contours of filtered regions.
Compute enclosing circles.
Kymograph Extraction
Generate trajectories at three radii (r-3, r-1, r+1).
Sample intensities across all frames.
Combine trajectories into a single kymograph.
## Output
output_<i>.tif: Kymograph for each detected ring.
region_map.tif: RGB map with contours and labels.
## 📊 Example Output
Kymographs: Intensity vs. angle vs. time plots for each ring.
Region Map: Green contours with numeric labels overlaid on detected rings.
## 🛠️ Customization
Thresholds: Adjust values in
python
mask = ((image > 110) & (image < 180)).astype(int)
Circularity/area filters: Modify conditions in the region loop.
Output directory: Replace direct saves with os.makedirs("results", exist_ok=True) for organized outputs.
## 📖 Notes
Ensure input images are grayscale TIFFs.
The pipeline assumes (row, col) indexing for trajectories.
Vectorized extraction (stack[:, loc[:,0], loc[:,1]]) can speed up kymograph generation.

# Fourier Rings Velocity Detection

This repository contains a Python pipeline for detecting velocities from Fourier ring patterns in TIFF microscopy images.  
The workflow combines image preprocessing, FFT-based frequency analysis, and phase-slope regression to estimate nanometer-scale velocities across vertical slices of the input images.

## 📂 Project Structure

- **main.py** → Core script (the one you shared)
- **/data** → Directory containing TIFF images (user-defined)
- **/results** → Optional output folder for CSVs and plots

## ⚙️ Requirements

pip install numpy scipy scikit-learn opencv-python tifffile matplotlib seaborn

🔧 Parameters
Global parameters are defined at the top of the script:
Parameter	Description	Default
IMAGE_DIR	Path to directory containing TIFF images	/Users/.../Fourier_Rings
FILENAME_ROOT	Prefix for target image files	output_
PIXEL_SIZE	Pixel size in nanometers	43
Fs	Sampling frequency (frames per second)	1/3
SLICE_STEP	Step size for vertical slicing	10
SLICE_HEIGHT	Height of each vertical slice	50
SLICE_START_MAX	Max vertical start index	150
SHOW_PLOTS	Toggle for visualization	False
## 🖼️ Workflow Overview
Preprocessing
Normalize to 8-bit grayscale
Apply Savitzky–Golay smoothing
Enhance contrast with CLAHE
FFT & Frequency Profile
Compute FFT along slice columns
Extract magnitude spectrum
Smooth positive half of spectrum
Velocity Detection
Identify dominant frequency peak
Unwrap phase and detect downward slope
Fit linear regression to slope
Convert slope → velocity (nm/s)
Reject poor fits (low R², unrealistic velocities)
## Visualization (optional)
FFT magnitude profile
Phase slope regression fit
Velocity distribution histogram
## ▶️ Usage
Run the script:
bash
python main.py
Console output will show per-slice velocities and summary statistics. If SHOW_PLOTS=True, diagnostic plots will be displayed.
## 📊 Outputs
Console summary: Mean velocity per image and per slice
Velocity distribution plot: Histogram with KDE overlay

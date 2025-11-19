#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 13:43:52 2025

@author: diegoramirez
"""

# === Import required libraries ===
import os
import tifffile
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from sklearn.linear_model import LinearRegression
from numpy import unwrap
import seaborn as sns

# === Global parameters ===
IMAGE_DIR = '/Users/diegoramirez/Documents/python_projects/Fourier_Rings'  # Directory containing TIFF images
FILENAME_ROOT = 'output_'  # Prefix for target image files
PIXEL_SIZE = 43  # Pixel size in nanometers
Fs = 1 / 3  # Sampling frequency (frames per second)
SLICE_STEP = 10  # Step size for vertical slicing
SLICE_HEIGHT = 50  # Height of each vertical slice
SLICE_START_MAX = 150  # Max vertical start index
SHOW_PLOTS = False  # Toggle for visualization

# === Image preprocessing ===
def preprocess_image(image_path):
    # Load TIFF image
    img = tifffile.imread(image_path)
    
    # Normalize to 8-bit grayscale
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Apply Savitzky-Golay smoothing along both axes
    smt = savgol_filter(img, window_length=3, polyorder=1, axis=1)
    img2 = savgol_filter(smt, window_length=3, polyorder=1, axis=0)
    
    # Apply CLAHE for contrast enhancement
    clahe = cv2.createCLAHE(clipLimit=8, tileGridSize=(15,15))
    return clahe.apply(img2.astype(np.uint8))

# === FFT and frequency profile extraction ===
def extract_fft_phase(img_slice):
    L = img_slice.shape[0]  # Slice height
    Y = np.fft.fft(img_slice, axis=0)  # FFT along columns
    P2 = np.mean(np.abs(Y) / L, axis=1)  # Mean magnitude spectrum
    P2 = np.fft.fftshift(P2)  # Center zero frequency
    P1 = savgol_filter(P2[L//2:-1], window_length=5, polyorder=2)  # Smooth positive half
    f = Fs * np.arange(0, L // 2) / L  # Frequency axis
    return Y, P1, f

# === Velocity detection from phase slope ===
def detect_velocity(Y, P1, f):
    # Find dominant frequency peak
    peaks, _ = find_peaks(P1)
    if len(peaks) == 0 or P1[peaks[0]] < 5.5 * np.mean(P1[-5:-1]):
        return None

    peak_idx = peaks[0]  # Index of dominant frequency

    # Extract and unwrap phase at dominant frequency
    phase_deg = np.angle(Y) * 180 / np.pi
    phase = unwrap(phase_deg[peak_idx, :] * np.pi / 180) * 180 / np.pi
    phase = savgol_filter(phase, window_length=10, polyorder=1)
    x = np.linspace(0, len(phase), len(phase))  # X-axis for regression

    # Detect first upward peak to segment downward slope
    peaks_phase, _ = find_peaks(phase)
    if len(peaks_phase) == 0:
        return None
    
    

    peak_idx2 = peaks_phase[0]
    x_down = x[peak_idx2:]
    y_down = phase[peak_idx2:]
    
    if len(x_down) <45:
        return None

    # Fit linear model to downward segment
    model = LinearRegression().fit(x_down.reshape(-1, 1), y_down)
    slope = model.coef_[0]
    r_squared = model.score(x_down.reshape(-1, 1), y_down)

    # Reject poor fits
    if r_squared < 0.95:
        return None

    # Convert phase slope to velocity
    period = 1 / f[peak_idx]
    time_per_cycle = period * slope / 360
    velocity = PIXEL_SIZE / time_per_cycle
    
    if abs(velocity)>65:
        return None
        
    return velocity, slope, r_squared, x, phase, x_down, y_down, model

# === Optional visualization ===
def plot_results(P1, x, phase, x_down, y_down, model):

    plt.figure()
    plt.plot(P1)
    plt.title("FFT Magnitude Profile")
    plt.xlabel("Frequency Index")
    plt.ylabel("Magnitude")
    plt.show()

    plt.figure()
    plt.scatter(x, phase, label='Full data', color='gray')
    plt.scatter(x_down, y_down, label='Downward segment', color='blue')
    plt.plot(x_down, model.predict(x_down.reshape(-1, 1)), color='red', label='Linear fit')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Automatic Downward Segment Detection')
    plt.legend()
    plt.show()

# === Main loop over image files ===
results = {}
vel_total = []
for filename in os.listdir(IMAGE_DIR):
    # Filter for relevant TIFF files
    if not filename.startswith(FILENAME_ROOT) or not filename.endswith('.tif'):
        continue

    image_path = os.path.join(IMAGE_DIR, filename)
    print(f"\nProcessing {filename}...")
    img_clahe = preprocess_image(image_path)
    velocities = []


    # Iterate over vertical slices
    for i in range(0, SLICE_START_MAX, SLICE_STEP):
        img_slice = img_clahe[i:i+SLICE_HEIGHT, :]
        Y, P1, f = extract_fft_phase(img_slice)
        result = detect_velocity(Y, P1, f)

        if result is None:
            print(f"Slice {i}: No valid velocity detected.")
            continue

        velocity, slope, r_squared, x, phase, x_down, y_down, model = result
        velocities.append(velocity)
        vel_total.append(abs(velocity))
        print(f"Slice {i}: Velocity = {velocity:.2f} nm/s, R² = {r_squared:.3f}")
        plot_results(P1, x, phase, x_down, y_down, model)

    results[filename] = velocities

# === Summary output ===
print("\n=== Velocity Summary ===")
for fname, vels in results.items():
    if vels:
        print(f"{fname}: Mean velocity = {np.mean(vels):.2f} nm/s over {len(vels)} slices")
    else:
        print(f"{fname}: No valid velocities detected.")
        

# === Velocity Distribution Plot ===
if vel_total:
    plt.figure(figsize=(9, 6))
    bin_edges = np.arange(15, 70, 8)
    sns.histplot(vel_total, bins=bin_edges, kde=True, color='skyblue', edgecolor='black')
    plt.title('Distribution of Detected Velocities')
    plt.xlabel('Velocity (nm/s)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
else:
    print("No valid velocities found across all images.")
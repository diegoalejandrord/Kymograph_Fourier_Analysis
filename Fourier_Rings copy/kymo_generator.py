#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 13 14:52:40 2025

@author: diegoramirez
"""

import numpy as np
from skimage.io import imread, imsave
from skimage.morphology import disk, erosion
from skimage.measure import label, regionprops
from scipy import ndimage
import cv2
import skimage.transform

# Generate a clockwise circular trajectory around a center point
def circular_trajectory(center, radius, num_points=360):
    cx, cy = center
    angles = 2 * np.pi - np.linspace(0, 2 * np.pi, num_points, endpoint=False)  # clockwise
    x = cx + radius * np.cos(angles)
    y = cy + radius * np.sin(angles)

    # Round coordinates and remove duplicates while preserving order
    seen = set()
    coords = []
    for xi, yi in zip(x, y):
        pt = (int(round(xi)), int(round(yi)))
        if pt not in seen:
            seen.add(pt)
            coords.append(pt)

    return np.array(coords)

# Load static image and dynamic stack
image = imread('/Users/diegoramirez/Documents/python_projects/Fourier_Rings/AVG_SS7_2_EC_mean2.tif')
stack = imread('/Users/diegoramirez/Documents/python_projects/Fourier_Rings/SS7_2_EC_mean2.tif')

# Threshold to isolate ring-like structures (adjust thresholds as needed)
mask = ((image > 110) & (image < 180)).astype(int)

# Fill internal holes in binary mask
mask = ndimage.binary_fill_holes(mask).astype(int)

# Apply morphological erosion to clean up edges
mask = erosion(mask, disk(2))

# Convert mask to 8-bit format for contour detection
mask = mask * 255

# Label connected components in the mask
labels = label(mask, background=0)
regions = regionprops(labels)

# Initialize blank canvases
rows, cols = mask.shape
test = np.zeros((rows, cols), dtype=np.uint8)
i = 1

# Filter regions based on circularity and area
for region in regions:
    circ = 4 * np.pi * region.area / region.perimeter**2
    im_test = (labels == region.label).astype(np.uint8)

    if (0.7 < circ < 1.0) and (region.area > 250):
        test += im_test

    i += 1

# Detect contours from filtered mask
contours, _ = cv2.findContours(test, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Prepare output canvas and kymograph size
output = np.zeros_like(test)
new_size = (200, 75)
i = 0

# Initialize labeled region map
region_map = np.zeros_like(test, dtype=np.uint8)
region_map_color = np.stack([region_map]*3, axis=-1)  # RGB image for visualization

font = cv2.FONT_HERSHEY_SIMPLEX
font_scale = 0.5
thickness = 1
color = (255, 255, 255)  # white text


# Loop through each contour to extract circular kymographs
for cnt in contours:
    (x, y), radius = cv2.minEnclosingCircle(cnt)
    center = (int(y), int(x))  # Note: (row, col) format

    # Generate circular trajectories at three radii
    loc = circular_trajectory(center, radius - 3)
    loc1 = circular_trajectory(center, radius - 1)
    loc2 = circular_trajectory(center, radius + 1)

    # Skip if any coordinate exceeds image bounds
    if np.any(loc[:, 0] >= image.shape[0]) or np.any(loc[:, 1] >= image.shape[0]):
        continue

    # Extract pixel intensities along each trajectory for all frames
    kymo1 = np.array([stack[k][loc[:, 0], loc[:, 1]] for k in range(stack.shape[0])])
    kymo2 = np.array([stack[k][loc1[:, 0], loc1[:, 1]] for k in range(stack.shape[0])])
    kymo3 = np.array([stack[k][loc2[:, 0], loc2[:, 1]] for k in range(stack.shape[0])])

    # Resize kymographs to match shape (optional, currently redundant)
    kymo2 = skimage.transform.resize(kymo2, kymo1.shape)
    kymo3 = skimage.transform.resize(kymo3, kymo1.shape)

    # Combine kymographs (simple sum)
    kymo = kymo1 + kymo2 + kymo3

    # Save combined kymograph to disk
    name_file = f'output_{i}.tif'
    imsave(name_file, kymo.astype(np.uint8))
    
    # Draw contour and label on region map
    cv2.drawContours(region_map_color, [cnt], -1, (0, 255, 0), 1)  # green contour
    cv2.putText(region_map_color, str(i), (center[1], center[0]), font, font_scale, color, thickness)

    i += 1
    
# Save labeled region map
imsave('region_map.tif', region_map_color)
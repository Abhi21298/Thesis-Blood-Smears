import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.stats import gaussian_kde
import os

def count_cells(mask_image_path):
    # Read the mask image
    mask_img = cv2.imread(mask_image_path)
    
    # Define the color values for red, white, and black in BGR format
    red_color_lb = np.array([0, 0, 128])
    red_color_ub = np.array([0, 0, 256])
    white_color_lb = np.array([0, 128, 0])
    white_color_ub = np.array([0, 256, 0])
    
    # Create masks for red and white colors
    red_mask = cv2.inRange(mask_img, red_color_lb, red_color_ub)
    white_mask = cv2.inRange(mask_img, white_color_lb, white_color_ub)
    
    # Find connected components in the red mask
    num_labels_red, labels_red, stats_red, centroids_red = cv2.connectedComponentsWithStats(red_mask, connectivity=8)
    
    # Subtract 1 from the count to ignore the background
    num_red = 0
    
    # Areas of all RBCs detected
    areas_red = stats_red[1:, cv2.CC_STAT_AREA]
    
    # Plot the histogram
    plt.figure(figsize=(12, 6))
    hist, bins, _ = plt.hist(areas_red, bins=30, color='red', edgecolor='black', alpha=0.6, density=True)

    # Generate the KDE curve
    kde = gaussian_kde(areas_red)
    x = np.linspace(areas_red.min(), areas_red.max(), 1000)
    kde_curve = kde(x)
    
    # Plot the KDE curve
    plt.plot(x, kde_curve, '-k', label='Gaussian KDE')

    # Find peaks in the KDE curve
    peaks, _ = find_peaks(kde_curve)

    thresholds = []
    for i in range(len(peaks) - 1):
        if i >= 3:
            break
        threshold = (x[peaks[i]] + x[peaks[i + 1]]) / 2
        thresholds.append(threshold)
        plt.axvline(x=threshold, color='blue', linestyle='--', linewidth=2)

    plt.title('Histogram of RBC Component Areas with KDE Curve')
    plt.xlabel('Area')
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(os.path.join(os.path.dirname(mask_image_path), "RBC_area_histogram_with_kde.png"))
 
    for area in areas_red:
        count = 1
        for threshold in thresholds:
            if area > threshold:
                count += 1
            else:
                break
        num_red += count

    # Count white cells similarly
    num_labels_white, labels_white, stats_white, centroids_white = cv2.connectedComponentsWithStats(white_mask, connectivity=8)
    num_white = num_labels_white - 1

    return (num_red, num_white)

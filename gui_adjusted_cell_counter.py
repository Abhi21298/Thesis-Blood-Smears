import cv2
import numpy as np
import matplotlib.pyplot as plt
#from sklearn.cluster import KMeans
from scipy.signal import find_peaks
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
    # num_red = num_labels_red - 1
    
    # areas of all rbcs detected
    areas_red = stats_red[1:, cv2.CC_STAT_AREA]
    
    # Find connected components in the white mask
    num_labels_white, labels_white, stats_white, centroids_white = cv2.connectedComponentsWithStats(white_mask, connectivity=8)
    # Subtract 1 from the count to ignore the background
    num_white = num_labels_white - 1
    print("Cell counting done")

    areas_red = areas_red.reshape(-1, 1)
    #kmeans = KMeans(n_clusters=4, random_state=0).fit(areas_red)
    #cluster_centers = kmeans.cluster_centers_

    # Sort cluster centers to determine cutoffs
    #sorted_centers = np.sort(cluster_centers.flatten())

    plt.figure(figsize=(12, 6))
    plt.hist(areas_red, bins=30, color='red', edgecolor='black')
    plt.title('Histogram of RBC Component Areas')
    plt.xlabel('Area')
    plt.ylabel('Frequency')
    
    # Detect peaks in the histogram
    hist, bin_edges = np.histogram(areas_red, bins=30)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    peaks, _ = find_peaks(hist)
    
    first_peak = bin_centers[peaks[0]]
    plt.axvline(x=first_peak*1.5, color='blue', linestyle='--', linewidth=2)
    plt.axvline(x=first_peak*2.5, color='blue', linestyle='--', linewidth=2)
    plt.axvline(x=first_peak*3.5, color='blue', linestyle='--', linewidth=2)
    #thresholds = [(sorted_centers[i] + sorted_centers[i+1]) / 2 for i in range(len(sorted_centers) - 1)]

    # Mark the thresholds with vertical lines on the histogram
    #for threshold in thresholds[1:]:
    #    plt.axvline(x=threshold, color='blue', linestyle='--', linewidth=2)
    
    # Save the histogram plot
    plt.savefig(os.path.join(os.path.dirname(mask_image_path), "RBC_area_histogram.png"))
    
    #for area in areas_red:
    #    if area > thresholds[2]:
    #        num_red += 3
    #    elif area > thresholds[1]:
    #        num_red += 2
    #    else:
    #        num_red += 1
    
    for area in areas_red:
        if area > (first_peak * 3.5):
            num_red += 4
        elif area > (first_peak * 2.5):
            num_red += 3
        elif area > (first_peak * 1.5):
            num_red += 2
        else:
            num_red += 1

    return (num_red, num_white)

#if __name__ == "__main__":
#    rbc, wbc = count_cells(r"/home/amr1/Documents/gui_predict_IMG00360_2024-07-31T14:28:27/instance_segmented_mask.png")
#    print(rbc, wbc)

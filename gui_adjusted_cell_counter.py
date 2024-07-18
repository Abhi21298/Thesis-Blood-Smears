import cv2
import numpy as np

def count_cells(mask_image_path):
    # Read the mask image
    mask_img = cv2.imread(mask_image_path)
    
    # Define the color values for red, white, and black in BGR format
    red_color = np.array([0, 0, 128])
    white_color = np.array([255, 255, 255])
    
    # Create masks for red and white colors
    red_mask = cv2.inRange(mask_img, red_color, red_color)
    white_mask = cv2.inRange(mask_img, white_color, white_color)
    
    # Find connected components in the red mask
    num_labels_red, labels_red, stats_red, centroids_red = cv2.connectedComponentsWithStats(red_mask, connectivity=8)
    # Subtract 1 from the count to ignore the background
    num_red = num_labels_red - 1
    
    # Find connected components in the white mask
    num_labels_white, labels_white, stats_white, centroids_white = cv2.connectedComponentsWithStats(white_mask, connectivity=8)
    # Subtract 1 from the count to ignore the background
    num_white = num_labels_white - 1
    print("Cell counting done")
    return (num_red, num_white)

import numpy as np
import matplotlib.pyplot as plt
import cv2, os
from gui_masks_generator import masks
import tensorflow as tf

def adjust_contrast(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final

def Canny(image):

    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_img, (3,3), 3)
    edges = cv2.Canny(blurred, 80, 220, apertureSize=3)
    return edges

def split_connected_components(binary_mask):
    # Ensure output directory exists
    #os.makedirs(output_dir, exist_ok=True)
    input_dir = r'D:\UCC\Thesis\segment-anything-main\grid'
    # Label connected components
    num_labels, labels = cv2.connectedComponents(binary_mask)

    # Create a list to store bounding box coordinates
    bounding_boxes = []

    # Iterate through each label and create a separate mask
    for label in range(1, num_labels):  # Start from 1 to skip the background
        # Create a mask for the current label
        component_mask = (labels == label).astype(np.uint8) * 255

        # Find contours of the current component mask
        contours, _ = cv2.findContours(component_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            # Get the bounding box for the largest contour
            x, y, w, h = cv2.boundingRect(contours[0])
            bounding_boxes.append((x, y, w, h))

            # Save the mask
            mask_filename = os.path.join(input_dir, f'component_{label}.png')
            cv2.imwrite(mask_filename, component_mask)

            # Optionally display the mask
            cv2.imshow(f'Component {label}', component_mask)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

    return bounding_boxes




if __name__ == "__main__":
    # image = cv2.imread(r'D:\UCC\Thesis\segment-anything-main\0_0.png')

    # kernel = np.array([[0, -1, 0],
    #                 [-1, 5, -1],
    #                 [0, -1, 0]])

    # convolved_image = cv2.filter2D(image, -1, kernel)



    # #contrast_image = adjust_contrast(convolved_image)
    # cv2.imwrite(r"D:\UCC\Thesis\segment-anything-main\grid\0_0.png", convolved_image)
    # #cv2.imwrite(r"D:\UCC\Thesis\segment-anything-main\grid\0_0.png", contrast_image)
    
    # # edges = cv2.Canny(image, 100, 255)
    # edges = Canny(image=image)
    
    
    # masks(input_dir=r"D:\UCC\Thesis\segment-anything-main\grid", output_dir=r"D:\UCC\Thesis\segment-anything-main\test")
    # # plt.subplot(221), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Original Image')
    # # plt.subplot(222), plt.imshow(cv2.cvtColor(convolved_image, cv2.COLOR_BGR2RGB)), plt.title('Convolved Image')
    # # plt.subplot(111), plt.imshow(cv2.cvtColor(contrast_image, cv2.COLOR_BGR2RGB)), plt.title('Contrasted Image')
    # # plt.subplot(224), plt.imshow(cv2.cvtColor(edges, cv2.COLOR_BGR2RGB)), plt.title('edged Image')
    # # cv2.bitwise_and()
    
    
    # edges_inverted = cv2.bitwise_not(edges)
    # overlapped_mask = cv2.bitwise_and(image, image, mask=edges_inverted)

    # cv2.imwrite(r"D:\UCC\Thesis\segment-anything-main\grid\0_0.png", overlapped_mask)
    # masks(input_dir=r"D:\UCC\Thesis\segment-anything-main\grid", output_dir=r"D:\UCC\Thesis\segment-anything-main\test")

    # plt.subplot(141), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Image')
    # plt.subplot(142), plt.imshow(cv2.cvtColor(edges, cv2.COLOR_BGR2RGB)), plt.title('Canny-Edges')
    # plt.subplot(143), plt.imshow(cv2.cvtColor(edges_inverted, cv2.COLOR_BGR2RGB)), plt.title('Inverted Canny edges')
    # plt.subplot(144), plt.imshow(cv2.cvtColor(overlapped_mask, cv2.COLOR_BGR2RGB)), plt.title('Final')
    # plt.show()

    image = cv2.imread(r'D:\UCC\Thesis\segment-anything-main\grid\131.png', cv2.IMREAD_GRAYSCALE)
    
    # Example usage
    # binary_mask = cv2.imread('path/to/your/binary_mask.png', cv2.IMREAD_GRAYSCALE)
    
    bounding_boxes = split_connected_components(image)

    print("Bounding boxes:", bounding_boxes)



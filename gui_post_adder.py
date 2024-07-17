import os
import numpy as np
import cv2
import pandas as pd

def mask_combiner(path, colour_masks, patch_size=512):
    os.makedirs(colour_masks, exist_ok=True)

    for root, subdirs, masks in os.walk(path):
        if masks ==() or masks==[]:
            continue
        
        folder_name = str(os.path.basename(root))
        csv_file = os.path.join(root, folder_name + ".csv")
        df = pd.read_csv(csv_file, header=0).to_dict('records')
        combined_mask = np.zeros((patch_size, patch_size, 3), dtype=np.uint8)
        
        for row in df:
            mask_path = os.path.join(root, str(row['id']) + '.png')
            label = row['label']

            color = (0, 0, 128)  # Red for RBC, white for WBC
            if label == 'WBC':
                color = (255, 255, 255)

            # Load binary mask
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            # print(mask.shape)
            # # Convert single-channel mask to 3-channel (BGR)
            # mask_rgb = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            # print(mask.shape)

            # Apply color to combined mask where mask is white (255)
            combined_mask[mask == 255] = color
        
        combined_mask_path = os.path.join(colour_masks, folder_name + ".png")
        cv2.imwrite(combined_mask_path, combined_mask)




    

    

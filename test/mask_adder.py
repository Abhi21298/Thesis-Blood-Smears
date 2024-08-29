import cv2
import sys
import os
import argparse
import pandas as pd 
import numpy as np
parser = argparse.ArgumentParser(description= "Code to create complete \
                                 masks and separate masks and csv files \
                                 into separate folders")

parser.add_argument("--input_dir",
                    type=str,
                    required=True,
                    default=None,
                    help = "Input folder containing sub-folders of masks")

parser.add_argument("--csv_dir",
                    type=str,
                    required=True,
                    default=None,
                    help = "folder to store all csv_files")

parser.add_argument("--full_masks_dir",
                    type=str,
                    required=True,
                    default=None,
                    help = "folder to store all masks added up together for each image")

def mask_and_csv(args):

    inp_path = str(args.input_dir)
    csv_dir = str(args.csv_dir)
    full_masks_dir = str(args.full_masks_dir)

    if not os.path.exists(inp_path):
        raise("Enter a proper input directory which actually exists! :-( ")
    
    os.makedirs(csv_dir, exist_ok=True)
    os.makedirs(full_masks_dir, exist_ok=True)

    for subroot, subdirs, masks in os.walk(inp_path):
        if masks == () or masks == []:
            continue
        
        file_name = str(subroot.rsplit("\\", maxsplit = 1)[1])
        csv_file = os.path.join(subroot, file_name + ".csv")
        df = pd.read_csv(csv_file, header = 0).to_dict('records')

        new_df = []
        img_overall = np.zeros((512,512), dtype=np.uint8)
        for row in df:

            if pd.isna(row["label"]):
                continue
            
            new_df.append(row)
            sub_img_path = os.path.join(subroot, str(row["id"]) + ".png")
            img = cv2.imread(sub_img_path, 0)

            if row["label"] == "RBC":
                img_overall = cv2.add(img_overall, img)
            else:
                print(file_name)
                img_overall = cv2.add(img_overall, img//2)

        img_path = os.path.join(full_masks_dir, file_name + ".png")
        cv2.imwrite(img_path, img_overall)

        csv_path = os.path.join(csv_dir, file_name + ".csv")
        df_new = pd.DataFrame(new_df)
        df_new.to_csv(csv_path, index= False)



if __name__ == "__main__":
    args = parser.parse_args()
    mask_and_csv(args)




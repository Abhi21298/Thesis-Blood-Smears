from PIL import Image, ImageTk
import os
from gui_grids import save_image_patches
import argparse
from gui_masks_generator import masks
from gui_filter_masks import edit_csv
from gui_create_cutouts import create_cutouts_dataset
from gui_predict import prediction
import tensorflow as tf

parser = argparse.ArgumentParser(description="Enter absolute path (full path) of the image")

parser.add_argument("--input_image",
                    required=True,
                    type=str,
                    default=None,
                    help="Enter full image path obtained from properties")

        
def predict_image(args):
    
    file_path = args.input_image

    if os.path.exists(file_path):
        directory = os.path.dirname(file_path)
        grids_dir = os.path.join(directory, r"gui_predict/grids")
        masks_dir = os.path.join(directory, r"gui_predict/sam_masks")
        final_cutouts = os.path.join(directory, r"gui_predict/final_cutouts")

        print(grids_dir)
        print(masks_dir)
        print(final_cutouts)

        #save_image_patches(file_path, grids_dir)

        #masks(input_dir=grids_dir, output_dir=masks_dir)

        #edit_csv(masks_dir)
        
        #create_cutouts_dataset(orig_image_path=grids_dir, input_dir=masks_dir, output_dir=final_cutouts)

        model = tf.keras.models.load_model(r"rbc_wbc_classifier.h5")
        counts = prediction(model=model, path=final_cutouts)
        RBC = counts["RBC"]
        WBC = counts["WBC"]
        print("RBC count:", RBC, ", WBC Count:", WBC)
    else:
        print("Enter the correct path of the image")


if __name__ == "__main__":
    args = parser.parse_args()
    predict_image(args)

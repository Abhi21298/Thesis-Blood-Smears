from PIL import Image, ImageTk
import os, shutil
from gui_grids import save_image_patches
import argparse
from gui_masks_generator import masks
from gui_filter_masks import edit_csv
from gui_create_cutouts import create_cutouts_dataset
from gui_predict import prediction
import tensorflow as tf
from gui_post_adder import masks_combiner
from gui_masks_stitcher import masks_stitcher
from gui_adjusted_cell_counter import count_cells

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
        masks_dir_copy = os.path.join(directory, r"gui_predict/sam_masks_copy")
        colour_masks = os.path.join(directory, r"gui_predict/colour_masks")
        final_cutouts = os.path.join(directory, r"gui_predict/final_cutouts")

        print(grids_dir)
        print(masks_dir)
        print(final_cutouts)

        # save_image_patches(file_path, grids_dir)

        # masks(input_dir=grids_dir, output_dir=masks_dir)
        shutil.copytree(masks_dir, masks_dir_copy) # backup of original masks to play with testing the below functions

        edit_csv(masks_dir)
        
        create_cutouts_dataset(orig_image_path=grids_dir, input_dir=masks_dir, output_dir=final_cutouts)

        model = tf.keras.models.load_model(r"rbc_wbc_classifier.h5", compile=False)

        model.compile(
            optimizer=tf.keras.optimizers.Adam(),  # Use an appropriate optimizer
            loss='binary_crossentropy',  # Use binary cross-entropy as the loss function
            metrics=['accuracy']  # Add any metrics you need
        )

        counts = prediction(model=model, path=final_cutouts, masks_dir = masks_dir)

        masks_combiner(masks_dir, colour_masks)
        
        overall_mask_path = masks_stitcher(file_path, colour_masks)

        RBC = counts["RBC"]
        WBC = counts["WBC"]
        print("RBC count:", RBC, ", WBC Count:", WBC)

        print("*"*100)
        print()
        print("Adjusted cell count")

        count_cells(f"{overall_mask_path}")
    else:
        print("Enter the correct path of the image")

if __name__ == "__main__":
    args = parser.parse_args()
    predict_image(args)

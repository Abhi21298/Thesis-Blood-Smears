import tkinter as tk
from tkinter import filedialog
from tkinter import Label, Button
from PIL import Image, ImageTk
import os
from time import sleep
import shutil
from datetime import datetime
from gui_increase_brightness import increase_brightness
from gui_grids import save_image_patches
from gui_masks_generator import masks
from gui_filter_masks import edit_csv
from gui_create_cutouts import create_cutouts_dataset
from gui_predict import prediction
from gui_post_adder import masks_combiner
from gui_masks_stitcher import masks_stitcher
from gui_adjusted_cell_counter import count_cells

import tensorflow as tf
#import torch



def upload_image():

    file_path = filedialog.askopenfilename(filetypes= [("Blood smear Image files", "*.tif *.tiff *.png *.jpg *.jpeg *.JPG *.JPEG *.TIF *.TIFF")])
    if file_path:

        img = Image.open(file_path)
        img.thumbnail((500,500))

        # Below four lines are important to keep reference to image alive
        # or else image display will disappear from the GUI window
        # Reason function death will result in python garbage collector erasing 
        # image from memory.
        img_display = ImageTk.PhotoImage(img)
        label_image.configure(image=img_display)
        label_image.image = img_display
        label_image.file_path = file_path

def display_image(image_path, target_label):
    img = Image.open(image_path)
    img.thumbnail((200, 200))
    img_display = ImageTk.PhotoImage(img)
    target_label.configure(image=img_display)
    target_label.image = img_display

def predict_image():
    file_path = getattr(label_image, 'file_path', None)
    
    if file_path:
        directory = os.path.dirname(file_path)
        name = "_".join(list(map(str, os.path.splitext(os.path.basename(file_path))[0].split(" "))))
        #result_folder = "gui_predict_2024-07-18T22:27:48"
        result_folder = "gui_predict_"+name+"_"+str(datetime.now().strftime("%Y-%m-%dT%H:%M:%S"))
        grids_dir = os.path.join(directory, os.path.join(result_folder, "grids"))
        masks_dir = os.path.join(directory, os.path.join(result_folder, "sam_masks"))
        masks_dir_copy = os.path.join(directory, os.path.join(result_folder, "sam_masks_copy"))
        colour_masks = os.path.join(directory, os.path.join(result_folder, "colour_masks"))
        final_cutouts = os.path.join(directory, os.path.join(result_folder, "final_cutouts"))

        print(grids_dir)
        print(masks_dir)
        print(final_cutouts)

        increase_brightness(file_path)
        
        save_image_patches(file_path, grids_dir)

        masks(input_dir=grids_dir, output_dir=masks_dir)
        shutil.copytree(masks_dir, masks_dir_copy) # backup of original masks to play with testing the below functions

        edit_csv(masks_dir)
        
        create_cutouts_dataset(orig_image_path=grids_dir, input_dir=masks_dir, output_dir=final_cutouts)

        model = tf.keras.models.load_model(r"rbc_wbc_classifier_finetuned.h5", compile=False)

        model.compile(
           optimizer=tf.keras.optimizers.Adam(),  # Use an appropriate optimizer
           loss='binary_crossentropy',  # Use binary cross-entropy as the loss function
           metrics=['accuracy']  # Add any metrics you need
        )

        prediction(model=model, path=final_cutouts, masks_dir = masks_dir)
        
        sleep(5)
        masks_combiner(masks_dir, colour_masks)
        
        sleep(5)
        overall_mask_path = masks_stitcher(file_path, colour_masks)
        
        sleep(3)
        RBC, WBC = count_cells(f"{overall_mask_path}")
        display_image(file_path, label_image)
        display_image(overall_mask_path, label_final_mask)
        display_image(os.path.join(os.path.dirname(overall_mask_path), "final_segmentation_overlap.png"), label_overlap)
        result_text.set(f"RBC Count: {RBC}, WBC Count: {WBC}")
        # result_text.set(f"RBC Count: {0}, WBC Count: {0}")



if __name__ == "__main__":
    
    window = tk.Tk()
    window.title("RBC/WBC Counter")
    window.geometry("1000x800")

    label_image = Label(window)
    label_image.pack(pady=10)

    label_final_mask = Label(window)
    label_final_mask.pack(pady=10)

    label_overlap = Label(window)
    label_overlap.pack(pady=10)

    button_upload = Button(window, text="Upload Image", command=upload_image)
    button_upload.pack(pady=10)

    button_predict = Button(window, text="Predict", command=predict_image)
    button_predict.pack(pady=10)

    result_text = tk.StringVar()
    label_result = Label(window, textvariable=result_text, font=("Helvetica", 16))
    label_result.pack(pady=20)

    window.mainloop()




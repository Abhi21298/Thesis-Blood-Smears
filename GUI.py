import tkinter as tk
from tkinter import filedialog
from tkinter import Label, Button
from PIL import Image, ImageTk
import os
from gui_grids import save_image_patches
# import argparse
#from mask_generator import main as main_mask_generator
from gui_masks_generator import masks
from gui_filter_masks import edit_csv
from gui_create_cutouts import create_cutouts_dataset
from gui_predict import prediction

import tensorflow as tf
#import torch

model = tf.keras.models.load_model(r"D:\UCC\Thesis\rbc_wbc_classifier.h5")

def upload_image():

    file_path = filedialog.askopenfilename(filetypes= [("Blood smear Image files", "*.tif;*.png;*.jpg;*.jpeg;")])
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
        
def predict_image():
    file_path = getattr(label_image, 'file_path', None)
    
    if file_path:
        directory = os.path.dirname(file_path)
        grids_dir = os.path.join(directory, r"gui_predict\grids")
        masks_dir = os.path.join(directory, r"gui_predict\sam_masks")
        final_cutouts = os.path.join(directory, r"gui_predict\final_cutouts")

        print(grids_dir)
        print(masks_dir)
        print(final_cutouts)

        # save_image_patches(file_path, grids_dir)

        # masks(input_dir=grids_dir, output_dir=masks_dir)

        # edit_csv(masks_dir)
        
        # create_cutouts_dataset(orig_image_path=grids_dir, input_dir=masks_dir, output_dir=final_cutouts)

        counts = prediction(model=model, path=final_cutouts)
        RBC = counts["RBC"]
        WBC = counts["WBC"]
        result_text.set(f"RBC Count: {RBC}, WBC Count: {WBC}")
        # result_text.set(f"RBC Count: {0}, WBC Count: {0}")



if __name__ == "__main__":
    
    window = tk.Tk()
    window.title("RBC/WBC counter")
    window.geometry("700x700")

    label_image = Label(window)
    label_image.pack(pady=20)

    button_upload = Button(window, text = "Upload Image", command = upload_image)
    button_upload.pack(pady=10)

    button_predict = Button(window, text = "Predict", command= predict_image)
    button_predict.pack(pady=10)

    result_text = tk.StringVar()
    label_result = Label(window, textvariable=result_text, font=("Helvetica", 16))
    label_result.pack(pady=20)

    window.mainloop()




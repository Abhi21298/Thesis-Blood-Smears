import tkinter as tk
from tkinter import filedialog
from tkinter import Label, Button
from PIL import Image, ImageTk
import os
from time import sleep, time
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


def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Blood smear Image files", "*.tif *.tiff *.png *.jpg *.jpeg *.JPG *.JPEG *.TIF *.TIFF")])

    if file_path:
        img = Image.open(file_path)
        img.thumbnail((700, 700))

        img_display = ImageTk.PhotoImage(img)
        label_image.configure(image=img_display)
        label_image.image = img_display
        label_image.file_path = file_path

        # Update layout to center the uploaded image
        label_image.grid(row=1, column=1, pady=10)
        button_upload.grid(row=2, column=1, pady=10)
        button_predict.grid(row=3, column=1, pady=10)


def display_image(image_path, target_label):
    img = Image.open(image_path)
    img.thumbnail((600, 500))
    img_display = ImageTk.PhotoImage(img)
    target_label.configure(image=img_display)
    target_label.image = img_display


def predict_image():
    file_path = getattr(label_image, 'file_path', None)

    if file_path:
        start = time()
        directory = os.path.dirname(file_path)
        name = "_".join(list(map(str, os.path.splitext(os.path.basename(file_path))[0].split(" "))))
        result_folder = "gui_predict_" + name + "_" + str(datetime.now().strftime("%Y-%m-%dT%H:%M:%S"))
        grids_dir = os.path.join(directory, os.path.join(result_folder, "grids"))
        masks_dir = os.path.join(directory, os.path.join(result_folder, "sam_masks"))
        masks_dir_copy = os.path.join(directory, os.path.join(result_folder, "sam_masks_copy"))
        colour_masks = os.path.join(directory, os.path.join(result_folder, "colour_masks"))
        final_cutouts = os.path.join(directory, os.path.join(result_folder, "final_cutouts"))

        increase_brightness(file_path)
        save_image_patches(file_path, grids_dir)
        masks(input_dir=grids_dir, output_dir=masks_dir)
        shutil.copytree(masks_dir, masks_dir_copy)

        edit_csv(masks_dir)
        create_cutouts_dataset(orig_image_path=grids_dir, input_dir=masks_dir, output_dir=final_cutouts)

        model = tf.keras.models.load_model(r"rbc_wbc_classifier_finetuned.h5", compile=False)
        model.compile(
           optimizer=tf.keras.optimizers.Adam(),
           loss='binary_crossentropy',
           metrics=['accuracy']
        )

        prediction(model=model, path=final_cutouts, masks_dir=masks_dir)
        sleep(5)
        masks_combiner(masks_dir, colour_masks)
        sleep(5)
        overall_mask_path = masks_stitcher(file_path, colour_masks)
        RBC, WBC = count_cells(f"{overall_mask_path}")
        display_image(file_path, label_image)
        display_image(overall_mask_path, label_final_mask)
        display_image(os.path.join(os.path.dirname(overall_mask_path), "final_segmentation_overlap.png"), label_overlap)

        # Hide the buttons after prediction
        button_upload.grid_forget()
        button_predict.grid_forget()

        # Arrange the images side by side and centered
        label_image.grid(row=0, column=0, padx=10, pady=10)
        label_final_mask.grid(row=0, column=1, padx=10, pady=10)
        label_overlap.grid(row=0, column=2, padx=10, pady=10)

        result_text.set(f"RBC Count: {RBC}, WBC Count: {WBC}")
        label_result.grid(row=1, column=0, columnspan=3, pady=20)

        end = time()
        width, height = Image.open(file_path).size
        print(f"Execution time for image with size ({width},{height}) = {end-start} seconds")


if __name__ == "__main__":
    window = tk.Tk()
    window.title("RBC/WBC Counter")
    window.geometry("1000x1000")

    # Set up the grid with equal weight columns to center the content
    window.grid_columnconfigure(0, weight=1)
    window.grid_columnconfigure(1, weight=1)
    window.grid_columnconfigure(2, weight=1)

    # Create labels for images (but don't place them yet)
    label_image = Label(window)
    label_final_mask = Label(window)
    label_overlap = Label(window)

    # Initially, only the Upload button is visible and centered
    button_upload = Button(window, text="Upload Image", command=upload_image)
    button_upload.grid(row=1, column=1, pady=20)

    # Predict button hidden initially
    button_predict = Button(window, text="Predict", command=predict_image)
    button_predict.grid(row=2, column=1, pady=20)

    result_text = tk.StringVar()
    label_result = Label(window, textvariable=result_text, font=("Helvetica", 16))

    window.mainloop()

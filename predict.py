import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

def prediction(path):
    #model = load_model(r"D:\UCC\Thesis\segment-anything-main\assets\compiled_dataset\rbc_wbc_classifier.h5")
    model = load_model(r"D:\UCC\Thesis\rbc_wbc_classifier.h5")
    imgs = os.listdir(path)
    
    tracker = {"RBC": [], "WBC": []}

    for subroot, subdirs, imgs in os.walk(path):
        if imgs == [] or imgs == () or len(subdirs) > 0:
            continue
        
        for img in imgs:
            if img.endswith(".png"):
                ## preprocess image
                imag = image.load_img(os.path.join(subroot, img), target_size = (96,96))
                imag_array = image.img_to_array(imag)
                imag_array = np.expand_dims(imag_array, axis = 0)

                ## standardise it
                imag_array = imag_array/255.0

                res = model.predict(imag_array)
                res = tf.nn.sigmoid(res[0])
                classification = "RBC" if res[0] < 0.5 else "WBC"

                if classification == "WBC":
                    plt.imshow(image.load_img(os.path.join(subroot, img)))
                    plt.title(f"Class result for {img} - {classification}")
                    plt.axis('off')
                    plt.show()

                tracker[classification].append(img)

    print("RBC count:", len(tracker["RBC"]))
    print("WBC count:", len(tracker["WBC"]))

    print("WBC images:", tracker["WBC"])

if __name__ == "__main__":
    prediction(r"D:\UCC\Thesis\segment-anything-main\assets\compiled_dataset\test")




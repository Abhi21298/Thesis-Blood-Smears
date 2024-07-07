import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import shutil

def prediction(model, path):
    model = model
    #model = load_model(model)
    imgs = os.listdir(path)
    
    tracker = {"RBC": 0, "WBC": 0}

    for subroot, _, imgs in os.walk(path):
        if imgs == [] or imgs == ():
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

                # if classification == "WBC":
                #     plt.imshow(image.load_img(os.path.join(subroot, img)))
                #     plt.title(f"Class result for {img} - {classification}")
                #     plt.axis('off')
                #     plt.show()

                tracker[classification] += 1
                src_path = os.path.join(subroot, img)
                dst_path = os.path.join(subroot, classification)

                os.makedirs(dst_path, exist_ok=True)
                shutil.move(src_path, dst_path)

    #print("RBC count:", len(tracker["RBC"]))
    #print("WBC count:", len(tracker["WBC"]))

    #print("WBC images:", tracker["WBC"])

    return tracker

# if __name__ == "__main__":
#     prediction()
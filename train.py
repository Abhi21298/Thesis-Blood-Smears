import os
import argparse
import shutil
import random
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
import numpy as np

parser = argparse.ArgumentParser(description="Copy mask folders created by SAM")

parser.add_argument("--work_dir",
                    required=True,
                    type=str,
                    default=None,
                    help="Pass the root folder working directory")

def balance_dataset(rbc_dir, excluded_dir, num_keep=500):

    # Ensure the excluded directory exists
    if not os.path.exists(excluded_dir):
        os.makedirs(excluded_dir)

    # List all images in the RBC directory
    rbc_images = [img for img in os.listdir(rbc_dir) if img.endswith('.png')]
    if num_keep >= len(rbc_images):
        print(f"Number of images to keep ({num_keep}) is greater than or equal to the total images ({len(rbc_images)}). No images will be moved.")
        return
    
    # Randomly select a subset of images to keep
    images_to_keep = random.sample(rbc_images, num_keep)
    images_to_exclude = set(rbc_images) - set(images_to_keep)

    for img in images_to_exclude:
        src_path = os.path.join(rbc_dir, img)
        dst_path = os.path.join(excluded_dir, img)
        shutil.move(src_path, dst_path)
        print(f"Moved {img} to excluded directory.")
    
    print(f"Dataset balanced. Kept {num_keep} images in {rbc_dir} and moved {len(images_to_exclude)} images to {excluded_dir}.")

def plot_training_history(history, output_path='training_history.png'):
    # Retrieve the data from the history object
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    
    # Plot training and validation accuracy
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, 'bo', label='Training accuracy')
    plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend()

    # Plot training and validation loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, 'ro', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    # Save the plot as an image file
    plt.savefig(output_path)
    plt.show()

def CNN_model():

    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(96, 96, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.2),
        Dense(1)  # No activation function here (linear output)
    ])

    return model

def train_CNN(args):
    work_dir = args.work_dir

    # Image data generator
    train_datagen = image.ImageDataGenerator(rescale=1.0/255.0)
    validation_datagen = image.ImageDataGenerator(rescale=1.0/255.0)
    test_datagen = image.ImageDataGenerator(rescale=1.0/255.0)

    # Load datasets
    train_generator = train_datagen.flow_from_directory(
        os.path.join(work_dir, "train"),
        target_size=(96, 96),
        batch_size=4,
        class_mode='binary'
    )

    validation_generator = validation_datagen.flow_from_directory(
        os.path.join(work_dir, "val"),
        target_size=(96, 96),
        batch_size=4,
        class_mode='binary'
    )

    test_generator = test_datagen.flow_from_directory(
        os.path.join(work_dir, "test"),
        target_size=(96, 96),
        batch_size=4,
        class_mode='binary'
    )

    # Build the model
    model = CNN_model()

    # Compile the model with binary crossentropy and from_logits=True
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )

    # Train the model
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // train_generator.batch_size,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // validation_generator.batch_size,
        epochs=20
    )

    # Evaluate the model
    test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // test_generator.batch_size)
    print(f'Test accuracy: {test_acc}')

    # Save the model
    #model.save(os.path.join(work_dir, 'rbc_wbc_classifier.h5'))
    model.save(os.path.join(work_dir, 'rbc_wbc_classifier_average.h5'))
    plot_training_history(history, output_path= os.path.join(work_dir, 'training_history.png'))

def train_CNN_CV(args):
    label_map = {"RBC": 0, "WBC": 1}
    work_dir = args.work_dir
    weights_list = []
    accuracies = []

    if not os.path.exists(work_dir):
        raise("Enter a valid work directory path with images")
    images = []
    labels = []
    for dirs in ["train", "val"]:
        for root, _, files in os.walk(os.path.join(work_dir, f"{dirs}")):
            if files == [] or files == ():
                continue
            
            n = len(files)
            #images.extend([file for file in files if file.endswith(".png")])
            for file in files:
                img = image.load_img(os.path.join(root, file))
                imag_array = image.img_to_array(img)
                #img = np.expand_dims(img, axis = 0)
                imag_array = imag_array/255.0
                images.append(imag_array)
            label = label_map[os.path.basename(root)]
            labels.extend([label]*n)

    images = np.array(images)
    labels = np.array(labels)

    kfold = StratifiedKFold(n_splits= 10, random_state= 6500, shuffle= True)
    fold = 1

    for train_index, val_index in kfold.split(images, labels):
        x_train, y_train = images[train_index], labels[train_index]
        x_val, y_val = images[val_index], labels[val_index]

        model = CNN_model()

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
            metrics=['accuracy'],
        )

        history = model.fit(x_train, y_train, validation_data= (x_val, y_val),\
                            epochs= 30, batch_size= 4, verbose= 1)
        
        weights_list.append(model.get_weights())

        y_val_pred = model.predict(x_val)
        y_val_pred = tf.nn.sigmoid(y_val_pred)
        y_val_pred_classes = np.array((y_val_pred > 0.5))
        accuracy = accuracy_score(np.array(y_val), y_val_pred_classes)
        accuracies.append(accuracy)
        print(f"Fold {fold} accuracy: {accuracy:.4f}")
        fold += 1
    
    average_accuracy = np.mean(accuracies)
    print(f"Average accuracy across 10 folds: {average_accuracy:.4f}")

    # Average the weights
    average_weights = [np.mean([weights_list[fold][layer] for fold in range(10)], axis=0) for layer in range(len(weights_list[0]))]

    # Create a new model and set its weights to the averaged weights
    average_model = CNN_model()
    average_model.set_weights(average_weights)

    # Save the averaged model
    average_model.save(os.path.join(work_dir, 'rbc_wbc_classifier.h5'))

if __name__ == "__main__":
    args = parser.parse_args()
    rbc_train_dir = os.path.join(args.work_dir, r'train/RBC')
    excluded_dir = os.path.join(args.work_dir, r'excluded')

    #os.makedirs(excluded_dir, exist_ok=True)
    #balance_dataset(rbc_train_dir, excluded_dir, num_keep=500)
    #train_CNN(args)
    
    train_CNN_CV(args)
import cv2
import numpy as np

def is_very_dark(img):

    image = cv2.imread(img)
    # Convert the image to grayscale to calculate brightness
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Calculate the average brightness
    avg_brightness = np.mean(gray_image)
    print(f"Average brightness: {avg_brightness}")
    # Check if the average brightness is below the threshold
    return image, avg_brightness #< threshold

def increase_brightness(image, brightness_thresh=140):
    img_path = image
    image, current_brightness = is_very_dark(image)

    diff_brightness = brightness_thresh - current_brightness if brightness_thresh - current_brightness >=0 else 0
    # Convert to HSV (Hue, Saturation, Value) color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Increase the V (value) channel by the specified value
    h, s, v = cv2.split(hsv)
    v = np.clip(v + diff_brightness, 0, 255).astype(h.dtype)
    final_hsv = cv2.merge((h, s, v))
    # Convert back to BGR color space
    bright_image = cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    cv2.imwrite(img_path, bright_image)
    return bright_image

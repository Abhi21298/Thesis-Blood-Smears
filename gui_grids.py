from PIL import Image
import os
import shutil

def save_image_patches(image_path, output_dir, patch_size=512):
   
    img = Image.open(image_path)
    img_width, img_height = img.size
    
    if img.width <= patch_size and img.height <= patch_size:
        shutil.copy(image_path, output_dir)
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    patch_count = 0
    for top in range(0, img_height, patch_size):
        for left in range(0, img_width, patch_size):
            bottom = min(top + patch_size, img_height)
            right = min(left + patch_size, img_width)
            patch = img.crop((left, top, right, bottom))
            patch.save(os.path.join(output_dir, f'{patch_count}.png'))
            patch_count += 1

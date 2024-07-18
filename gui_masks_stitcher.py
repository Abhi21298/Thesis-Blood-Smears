from PIL import Image
import os
import sys
import cv2

def masks_stitcher(original_img_path, path, patch_size=512, alpha=0.5):

    original_img = cv2.imread(original_img_path)
    original_img_dims = original_img.shape
    orig_h, orig_w, _ = original_img_dims
    
    if not os.path.exists(path):
        raise("Enter a valid path to work on!")
        sys.exit()
    
    masks = [mask for mask in os.listdir(path) if mask.endswith(".png")]

    # masks = sorted(masks, key = lambda x: int(x.split(".")[0]))

    patch = 0

    row_patches = (orig_w // patch_size) + 1
    col_patches = (orig_h // patch_size) + 1
    overall_image = Image.new('RGB', (orig_w, orig_h))
    
    for row in range(col_patches):
        for col in range(row_patches):

            img_path = os.path.join(path, f"{patch}.png")
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Required Image {patch}.png patch not found, please avoid manual tampering of {path} directory")
            
            grid = Image.open(img_path)
            left = col * patch_size
            top = row * patch_size
            overall_image.paste(grid, (left, top))
            patch += 1
    
    overall_image_savepath = os.path.join(os.path.dirname(path), "instance_segmented_mask.png")
    overall_image.save(overall_image_savepath)

    overall_image = cv2.imread(overall_image_savepath)

    # alpha = 0.5
    overlay_img = cv2.addWeighted(original_img, 1 - alpha, overall_image, alpha, 0)
    cv2.imwrite(os.path.join(os.path.dirname(path), "final_segmentation_overlap.png"), overlay_img)
    return overall_image_savepath

# if __name__ == "__main__":
#     masks_stitcher(r"D:\UCC\Thesis\Healthy_BF_Sample 3.tif", r"D:\UCC\Thesis\gui_predict_HS3\colour_masks")


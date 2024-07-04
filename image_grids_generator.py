from PIL import Image
import os

def split_image(image_path, output_folder, tile_size):
    # Open the image
    img = Image.open(image_path)
    img_width, img_height = img.size

    # Calculate the number of tiles in each dimension
    x_tiles = (img_width + tile_size - 1) // tile_size
    y_tiles = (img_height + tile_size - 1) // tile_size

    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Split the image into tiles
    for y in range(y_tiles):
        for x in range(x_tiles):
            left = x * tile_size
            upper = y * tile_size

            # Adjust left and upper boundaries for edge tiles to ensure overlap
            if x == x_tiles - 1:
                left = img_width - tile_size

            if y == y_tiles - 1:
                upper = img_height - tile_size
            
            # Define right and lower boundaries
            right = left + tile_size
            lower = upper + tile_size
            
            # Save the tile
            tile = img.crop((left, upper, right, lower))
            tile_name = f"HS2_{x}_{y}.png"
            tile.save(os.path.join(output_folder, tile_name))

# Usage
image_path = r"D:\UCC\Thesis\Healthy_BF_Sample 2.tif"
output_folder = r"D:\UCC\Thesis\segment-anything-main\assets\grid_images"
tile_size = 512

split_image(image_path, output_folder, tile_size)

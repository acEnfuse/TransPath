import os
from PIL import Image

# Get the directory path of the current script
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))

# Converts image to 64x64
def resize_and_pad(image_path, target_size_x, target_size_y):
    # this is a hard-coded path reference to TransPath/images/
    image_path = os.path.join(CURRENT_DIR, '..', '..', 'images', image_path)
    img = Image.open(image_path)
    original_width, original_height = img.size
    aspect_ratio = original_width / original_height

    if aspect_ratio > 1:
        new_width = target_size_x
        new_height = round(target_size_x / aspect_ratio)
    else:
        new_width = round(target_size_y * aspect_ratio)
        new_height = target_size_y

    # this is a hard-coded path reference to TransPath/map_data/
    rescaled_img_path = os.path.join(CURRENT_DIR, '..', '..', 'map_data', 'rescaled_map.png')
    target_size = (target_size_x, target_size_y)
    print(img.getextrema())
    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    padded_img = Image.new("L", target_size, color="black")
    padded_img.paste(img, ((target_size_x - new_width) // 2, (target_size_y - new_height) // 2))
    # Threshold the image
    print(padded_img.getextrema())
    padded_img = padded_img.point(lambda x: 0 if x < 5 else 255)
    padded_img.save(rescaled_img_path, 'PNG')
    print("Image resized!")

def create_start_or_goal_image(x_point, y_point, filename, target_size_x, target_size_y):
    # this is a hard-coded reference to TransPath/map_data
    output_path = os.path.join(CURRENT_DIR, '..', '..', 'map_data')
    image = Image.new('L', (target_size_x, target_size_y), 'black')
    pixels = image.load()
    pixels[x_point, y_point] = 255 # white pixel at the specified position
    filepath = os.path.join(output_path, filename + '.png')
    image.save(filepath, 'PNG')
    print('Image created successfully!')

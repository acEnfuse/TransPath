
from pathlib import Path

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image

pl.seed_everything(42)

from models.autoencoder import Autoencoder
from modules.planners import DifferentiableDiagAstar

def resize_and_pad_image(image, resolution):
    img = Image.fromarray(image)
    original_width, original_height = img.size
    aspect_ratio = original_width / original_height

    if aspect_ratio > 1:
        new_width = resolution[0]
        new_height = round(new_width / aspect_ratio)
    else:
        new_height = resolution[1]
        new_width = round(new_height * aspect_ratio)

    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
    padded_img = Image.new("L", resolution, color="black")
    padded_img.paste(img, ((resolution[0] - new_width) // 2, (resolution[1] - new_height) // 2))
    padded_img = padded_img.point(lambda x: 1 if x > 0 else 0)

    padding = (padded_img.width - img.width, padded_img.height - img.height)
    img = np.asarray(padded_img)

    return img, padding

def unpad_and_resize_image(image, padding, resolution):
    img = Image.fromarray(image)
    width, height = resolution

    cropped_img = img.crop((padding[0] / 2, padding[1] / 2, img.width - padding[0] / 2, img.height - padding[1] / 2))
    resized_img = cropped_img.resize((width, height), Image.Resampling.LANCZOS)

    return np.asarray(resized_img)

def create_input_tensor(file_path, resolution):
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    image, _ = resize_and_pad_image(image, resolution)
    tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    return tensor

def create_output_tensor(image, padding, resolution):
    dtype = image.dtype
    image = image[0, 0].cpu().numpy()
    image = unpad_and_resize_image(image, padding, resolution)
    image = torch.tensor(image, dtype=dtype).unsqueeze(0).unsqueeze(0)

    return image

def infer_path(pathfinding_method = 'f',
               model_resolution = (64, 64),
               img_resolution = (512, 512),
               goal_path = 'example/mw/goal.png',
               map_path = 'example/mw/map.png',
               start_path = 'example/mw/start.png',
               weights_path = 'weights/focal.pth'
               ):

    orig_resolution = Image.open(map_path).size
    _, padding = resize_and_pad_image(cv2.imread(map_path, cv2.IMREAD_GRAYSCALE), img_resolution)

    goal = create_input_tensor(goal_path, resolution = img_resolution)
    map_design = create_input_tensor(map_path, resolution = img_resolution)
    start = create_input_tensor(start_path, resolution = img_resolution)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = torch.load(weights_path, map_location = device)
    weights = weights['state_dict'] if Path(weights_path).suffix == '.ckpt' else weights

    model = None
    inputs = None

    if pathfinding_method in ['f', 'fw100']:
        inputs = torch.cat([map_design, start + goal], dim=1)
        model = Autoencoder(mode='f', resolution = model_resolution)
        model.load_state_dict(weights)

        _resolution = (img_resolution[0] // 2**3, img_resolution[1] // 2**3) # 3 is hardcoded downsample steps
        model.pos.change_resolution(_resolution, 1.)
        model.decoder_pos.change_resolution(_resolution, 1.)
        model.eval()

        if pathfinding_method == 'fw100':
            planner = DifferentiableDiagAstar(mode='f', f_w=100)
        else:
            planner = DifferentiableDiagAstar(mode=' f')

    elif pathfinding_method == 'cf':
        inputs = torch.cat([map_design, goal], dim=1)
        planner = DifferentiableDiagAstar(mode = 'k')
        model = Autoencoder(mode = 'k', resolution = model_resolution)
        model.load_state_dict(weights)
        _resolution = (img_resolution[0] // 2**3, img_resolution[1] // 2**3) # 3 is hardcoded downsample steps
        model.pos.change_resolution(_resolution, 1.)
        model.decoder_pos.change_resolution(_resolution, 1.)
        model.eval()

    elif pathfinding_method == 'w2':
        planner = DifferentiableDiagAstar(mode = 'default', h_w = 2)

    elif pathfinding_method == 'vanilla':
        planner = DifferentiableDiagAstar(mode = 'default', h_w = 1)

    else:
        raise ValueError("Invalid pathfinding_method value. Choose from 'f', 'fw100', 'cf', 'w2', 'vanilla'.")

    with torch.no_grad():
        if model:
            pred = (model(inputs) + 1) / 2
        else:
            pred = (map_design == 0) * 1.
        outputs = planner(
            pred,
            start,
            goal,
            (map_design == 0) * 1.
        )

    map_design = create_output_tensor(image=map_design.to(torch.uint8),
                                      padding=padding,
                                      resolution=orig_resolution
                                      )
    outputs.g = create_output_tensor(image=outputs.g.to(torch.float32),
                                     padding=padding,
                                     resolution=orig_resolution
                                     )
    outputs.paths = create_output_tensor(image=outputs.paths.to(torch.uint8),
                                         padding=padding,
                                         resolution=orig_resolution
                                         )
    outputs.histories = create_output_tensor(image=outputs.histories.to(torch.float32),
                                             padding=padding,
                                             resolution=orig_resolution
                                             )
    pred = create_output_tensor(image=pred.to(torch.float32),
                                padding=padding,
                                resolution=orig_resolution
                                )

    return {
        'map_design': map_design,
        'outputs': outputs,
        'prediction': pred
    }

if __name__ == "__main__":
    infer_path()
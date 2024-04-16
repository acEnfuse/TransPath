"""
Module to generate pathfinding results using an autoencoder-based approach.

This module provides functions to generate pathfinding results based on input map, start, and goal images
using an autoencoder-based approach.

Author: Ashwin Sakhare

"""

import argparse
from pathlib import Path
from typing import Union, Tuple

import cv2
import numpy as np
import pytorch_lightning as pl
import torch
from PIL import Image

pl.seed_everything(42)

from .models.autoencoder import Autoencoder
from .modules.planners import DifferentiableDiagAstar


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Generate pathfinding results using an autoencoder-based approach.")
    parser.add_argument("--map",
                        default='./example/mw/map.png',
                        help="Path to the map image file or the map image as a numpy array."
                        )
    parser.add_argument("--start",
                        default='./example/mw/start.png',
                        help="Path to the start image file or the start image as a numpy array."
                        )
    parser.add_argument("--goal",
                        default='./example/mw/goal.png',
                        help="Path to the goal image file or the goal image as a numpy array."
                        )
    parser.add_argument("--method",
                        default='f',
                        choices=['f', 'fw100', 'cf', 'w2', 'vanilla'],
                        help="Method for pathfinding."
                        )
    parser.add_argument("--model_resolution",
                        nargs=2,
                        type=int,
                        default=[64, 64],
                        help="Resolution of the autoencoder model."
                        )
    parser.add_argument("--img_resolution",
                        nargs=2,
                        type=int,
                        default=[512, 512],
                        help="Resolution of the input images."
                        )
    parser.add_argument("--weights_filepath",
                        default='./weights/focal.pth',
                        help="Path to the weights file or the weights image."
                        )

    return parser.parse_args()

def resize_and_pad_image(image: np.ndarray, resolution: Tuple[int, int]) -> Tuple[np.ndarray, Tuple[int, int]]:
    """
    Resize and pad the input image to match the specified resolution.

    Parameters
    ----------
    image : np.ndarray
        Input image as a numpy array.

    resolution : Tuple[int, int]
        Desired resolution (width, height).

    Returns
    -------
    Tuple[np.ndarray, Tuple[int, int]]
        Resized and padded image and padding applied.

    """

    img = Image.fromarray(image)
    original_width, original_height = img.size
    aspect_ratio = original_width / original_height

    if aspect_ratio > 1:
        new_width = resolution[0]
        new_height = round(new_width / aspect_ratio)
    else:
        new_height = resolution[1]
        new_width = round(new_height * aspect_ratio)

    # Resize without modifying the pixel values
    img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

    # Create a new black image of the desired resolution
    padded_img = Image.new("L", resolution, color="black")

    # Calculate the position to paste the resized image
    paste_position = ((resolution[0] - new_width) // 2, (resolution[1] - new_height) // 2)

    # Paste the resized image onto the padded image
    padded_img.paste(img, paste_position)

    # Convert to binary image preserving the pixel values
    padded_img = padded_img.point(lambda x: 1 if x > 0 else 0)

    # Calculate padding
    padding = (padded_img.width - img.width, padded_img.height - img.height)

    img = np.asarray(padded_img)

    return img, padding

def unpad_and_resize_image(image: np.ndarray, padding: Tuple[int, int], resolution: Tuple[int, int]) -> np.ndarray:
    """
    Unpad and resize the input image based on the given padding and resolution.

    Parameters
    ----------
    image : np.ndarray
        Input image as a numpy array.

    padding : Tuple[int, int]
        Padding applied to the image.

    resolution : Tuple[int, int]
        Desired resolution (width, height).

    Returns
    -------
    np.ndarray
        Unpadded and resized image.

    """

    img = Image.fromarray(image)
    width, height = resolution

    cropped_img = img.crop((padding[0] // 2,
                            padding[1] // 2,
                            round(img.width - padding[0] / 2),
                            round(img.height - padding[1] / 2)
                            ))
    resized_img = cropped_img.resize((width, height), Image.Resampling.LANCZOS)

    return np.asarray(resized_img)

def create_input_tensor(image: np.ndarray, resolution: Tuple[int, int]) -> torch.Tensor:
    """
    Create a torch tensor from the input image with the specified resolution.

    Parameters
    ----------
    image : np.ndarray
        Input image as a numpy array.

    resolution : Tuple[int, int]
        Desired resolution (width, height).

    Returns
    -------
    torch.Tensor
        Input tensor.

    """

    image, _ = resize_and_pad_image(image, resolution)
    tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    return tensor

def create_output_tensor(image: torch.Tensor, padding: Tuple[int, int], resolution: Tuple[int, int]) -> torch.Tensor:
    """
    Create a torch tensor from the output image with the specified padding and resolution.

    Parameters
    ----------
    image : torch.Tensor
        Output image as a torch tensor.

    padding : Tuple[int, int]
        Padding applied to the image.

    resolution : Tuple[int, int]
        Original resolution (width, height).

    Returns
    -------
    torch.Tensor
        Output tensor.

    """

    dtype = image.dtype
    image = image[0, 0].cpu().numpy()
    image = unpad_and_resize_image(image, padding, resolution)
    image = torch.tensor(image, dtype=dtype).unsqueeze(0).unsqueeze(0)

    return image

def get_path(map: Union[str, np.ndarray],
             start: Union[str, np.ndarray],
             goal: Union[str, np.ndarray],
             weights_filepath: Union[str, Image],
             pathfinding_method: str ='f',
             model_resolution: Tuple = (64, 64),
             img_resolution: Tuple = (512, 512)
             ):
    """
    Generate pathfinding results based on input map, start, and goal images using an autoencoder-based approach.

    Parameters
    ----------
    map : Union[str, np.ndarray]
        Path to the map image file or the map image as a numpy array.

    start : Union[str, np.ndarray]
        Path to the start image file or the start image as a numpy array.

    goal : Union[str, np.ndarray]
        Path to the goal image file or the goal image as a numpy array.

    pathfinding_method : str, optional
        Method for pathfinding, by default 'f'.

    model_resolution : Tuple[int, int], optional
        Resolution of the autoencoder model, by default (64, 64).

    img_resolution : Tuple[int, int], optional
        Resolution of the input images, by default (512, 512).

    weights_filepath : Union[str, Image], optional
        Path to the weights file.

    Returns
    -------
    dict
        Dictionary containing the map design, outputs, and prediction.

    """

    if isinstance(map, str):
        map = cv2.imread(map, cv2.IMREAD_GRAYSCALE)

    if isinstance(start, str):
        start = cv2.imread(start, cv2.IMREAD_GRAYSCALE)

    if isinstance(goal, str):
        goal = cv2.imread(goal, cv2.IMREAD_GRAYSCALE)

    assert map.shape == start.shape == goal.shape, "Dimension mismatch between map, start, and goal"

    orig_resolution = Image.fromarray(map).size
    _, padding = resize_and_pad_image(map, img_resolution)

    goal = create_input_tensor(goal, resolution = img_resolution)
    map_design = create_input_tensor(map, resolution = img_resolution)
    start = create_input_tensor(start, resolution = img_resolution)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    weights = torch.load(weights_filepath, map_location = device)
    weights = weights['state_dict'] if Path(weights_filepath).suffix == '.ckpt' else weights

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

    # Get path mask
    path_mask =  result['outputs'].paths[0, 0].cpu().numpy()

    # Get the indices where the array is equal to 1
    path_indices = np.where(path_mask == 1)

    # Convert indices to a list of tuples
    path = list(zip(path_indices[0], path_indices[1]))

    return {
        'path': path,
        'map': map_design,
        'planner_outputs': outputs,
        'model_outputs': pred
    }


if __name__ == "__main__":
    args = parse_args()

    from .visualizer import visualize

    result = get_path(map=args.map,
                      start=args.start,
                      goal=args.goal,
                      pathfinding_method=args.method,
                      model_resolution=(args.model_resolution[0], args.model_resolution[1]),
                      img_resolution=(args.img_resolution[0], args.img_resolution[1]),
                      weights_filepath=args.weights_filepath
                      )

    visualize(result)


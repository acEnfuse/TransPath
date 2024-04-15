import argparse
import cv2
import numpy as np
import os
import sys
import torch

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
src_dir = os.path.join(CURRENT_DIR, '..', '..', 'transpath')
sys.path.append(src_dir)

from utils import bw_map_data_generator as map_gen
import transpath.inference as inf

def generate_map_with_path(start_point, goal_point, results, file_name):
    start_x, start_y = start_point
    goal_x, goal_y = goal_point
    data =  torch.cat([results['map_design'], results['outputs'].paths, results['outputs'].histories - results['outputs'].paths], dim=1)
    np_data = data.numpy()
    image_data = np_data.transpose(0, 2, 3, 1)
    scaled_image_data = (image_data * 255).astype(np.uint8)
    start_dot_color = [191, 64, 191]
    goal_dot_color = [255, 140, 0]
    scaled_image_data[0][start_y][start_x][0] = start_dot_color[0]
    scaled_image_data[0][start_y][start_x][1] = start_dot_color[1]
    scaled_image_data[0][start_y][start_x][2] = start_dot_color[2]
    scaled_image_data[0][goal_y][goal_x][0] = goal_dot_color[0]
    scaled_image_data[0][goal_y][goal_x][1] = goal_dot_color[1]
    scaled_image_data[0][goal_y][goal_x][2] = goal_dot_color[2]
    script_dir = os.path.dirname(os.path.abspath(__file__))
    map_data_dir = os.path.join(script_dir, '..', '..', 'map_data')
    os.makedirs(map_data_dir, exist_ok = True)
    output_image_path = os.path.join(map_data_dir, file_name)
    cv2.imwrite(output_image_path, scaled_image_data[0])

def parse_args():
    parser = argparse.ArgumentParser(description = "Path Planning Visualization Script")
    parser.add_argument("--image_path", type = str, help = "Path to unscaled map image")
    parser.add_argument("--target_size_x", type = int, default = 64, help = "Dictates size x(width) of image")
    parser.add_argument("--target_size_y", type = int, default = 64, help = "Dictates size y(height) of image")
    parser.add_argument("--start_point_x", type = int, help = "Starting x point for start image")
    parser.add_argument("--start_point_y", type = int, help = "Starting y point for start image")
    parser.add_argument("--goal_point_x", type = int, help = "Goal x point for goal image")
    parser.add_argument("--goal_point_y", type = int, help = "Goal y point for goal image")
    parser.add_argument("--pathfinding_method", type = str, default = 'cf', help = 'Use Weighted A* w/CF by default')
    parser.add_argument("--pathfinding_output_filename", type = str, default = 'cf-output', help = 'Output filename w/o .png')
    args = parser.parse_args()
    return args

def main(args):
    start_filename = 'start_map'
    goal_filename = 'goal_map'
    map_gen.resize_and_pad(
        image_path = args.image_path, 
        target_size_x = args.target_size_x, 
        target_size_y = args.target_size_y
    )
    map_gen.create_start_or_goal_image(
        x_point = args.start_point_x, 
        y_point = args.start_point_y, 
        filename = start_filename, 
        target_size_x = args.target_size_x, 
        target_size_y = args.target_size_y
    )
    map_gen.create_start_or_goal_image(
        x_point = args.goal_point_x, 
        y_point = args.goal_point_y, 
        filename = goal_filename, 
        target_size_x = args.target_size_x, 
        target_size_y = args.target_size_y
    )
    results = inf.infer_path(
        pathfinding_method = args.pathfinding_method,
        model_resolution = (64, 64),
        img_resolution = (args.target_size_x, args.target_size_y),
        goal_path = 'map_data/goal_map.png',
        map_path = 'map_data/rescaled_map.png',
        start_path = 'map_data/start_map.png',
        weights_path = 'transpath/weights/focal.pth'
    )
    generate_map_with_path(
        start_point = (args.start_point_x, args.start_point_y), 
        goal_point = (args.goal_point_x, args.goal_point_y), 
        results = results, 
        file_name = str(args.pathfinding_output_filename + '.png')
    )

if __name__ == "__main__":
    args = parse_args()
    main(args)
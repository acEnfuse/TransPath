import cv2
import matplotlib.pyplot as plt
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
import pytorch_lightning as pl
pl.seed_everything(42)
import sys
sys.path.append('../..')
import torch

from transpath.models.autoencoder_for_notebook import Autoencoder
from transpath.modules.planners import DifferentiableDiagAstar

def load_image_tensor(file_path):
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    tensor = torch.tensor(image, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    tensor[tensor == 255] = 1
    return tensor


def transform_plan(image):
    result = torch.ones_like(image)
    result[image[:, :, 0] == 1] = torch.tensor([0.1, 0.1, 0.1])
    result[image[:, :, 1] == 1] = torch.tensor([1., 0., 0])
    result[image[:, :, 2] == 1] = torch.tensor([0., 1, 0.])
    return result


def infer_path():
    cf_weight_path = os.path.join(CURRENT_DIR, '..', 'weights', 'cf.pth')
    focal_weight_path = os.path.join(CURRENT_DIR, '..', 'weights', 'focal.pth')
    map_filepath = os.path.join(CURRENT_DIR, '..', '..', 'map_data', 'rescaled_map.png')
    start_filepath = os.path.join(CURRENT_DIR, '..', '..', 'map_data', 'start_map.png')
    goal_filepath = os.path.join(CURRENT_DIR, '..', '..', 'map_data', 'goal_map.png')

    model_cf = Autoencoder(mode='k', resolution=(64, 64))
    model_focal = Autoencoder(mode='f', resolution=(64, 64))

    model_cf.load_state_dict(torch.load(cf_weight_path))
    model_cf.eval()

    model_focal.load_state_dict(torch.load(focal_weight_path))
    model_focal.eval()

    map_design = load_image_tensor(map_filepath)
    start = load_image_tensor(start_filepath)
    goal = load_image_tensor(goal_filepath)

    inputs_g = torch.cat([map_design, goal], dim=1)
    inputs_sg = torch.cat([map_design, start + goal], dim=1)

    f_planner = DifferentiableDiagAstar(mode='f')
    fw100_planner = DifferentiableDiagAstar(mode='f', f_w=100)
    cf_planner = DifferentiableDiagAstar(mode='k')
    w2_planner = DifferentiableDiagAstar(mode='default', h_w=2)
    vanilla_planner = DifferentiableDiagAstar(mode='default', h_w=1)

    with torch.no_grad():
        pred_f = (model_focal(inputs_sg) + 1) / 2
        pred_cf = (model_cf(inputs_g) + 1) / 2

        outputs_f = f_planner(
            pred_f,
            start,
            goal,
            (map_design == 0) * 1.
        )

        outputs_cf = cf_planner(
            pred_cf,
            start,
            goal,
            (map_design == 0) * 1.
        )

        outputs_fw100 = fw100_planner(
            pred_f,
            start,
            goal,
            (map_design == 0) * 1.
        )

        outputs_w2 = w2_planner(
            (map_design == 0) * 1.,
            start,
            goal,
            (map_design == 0) * 1.
        )

        outputs_vanilla = vanilla_planner(
            (map_design == 0) * 1.,
            start,
            goal,
            (map_design == 0) * 1.
        )

    return {
        'map_design': map_design,
        'outputs_vanilla': outputs_vanilla,
        'outputs_w2': outputs_w2,
        'outputs_fw100': outputs_fw100,
        'outputs_cf': outputs_cf,
        'outputs_f': outputs_f,
        'pred_f': pred_f,
        'pred_cf': pred_cf
    }

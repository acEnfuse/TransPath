import matplotlib.pyplot as plt
import numpy as np


def visualize(result):
    img = np.asarray(result['map'][0, 0].cpu().numpy())
    img = np.repeat(img[:, :, np.newaxis], 3, axis=2)

    overlay = result['planner_outputs'].paths[0, 0].cpu().numpy() * 255
    overlay = np.repeat(overlay[:, :, np.newaxis], 3, axis=2)

    # Replace white pixels with red (set the red channel to 255)
    white_indices = np.all(overlay == [255, 255, 255], axis=2)
    overlay[white_indices] = [255, 0, 0]

    plot = img + overlay

    plt.imshow(plot)
    plt.show()
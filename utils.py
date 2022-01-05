"""Necessary fuctions for implementing the Hopfiled
"""

import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt


def read_images(path: str) -> np.ndarray:
    """Read images from path. 'path' is compatible with glob patterns.
    """
    images = []
    for img in glob.glob(path):
        img = cv2.imread(img, cv2.IMREAD_GRAYSCALE).reshape((-1,))
        images.append(img)
    images = np.array(images)
    return images


def img_to_pattern(images: np.ndarray) -> np.ndarray:
    """Convert image pixel values to '-1' and '1', and return a matrix of patterns compatible with the Hopfiled network.
    """
    P = np.where(images >= 128, 1, -1)
    return P.T


def perturbe(pattern: np.ndarray, alpha: float = 0.1) -> np.ndarray:
    """Add noise to a pattern with 'alpha' as the percentage of values in the pattern to change.
    """
    p = np.copy(pattern)
    idx = np.random.choice(range(p.shape[0]), size=int(alpha * p.shape[0]))

    for i in idx:
        if p[i] == 1:
            p[i] = -1
        else:
            p[i] = 1

    return p


def show(pattern: np.ndarray, title: str = ""):
    """Show an image from an input pattern.
    """
    img = pattern.reshape(
        (int(np.sqrt(pattern.shape[0])), int(np.sqrt(pattern.shape[0])))
    )

    plt.figure()
    plt.imshow(img, cmap="gray", interpolation="none", vmin=-1, vmax=1)
    plt.title(title)
    plt.show()


def sgn(x: np.ndarray) -> np.ndarray:
    """Sign activation function
    """
    return np.where(x >= 0, 1, -1)

import matplotlib.pyplot as plt
import numpy as np
import math

def visualizeLandmarks(img = None, points = None, ax = None, facecolor = 'r', r = 2, inverse_channel=True):
    if ax is None:
        fig, ax = plt.subplots()
    if not img is None:
        if inverse_channel:
            ax.imshow(img[:,:,::-1])
        else:
            ax.imshow(img)
    for p in points:
        pt = plt.Circle(p, facecolor = facecolor, radius = r)
        ax.add_artist(pt)

def visualizeLocalBBox(img, rects, ax = None, inverse_channel=False, **kwargs):
    if ax is None:
        fig, ax = plt.subplots(**kwargs)

    if not inverse_channel:
        ax.imshow(img)
    else:
        ax.imshow(img[:,:,::-1])

    NUM_COLORS = len(rects)
    cm = plt.get_cmap('gist_rainbow')

    for i, rect in enumerate(rects):
        c = cm(1.*i/NUM_COLORS)
        x1, x2, y1, y2 = rect
        r = plt.Rectangle((y1, x1), y2-y1, x2-x1, linewidth = 1, edgecolor=c, facecolor='none')
        ax.add_patch(r)
    return ax


def plotImages(
        images,
        titles = None,
        shape = None,
        inverse_channel = False,
        **kwargs
    ):
    n = len(images)

    if not shape is None:
        assert len(shape) == 2 and shape[0] * shape[1] >= n
        fig, axes = plt.subplots(nrows = shape[0], ncols = shape[1], **kwargs)
    else:
        if n <= 5:
            fig, axes = plt.subplots(nrows = 1, ncols = n, **kwargs)
        elif n >= 6:
            fig, axes = plt.subplots(nrows = 2, ncols = math.ceil(n / 2), **kwargs)

    if titles is None:
        titles = [''] * n

    if len(images) == 1:
        axes = np.array([axes])

    axes = axes.reshape((-1))

    for i in range(n):
        img = images[i]
        ax = axes[i]

        if inverse_channel:
            ax.imshow(img[:,:,::-1])
        else:
            ax.imshow(img)

        ax.set_xticks([])
        ax.set_yticks([])
        if i <= len(titles) - 1:
            ax.set_title(titles[i])

    return axes

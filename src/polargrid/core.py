import numpy as np
from scipy.spatial import ConvexHull
from .utils import select, approx


def polargrid(points: np.ndarray) -> np.ndarray:
    """
    Generate a structured square grid from scattered 2D points
    using convex hull layering.

    Parameters
    ----------
    points : ndarray of shape (N, 2)
        Input 2D points.

    Returns
    -------
    grid : ndarray of shape (n, n, 2)
        Structured grid of points.
    """
    pts = points - points.mean(axis=0)

    N = len(pts)
    nn, n, layersizes = approx(N)
    pts = pts[:nn]

    x, y = pts[:, 0], pts[:, 1]
    thetas = np.arctan2(y, x)
    thetas0 = np.abs(thetas)

    grid = np.zeros((n, n, 2))
    remaining_idx = np.arange(nn)

    for lsize in layersizes:
        temp_idx = remaining_idx.copy()
        layer = []

        empty = False
        while (len(layer) < 1.3 * lsize) and (not empty):
            if len(temp_idx) > 2:
                hull = ConvexHull(pts[temp_idx]).vertices
            else:
                hull = np.arange(len(temp_idx))
                empty = True

            hull_global = temp_idx[hull]
            layer += list(hull_global)

            mask = np.zeros(len(temp_idx), dtype=bool)
            mask[hull] = True
            temp_idx = temp_idx[~mask]

        layer = np.array(layer)

        # angular sort
        layer = layer[np.argsort(thetas[layer])]

        # thinning
        thinning = select(thetas[layer], lsize)
        layer = layer[thinning]

        # rotate
        start_idx = np.argmin(thetas0[layer])
        layer = np.roll(layer, -start_idx)

        # insert
        if lsize != 1:
            d = lsize // 4
            indices = [i * d for i in range(4)] + [lsize]
            pl = pts[layer]
            segments = [pl[slice(indices[i], indices[i + 1])] for i in range(4)]

            L = (n - (d + 1)) // 2
            R = L + d

            grid[L:R, L] = segments[0]
            grid[R, L:R] = segments[1]
            grid[R:L:-1, R] = segments[2]
            grid[L, R:L:-1] = segments[3]

        remaining_idx = np.setdiff1d(remaining_idx, layer)

    return grid
\# polargrid



Generate structured square grids from scattered 2D points using convex hull layering.



\## Installation



```bash

pip install polargrid



import numpy as np

from polargrid import polargrid



points = np.random.rand(1000, 2)

grid = polargrid(points)



points are now structured as a perfect grid, such that grid\[i, j] is the point with index i, j



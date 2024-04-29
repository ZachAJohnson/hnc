import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv

from scipy.spatial.distance import cdist
from scipy.interpolate import interp1d

import pyvista as pv

from hnc.hnc.constants import *


gij_file = "results/CHNC_Al_rs2.998_TeV1.000.dat"
gij_data = read_csv(gij_file, delim_whitespace=True, header=3)
rs = 2.998 # AU
r_data, gei_data = gij_data['r/ri']*rs, gij_data['g_ei']

Al_MD_file = "results/Luciano_Al_checlpoint_10000.npz" 
MD_data = np.load(Al_MD_file)
MD_array = MD_data['pos']*cm_to_AU


# Radial distribution function g_ei(r)
g_ei = interp1d(r_data, gei_data, fill_value='extrapolate', bounds_error=False)

# Constants
L_simulation = 4.360647e-07*cm_to_AU
L = L_simulation/3 # Define smaller box

approx_dx = 0.2
num_points = int(L/approx_dx)


# Ion positions: Nx3 matrix
# Example positions, replace with your actual data
ion_positions = MD_array 

# Create a 3D grid
x = np.linspace(0, L, num_points)
y = np.linspace(0, L, num_points)
z = np.linspace(0, L, num_points)
xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')

# Compute the electron density at each grid point
electron_density = np.ones((num_points, num_points, num_points))

def min_image_dist(p1, p2, L):
    """ Calculate the minimum image distance accounting for periodic boundary conditions """
    return np.abs(p1 - p2) - L * np.round(np.abs(p1 - p2) / L)

new_ion_MD_array = []
for ion in ion_positions:
    if np.all(ion < L):
        # Calculate distances from this ion to all points in the grid considering periodic boundary conditions
        grid_points = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
        min_image_distances = np.array([min_image_dist(grid_points[:, i], ion[i], L_simulation) for i in range(3)]).T
        distances = np.linalg.norm(min_image_distances, axis=1)
        densities = g_ei(distances).reshape((num_points, num_points, num_points))
        
        # Sum the contributions of this ion's g_ei to the electron density
        electron_density *= densities
        new_ion_MD_array.append(ion)

new_ion_MD_array = np.array(new_ion_MD_array)
# For visualization, you can use your preferred plotting code similar to previous descriptions
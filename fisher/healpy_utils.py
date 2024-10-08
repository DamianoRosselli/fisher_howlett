#%% HEALPY STUFF
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 11:24:58 2024

@author: Laurent MAGRI-STELLA & DAMIANO ROSSELLI

C to Python translation of PV_fisher code by Cullan HOWLETT
"""


import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def get_map(path):
    """
    Load observation maps from specified file paths.

    Args:
        path (str): Path to the directory containing the map files.
        names (list of str): List of names of the map files (without extension).

    Returns:
        dict: A dictionary where keys are the names and values are the loaded maps.
    """

    return np.load(path)


def total_area(obsmap):
    """
    Calculate the total area of an observation map in steradians.

    Args:
        obsmap (numpy.ndarray): Observation map array.

    Returns:
        float: Total area of the map in steradians.
    """
    # Get the number of pixels in the map
    npx = obsmap.size
    # Get the nside parameter of the map
    nside = hp.get_nside(obsmap)
    # Calculate the area of each pixel
    surf_px = hp.nside2pixarea(nside)
    # Total area is the number of pixels times the area of each pixel
    surf_map = npx * surf_px
    return surf_map

def mollview_map(maps, name, i_fig):
    """
    Display a map using a Mollweide projection.

    Args:
        maps (dict): Dictionary of maps to display.
        name (str): Name of the map to display.
        i_fig (int): Figure number.

    Returns:
        None
    """
    plt.figure(i_fig, figsize=(8,6))
    # Display the map using Healpy's mollview function
    hp.mollview(maps[name], hold=True, nest=True, fig=i_fig, title=name, cmap='gray_r')
    hp.graticule()

def combine_mollview(list_maps):
    """
    Combine multiple maps using the maximum value at each pixel.

    Args:
        list_maps (list of numpy.ndarray): List of maps to combine.

    Returns:
        numpy.ndarray: Combined map.
    """
    # Combine the maps by taking the maximum value at each pixel
    return np.maximum.reduce(list_maps)

def get_map_area(path):
    """
    Calculate the area of an observation map with non-zero pixels.

    Args:
        obs_map (numpy.ndarray): Observation map array.

    Returns:
        float: Area of the map with non-zero pixels in steradians.
    """
    
    obs_map = get_map(path)
    
    # Count the number of non-zero pixels in the map
    n1side = np.count_nonzero(obs_map)
    # Get the nside parameter of the map
    nside = hp.get_nside(obs_map)
    # Calculate the area of each pixel
    surf_px = hp.nside2pixarea(nside)
    # Total area is the number of non-zero pixels times the area of each pixel
    surf_map = n1side * surf_px
    return surf_map

def combine_map_area(list_maps):
    """
    Combine multiple maps and calculate the area of the combined map with non-zero pixels.

    Args:
        list_maps (list of numpy.ndarray): List of maps to combine.

    Returns:
        float: Area of the combined map with non-zero pixels in steradians.
    """
    # Combine the maps using the maximum value at each pixel
    sum_maps = combine_mollview(list_maps)
    # Calculate the area of the combined map with non-zero pixels
    surf_maps = get_map_area(sum_maps)
    return surf_maps

def sr2_to_deg2(sr2):
    """
    Convert an area from steradians to square degrees.

    Args:
        sr2 (float): Area in steradians.

    Returns:
        float: Area in square degrees.
    """
    return sr2 * (180 / np.pi)**2

def deg2_to_sr2(deg2):
    return deg2 / (180 / np.pi)**2

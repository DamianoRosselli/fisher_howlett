#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 11:24:58 2024

@author: Laurent MAGRI-STELLA & DAMIANO ROSSELLI

C to Python translation of PV_fisher code by Cullan HOWLETT
"""

import pickle
import scipy as sc
import numpy as np




def read_power_spectra_files(PV_file):
    """Read power spectra from a pickle file.

    The structure of the file should be:
    - entry key: z (the redshift)
    - Each redshift entry contains arrays for:
        - k (wave number)
        - Pmm (matter density power spectrum)
        - Pmt (cross power spectrum density-divergence)
        - Ptt (divergence power spectrum)
    The number of redshift entries should match the number of bins for the Fisher forecast.
    """

    # Open the pickle file and load its content
    with open(PV_file, 'rb') as f:
        x = pickle.load(f)

    # Get the keys (redshifts) from the loaded dictionary
    key = list(x.keys())

    # Return the loaded dictionary and the k values corresponding to the first redshift
    return x, x[key[0]]['k']

def read_nz_files(files, density_unit):
    """Read number density files and convert to the required unit.

    The function reads two files:
    - The first file contains redshift and number density for velocities.
    - The second file contains redshift and number density for redshifts.

    Args:
    - files: List of two file paths.
    - density_unit: Multiplicative factor to convert densities to the required unit.

    Returns:
    - zv: Redshifts for velocities.
    - zr: Redshifts for redshifts.
    - n_vel: Number densities for velocities, converted to the required unit.
    - n_red: Number densities for redshifts, converted to the required unit.
    """

    # Read the first file and unpack its columns into zv and nv
    zv, nv = np.genfromtxt(files[0], unpack=True)
    # Read the second file and unpack its columns into zr and nr
    zr, nr = np.genfromtxt(files[1], unpack=True)

    # Convert the number densities to the required unit
    n_vel = nv * density_unit[0] # velocity tracers
    n_red = nr * density_unit[1] # galaxy tracers
    
    # Return the redshifts and the converted number densities
    return zv, zr, n_vel, n_red

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
    """ read pickle file for power spectra, the structure of the file should be 
    entry key: z (the redshift) and for each redshift is required array of k, Pmm (matter density power spectrum),
    Pmt (cross power spectrum density-divergence) and Ptt (divergence power spectrum)
    the number of z should equal to the number of bin  that  you want in the fisher forecast """

    with open(PV_file, 'rb') as f:
        x = pickle.load(f)

    key = list(x.keys())

    return x , x[key[0]]['k']


def read_nz_files(files, density_unit):
    """ """
    
    zv , nv = np.genfromtxt(files[0], unpack=True)
    zr , nr = np.genfromtxt(files[1], unpack=True)

    n_vel = nv * density_unit
    n_red = nr * density_unit
    
    return zv, zr, sc.interpolate.InterpolatedUnivariateSpline(zv, n_vel), sc.interpolate.InterpolatedUnivariateSpline(zr, n_red)



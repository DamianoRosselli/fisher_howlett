#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 11:24:58 2024

@author: Laurent MAGRI-STELLA & DAMIANO ROSSELLI

C to Python translation of PV_fisher code by Cullan HOWLETT
"""

from astropy import constants as cst
import scipy as sc
import numpy as np



C_LIGHT_KMS = cst.c.to('km/s').value

def expansion_z(z, Om, Od, w0, wa):
    """
    Compute the expansion rate as a function of redshift.

    Args:
        z (float or array-like): Redshift values.
        Om (float): Matter density parameter.
        Od (float): Dark energy density parameter.
        w0 (float): Dark energy equation of state parameter at z = 0.
        wa (float): Dark energy equation of state parameter slope.

    Returns:
        np.ndarray: Expansion rate values corresponding to the input redshift values.
    """
    matter = Om * (1 + z)**3
    dark = Od * (1 + z)**(3 * (1 + w0 + wa)) * np.exp(-3 * wa * z / (1 + z)) 
    E2 = matter + dark
    return np.sqrt(E2)

def ezinv(z, Om, Od, w0, wa):
    """
    Calculate the inverse of the E(z) function.

    Args:
        z (float): Redshift.
        Om (float): Matter density parameter.
        Od (float): Dark energy density parameter.
        w0 (float): Dark energy equation of state parameter at z = 0.
        wa (float): Dark energy equation of state parameter slope.

    Returns:
        float: The inverse of E(z).
    """
    return 1.0 / expansion_z(z, Om, Od, w0, wa)



def rz_func(z, Om, Od, w0, wa):
    """
    Calculate the comoving distance to redshift z.

    Args:
        z (float): Redshift.
        Om (float): Matter density parameter.
        Od (float): Dark energy density parameter.
        w0 (float): Dark energy equation of state parameter at z = 0.
        wa (float): Dark energy equation of state parameter slope.

    Returns:
        float: Comoving distance to redshift z in units of Mpc/h.
    """
    result = sc.integrate.quad(ezinv, 0., z, args=(Om, Od, w0, wa), limit=1000)[0]
    return C_LIGHT_KMS * result / 100.0



def growthfunc(x, Om, Od, w0, wa, gammaval):
    """
    Calculate the growth rate f as a function of scale factor x.

    Args:
        x (float): Scale factor, where x = 1 / (1 + z).
        Om (float): Matter density parameter.
        gammaval (float): Growth index.

    Returns:
        float: Growth rate f divided by the scale factor x.
    """
    red = (1.0 / x) - 1.0
    Omz = Om * ezinv(red, Om, Od, w0, wa)**2 / (x ** 3)
    f = Omz**(gammaval)
    return f / x


def growthz_func(red, Om, Od, w0, wa, gammaval):
    """
    Calculate the growth factor as a function of redshift.

    Args:
        red (float): Redshift.
        Om (float): Matter density parameter.
        Od (float): Dark energy density parameter.
        w0 (float): Dark energy equation of state parameter at z = 0.
        wa (float): Dark energy equation of state parameter slope.
        gammaval (float): Growth index.

    Returns:
        float: Growth factor as a function of redshift.
    """
    a = 1.0 / (1.0 + red)
    result = sc.integrate.quad(growthfunc, a, 1., args=(Om, Od, w0, wa, gammaval), limit=1000)[0]
    return np.exp(-result)

growthz = np.vectorize(growthz_func)
rz = np.vectorize(rz_func)


def compute_r_spline(Om, Od, w0, wa):
    """
    Compute a spline for the comoving distance as a function of redshift.

    Args:
        cosmo_params (dict): Dictionary containing cosmological parameters.

    Returns:
        InterpolatedUnivariateSpline: Spline function for comoving distance.
    """
    ztemp = np.linspace(0., 1., 100)
    rtemp = rz(ztemp, Om, Od, w0, wa)
    r_spline = sc.interpolate.InterpolatedUnivariateSpline(ztemp, rtemp)
    return r_spline


def compute_growth_spline(Om, Od, w0, wa, gammaval):
    """
    Compute a spline for the growth factor as a function of redshift.

    Args:
        cosmo_params (dict): Dictionary containing cosmological parameters.

    Returns:
        InterpolatedUnivariateSpline: Spline function for growth factor.
    """
    ztemp = np.linspace(0., 1., 100)
    gg = growthz(ztemp, Om, Od, w0, wa, gammaval) / growthz(0., Om, Od, w0, wa, gammaval)
    g_spline = sc.interpolate.InterpolatedUnivariateSpline(ztemp, gg)
    return g_spline


def compute_Pz_spline(P_dic):
    """
    Compute splines for power spectra as a function of redshift.

    Args:
        P_dic (dict): Dictionary containing power spectra data.

    Returns:
        tuple: Arrays of splines for Pmm, Pmt, and Ptt power spectra.
    """
    keys = list(P_dic.keys())
    z_pk = keys

    if len(z_pk) > 1:
        Pmm_spline = []
        Pmt_spline = []
        Ptt_spline = []

        for i, kk in enumerate(P_dic[keys[0]]['k']):
            Pmm_array = np.asarray([P_dic[z]['Pmm'][i] for z in z_pk])
            Pmt_array = np.asarray([P_dic[z]['Pmt'][i] for z in z_pk])
            Ptt_array = np.asarray([P_dic[z]['Ptt'][i] for z in z_pk])

            Pmm_spline.append(sc.interpolate.InterpolatedUnivariateSpline(z_pk, Pmm_array))
            Pmt_spline.append(sc.interpolate.InterpolatedUnivariateSpline(z_pk, Pmt_array))
            Ptt_spline.append(sc.interpolate.InterpolatedUnivariateSpline(z_pk, Ptt_array))

        return np.asarray(Pmm_spline), np.asarray(Pmt_spline), np.asarray(Ptt_spline)
    else:
        raise ValueError('You need the power spectrum in at least 2 redshift bins')

        
def compute_dendamp(mu, sigma_g, k):
    """
    Compute the damping term for density.

    Args:
        mu (float): Cosine of the angle between the wavevector and the line of sight.
        sigma_g (float): Gaussian smoothing scale.
        k (float): Wavenumber.

    Returns:
        float: The density damping term.
    """
    # Calculate the damping term based on the provided formula
    return np.sqrt(1.0 / (1.0 + 0.5 * (k * k * mu * mu * sigma_g * sigma_g)))


def compute_veldamp(mu, sigma_u, k):
    """
    Compute the damping term for velocity.

    Args:
        mu (float): Cosine of the angle between the wavevector and the line of sight.
        sigma_u (float): Velocity dispersion scale.
        k (float): Wavenumber.

    Returns:
        float: The velocity damping term.
    """
    # Calculate the velocity damping term using the sinc function
    return np.sinc(k * sigma_u / np.pi)


def compute_f_beta(zz, cosmo_params):
    """
    Compute the growth rate (f), redshift-space distortion parameter (beta), and matter density parameter (Omz) at a given redshift.

    Args:
        zz (float): Redshift.
        cosmo_params (dict): Dictionary containing cosmological parameters.

    Returns:
        tuple: A tuple containing:
            - f (float): Growth rate.
            - beta (float): Redshift-space distortion parameter.
            - Omz (float): Matter density parameter at redshift zz.
    """
    Om = cosmo_params['Om_0']
    Od = cosmo_params['Od_0']
    w0 = cosmo_params['w0']
    wa = cosmo_params['wa']
    gammaval = cosmo_params['gammaval']
    beta_0 = cosmo_params['beta_0']
    
    # Compute the matter density parameter at redshift zz   
    Omz = Om * ezinv(zz, Om, Od, w0, wa)**2 * (1.0 + zz)**3
    
    # Compute the growth rate using the gamma parameter
    f = Omz**(gammaval)
    
    # Compute the redshift-space distortion parameter
    beta = f * beta_0 * growthz(zz, Om, Od, w0, wa, gammaval) / (Om**(gammaval))

    return f, beta, Omz


def compute_prefac(mu, f, k, veldamp, dendamp, beta, cosmo_params):
    """
    Compute the prefactors for the power spectra.

    Args:
        mu (float): Cosine of the angle between the wavevector and the line of sight.
        f (float): Growth rate.
        k (float): Wavenumber.
        veldamp (float): Velocity damping term.
        dendamp (float): Density damping term.
        beta (float): Redshift-space distortion parameter.
        cosmo_params (dict): Cosmological parameters including r_g.

    Returns:
        tuple: A tuple containing:
            - vv_prefac (float): Prefactor for velocity-velocity power spectrum.
            - dd_prefac (float): Prefactor for density-density power spectrum.
            - dv_prefac (float): Prefactor for density-velocity power spectrum.
    """
    # Compute the prefactor for the velocity-velocity power spectrum
    vv_prefac = 100.0 * f * mu * veldamp / k
    
    # Compute the prefactor for the density-density power spectrum
    dd_prefac = ((1.0 / (beta**2)) + (2.0 * cosmo_params['r_g'] * mu**2 / beta) + mu**4) * (f**2 * dendamp**2)
    
    # Compute the prefactor for the density-velocity power spectrum
    dv_prefac = (cosmo_params['r_g'] / beta + mu**2) * (f * dendamp)

    return vv_prefac, dd_prefac, dv_prefac


def compute_f_growth(z_vals, cosmo_params):
    """
    Compute the growth rate and f_sigma8 as a function of redshift.

    Args:
        z_vals (array-like): Array of redshift values.
        cosmo_params (dict): Dictionary containing cosmological parameters.

    Returns:
        tuple: Arrays of growth rate f and f_sigma8 as a function of redshift.
    """
    f_sigma8_vals = np.zeros_like(z_vals)

    growth_spline = compute_growth_spline(cosmo_params)
    sigma8_0 = cosmo_params['sigma_8_0']

    for i, zz in enumerate(z_vals):
        f, beta, Omz = compute_f_beta(zz, cosmo_params)
        growth = growth_spline(zz)
        f_sigma8_vals[i] = f * sigma8_0 * growth

    return f_sigma8_vals

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 11:24:58 2024

@author: Laurent MAGRI-STELLA & DAMIANO ROSSELLI


"""


from astropy import constants as cst
from . import utils as ut
import scipy as sc
import numpy as np 


C_LIGHT_KMS = cst.c.to('km/s').value



def zeff(mu, cosmo_params, k, zz, Pmm, Ptt, survey, n_vel, n_red, errors, r, f, beta):
    """
    Calculate the effective redshift.

    Args:
        mu (float): Cosine of the angle between the line of sight and the wavevector.
        cosmo_params (dict): Cosmological parameters.
        k (array): Wavevector magnitude.
        zz (array): Redshift array.
        Pmm (array): Matter power spectrum.
        Ptt (array): Velocity divergence power spectrum.
        survey (list): Survey parameters.
        n_vel (array): Velocity tracer density.
        n_red (array): Redshift tracer density.
        errors (dict): Error parameters.
        r (float): Comoving distance.
        f (float): Growth rate.
        beta (float): Redshift space distortion parameter.

    Returns:
        array: Effective redshift values.
    """

    # Compute the damping factors
    dendamp = ut.compute_dendamp(mu, cosmo_params['sigma_g'], k)  # Density damping
    veldamp = ut.compute_veldamp(mu, cosmo_params['sigma_u'], k)  # Velocity damping

    # Compute prefactors for power spectra
    vv_prefac, dd_prefac, _ = ut.compute_prefac(mu, f, k, veldamp, dendamp, beta, cosmo_params)

    # Compute the galaxy power spectrum and velocity power spectrum
    P_gg = dd_prefac * Pmm
    P_uu = vv_prefac**2 * Ptt

    # Compute the effective number density of galaxies
    n_g = n_red / (1.0 + n_red * P_gg)

    # Compute the observational error in km/s
    error_obs = 100.0 * errors['dist'] * r
    error_noise = errors['rand']**2 + error_obs**2

    # Compute the effective number density of velocity tracers
    n_u = (n_vel / error_noise) / (1.0 + P_uu * (n_vel / error_noise))

    # Compute the effective redshift value
    val = (n_g * survey[0]) + (n_u * survey[1]) + (n_g**2 + n_u**2) * survey[2]

    return val


def zeff_integral(cosmo_params, k, zz, Pmm_spline, Ptt_spline, survey, n_vel, n_red, errors, r, f, beta):
    """
    Calculate the integral of the effective redshift over the relevant parameters.

    Args:
        cosmo_params (dict): Cosmological parameters.
        k (array): Wavevector magnitude.
        zz (array): Redshift array.
        Pmm_spline (array): Spline of the matter power spectrum.
        Ptt_spline (array): Spline of the velocity divergence power spectrum.
        survey (list): Survey parameters.
        n_vel (array): Velocity tracer density.
        n_red (array): Redshift tracer density.
        errors (dict): Error parameters.
        r (float): Comoving distance.
        f (float): Growth rate.
        beta (float): Redshift space distortion parameter.

    Returns:
        float: Integrated effective redshift value.
    """
    mu = np.linspace(0., 1., 50)
    K, MU, R = np.meshgrid(k, mu, r)
    zeff_val = zeff(MU, cosmo_params, K, zz, Pmm_spline, Ptt_spline, survey, n_vel, n_red, errors, R, f, beta)
    integral = np.trapz(zeff_val * zz * r**2, r, axis=2) / np.trapz(zeff_val * r**2, r, axis=2)
    # pose problème apparament
    return np.trapz(k**2 * np.trapz(integral, mu, axis=0), k, axis=0) / np.trapz(k**2, k)


def compute_derivatives(par, cosmo_params, beta, mu, f, dendamp, Pmm, Pmt, P_uu, P_gg, P_ug, vv_prefac, sigma8, k):
    """
    Compute the derivatives of the power spectra with respect to various parameters.

    Args:
        par (int): Parameter index to differentiate with respect to.
        cosmo_params (dict): Cosmological parameters.
        beta (float): Redshift space distortion parameter.
        mu (float): Cosine of the angle between the line of sight and the wavevector.
        f (float): Growth rate.
        dendamp (float): Density damping factor.
        Pmm (array): Matter power spectrum.
        Pmt (array): Cross power spectrum (matter-velocity).
        P_uu (array): Velocity power spectrum.
        P_gg (array): Galaxy power spectrum.
        P_ug (array): Cross power spectrum (velocity-galaxy).
        vv_prefac (float): Velocity prefactor.
        sigma8 (float): Amplitude of the matter power spectrum.
        k (array): Wavevector magnitude.

    Returns:
        array: Matrix of derivatives of the power spectra.
    """
    dPdt = np.zeros((2, 2), dtype=object)

    if par == 0:  # Differential w.r.t betaA
        dPdt[0, 0] = -2.0 * ((1.0 / beta) + cosmo_params['r_g'] * mu**2) * (f**2) * (dendamp**2) * Pmm / (beta**2)
        dPdt[0, 1] = -(vv_prefac * f * cosmo_params['r_g'] * dendamp * Pmt) / (beta**2)
        dPdt[1, 0] = -(vv_prefac * f * cosmo_params['r_g'] * dendamp * Pmt) / (beta**2)

    elif par == 1:  # Differential w.r.t fsigma8
        dPdt[0, 0] = 2.0 * (f / (beta**2) + 2.0 * f * cosmo_params['r_g'] * mu**2 / beta + f * mu**4) * (dendamp**2) * Pmm / sigma8
        dPdt[0, 1] = 2.0 * vv_prefac * (cosmo_params['r_g'] / beta + mu**2) * dendamp * Pmt / sigma8
        dPdt[1, 0] = 2.0 * vv_prefac * (cosmo_params['r_g'] / beta + mu**2) * dendamp * Pmt / sigma8
        dPdt[1, 1] = (2.0 * P_uu) / (f * sigma8)

    elif par == 2:  # Differential w.r.t r_g
        dPdt[0, 0] = 2.0 * (1.0 / beta) * mu**2 * f**2 * (dendamp**2) * Pmm
        dPdt[0, 1] = vv_prefac * (1.0 / beta) * f * dendamp * Pmt
        dPdt[1, 0] = vv_prefac * (1.0 / beta) * f * dendamp * Pmt

    elif par == 3:  # Differential w.r.t sigma_g
        dPdt[0, 0] = -k**2 * mu**2 * dendamp**2 * cosmo_params['sigma_g'] * P_gg
        dPdt[0, 1] = -0.5 * k**2 * mu**2 * dendamp**2 * cosmo_params['sigma_g'] * P_ug
        dPdt[1, 0] = -0.5 * k**2 * mu**2 * dendamp**2 * cosmo_params['sigma_g'] * P_ug

    else:  # Differential w.r.t sigma_u
        dPdt[0, 1] = P_ug * (k * np.cos(k * cosmo_params['sigma_u']) / np.sin(k * cosmo_params['sigma_u']) - 1.0 / cosmo_params['sigma_u'])
        dPdt[1, 0] = P_ug * (k * np.cos(k * cosmo_params['sigma_u']) / np.sin(k * cosmo_params['sigma_u']) - 1.0 / cosmo_params['sigma_u'])
        dPdt[1, 1] = 2.0 * P_uu * (k * np.cos(k * cosmo_params['sigma_u']) / np.sin(k * cosmo_params['sigma_u']) - 1.0 / cosmo_params['sigma_u'])

    return dPdt


def mu_integrand(mu, k, zz, Pmm, Pmt, Ptt, par1, par2,
                 cosmo_params, survey, n_vel, n_red, errors, r, f, beta, sigma8):
    """
    Compute the integrand for the Fisher matrix calculation.

    Args:
        mu (array): Cosine of the angle between the line of sight and the wavevector.
        k (array): Wavevector magnitude.
        zz (array): Redshift array.
        Pmm (array): Matter power spectrum.
        Pmt (array): Cross power spectrum (matter-velocity).
        Ptt (array): Velocity power spectrum.
        par1 (int): First parameter index for differentiation.
        par2 (int): Second parameter index for differentiation.
        cosmo_params (dict): Cosmological parameters.
        survey (list): Survey parameters.
        n_vel (float): Velocity tracer density.
        n_red (float): Redshift tracer density.
        errors (dict): Error parameters.
        r (float): Comoving distance.
        f (float): Growth rate.
        beta (float): Redshift space distortion parameter.
        sigma8 (float): Amplitude of the matter power spectrum.

    Returns:
        array: Integrand value for the Fisher matrix calculation.
    """
    
    # Compute the damping factors
    dendamp = ut.compute_dendamp(mu, cosmo_params['sigma_g'], k)  # Density damping factor, unitless
    veldamp = ut.compute_veldamp(mu, cosmo_params['sigma_u'], k)  # Velocity damping factor, unitless

    # Compute the prefactors for power spectra
    vv_prefac, dd_prefac, dv_prefac = ut.compute_prefac(mu, f, k, veldamp, dendamp, beta, cosmo_params)

    # Compute the power spectra
    P_gg = dd_prefac * Pmm  # Galaxy power spectrum
    P_ug = vv_prefac * dv_prefac * Pmt  # Cross power spectrum (velocity-galaxy)
    P_uu = vv_prefac**2 * Ptt  # Velocity power spectrum

    # Compute the derivatives of the power spectra with respect to the parameters of interest
    dPdt1 = compute_derivatives(par1, cosmo_params, beta, mu, f, dendamp, Pmm, Pmt, P_uu, P_gg, P_ug, vv_prefac, sigma8, k)
    dPdt2 = compute_derivatives(par2, cosmo_params, beta, mu, f, dendamp, Pmm, Pmt, P_uu, P_gg, P_ug, vv_prefac, sigma8, k)

    # Initialize the sum over survey components
    surv_sum = []
    for y, s in enumerate(survey):
        if s > 0:
            # Determine the effective number densities of galaxies and velocity tracers
            if y == 0:  # Redshift survey component
                n_g = n_red
                n_u = 0
            elif y == 1:  # Velocity survey component
                n_g = 0
                error_obs = 100.0 * errors['dist'] * r  # Observational error in km/s
                error_noise = errors['rand']**2 + error_obs**2  # Total error noise in km^2/s^2
                n_u = n_vel / error_noise
            else:  # Combined survey component
                n_g = n_red
                error_obs = 100.0 * errors['dist'] * r  # Observational error in km/s
                error_noise = errors['rand']**2 + error_obs**2  # Total error noise in km^2/s^2
                n_u = n_vel / error_noise

            # Compute the determinant of the Fisher information matrix
            det = 1.0 + n_u * n_g * (P_gg * P_uu - P_ug**2) + n_u * P_uu + n_g * P_gg

            # Compute the inverse of the Fisher information matrix
            iP = np.zeros((2, 2), dtype=object)
            iP[0, 0] = n_u * n_g * P_uu + n_g
            iP[1, 1] = n_g * n_u * P_gg + n_u
            iP[0, 1] = -n_g * n_u * P_ug
            iP[1, 0] = -n_g * n_u * P_ug

            # Compute the trace of the product of derivatives and the inverse Fisher matrix
            trace = np.trace(dPdt1 @ iP @ dPdt2 @ iP)

            # Append the contribution to the survey sum
            surv_sum.append((trace * s) / (det**2))

    # Sum over the survey components and multiply by r^2
    val = np.sum(np.asarray(surv_sum), axis=0) * (r**2)

    return val


def mu_integral(k, zz, Pmm_spline, Pmt_spline, Ptt_spline, par1, par2,
                cosmo_params, survey, n_vel, n_red, errors, r, f, beta, sigma8):
    """
    Compute the integral over mu for the Fisher matrix calculation.

    Args:
        k (array): Wavevector magnitude.
        zz (array): Redshift array.
        Pmm_spline (array): Spline of the matter power spectrum.
        Pmt_spline (array): Spline of the cross power spectrum (matter-velocity).
        Ptt_spline (array): Spline of the velocity power spectrum.
        par1 (int): First parameter index for differentiation.
        par2 (int): Second parameter index for differentiation.
        cosmo_params (dict): Cosmological parameters.
        survey (list): Survey parameters.
        n_vel (float): Velocity tracer density.
        n_red (float): Redshift tracer density.
        errors (dict): Error parameters.
        r (float): Comoving distance.
        f (float): Growth rate.
        beta (float): Redshift space distortion parameter.
        sigma8 (float): Amplitude of the matter power spectrum.

    Returns:
        float: Integrated value for the Fisher matrix calculation.
    """
    mu = np.linspace(0., 1., 100)  # Discretize mu
    K, MU, R = np.meshgrid(k, mu, r)  # Create a meshgrid for k, mu, and r
    mu_val = mu_integrand(MU, K, zz, Pmm_spline, Pmt_spline, Ptt_spline, par1, par2,
                          cosmo_params, survey, n_vel, n_red, errors, R, f, beta, sigma8)

    # Integrate over r, mu, and k
    return np.trapz(k**2 * np.trapz(np.trapz(mu_val, r, axis=2), mu, axis=0), k, axis=0)
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


def ezinv(x,Om):
    return 1.0 / np.sqrt(Om * (1.0 + x)**3 + (1.0 - Om))


def rz_func(z,Om):
    result = sc.integrate.quad(ezinv,0.,z,args=(Om),limit=1000)[0]
    return C_LIGHT_KMS * result / 100.0


def growthfunc(x,Om,gammaval):
    red = (1.0/x) - 1.0
    Omz = Om * ezinv(red,Om) ** 2 / (x **3)
    f = Omz**(gammaval)
    return f / x


def growthz_func(red,Om,gammaval):
    a = 1.0 / (1.0 + red)
    result = sc.integrate.quad(growthfunc,a,1.,args=(Om,gammaval),limit=1000)[0]
    return np.exp(-result)

growthz = np.vectorize(growthz_func)
rz = np.vectorize(rz_func)


def compute_r_spline(Om):
    ztemp = np.linspace(0., 1., 100)
    rtemp = rz(ztemp,Om)
    r_spline = sc.interpolate.InterpolatedUnivariateSpline(ztemp, rtemp)
    return r_spline


def compute_growth_spline(Om,gammaval):
    ztemp = np.linspace(0., 1., 100)
    gg = growthz(ztemp,Om,gammaval) / growthz(0., Om,gammaval)
    g_spline = sc.interpolate.InterpolatedUnivariateSpline (ztemp, gg)
    return g_spline


def compute_Pz_spline(P_dic,zbin):
    keys = list(P_dic.keys())

    if len(keys)>1:
        Pmm_spline = []
        Pmt_spline = []
        Ptt_spline = []
        for i,kk in enumerate(P_dic[keys[0]]['k']):

            Pmm_array = np.asarray([P_dic[key]['Pmm'][i] for key in keys])
            Pmt_array = np.asarray([P_dic[key]['Pmt'][i] for key in keys])
            Ptt_array = np.asarray([P_dic[key]['Ptt'][i] for key in keys])

            Pmm_spline.append(sc.interpolate.InterpolatedUnivariateSpline(zbin, Pmm_array))
            Pmt_spline.append(sc.interpolate.InterpolatedUnivariateSpline(zbin, Pmt_array))
            Ptt_spline.append(sc.interpolate.InterpolatedUnivariateSpline(zbin, Ptt_array))

        return np.asarray(Pmm_spline), np.asarray(Pmt_spline), np.asarray(Ptt_spline)

    else:
        raise ValueError('You need the power spectrum in at least 2 redshift bin')


def compute_dendamp(mu,sigma_g,k):
    return np.sqrt(1.0 / (1.0 + 0.5 * (k * k * mu * mu * sigma_g * sigma_g)))


def compute_veldamp(mu,sigma_u,k):
    return np.sinc(k*sigma_u/np.pi)


def compute_f_beta(zz,cosmo_params):

    Omz = cosmo_params['Om_0']*ezinv(zz,cosmo_params['Om_0'])**2 *(1.0+zz)**3
    f = Omz**(cosmo_params['gammaval'])
    beta = f*cosmo_params['beta_0'] * growthz(zz,cosmo_params['Om_0'],cosmo_params['gammaval'])/(cosmo_params['Om_0']**(cosmo_params['gammaval']))

    return f,beta,Omz


def compute_prefac(mu,f,k,veldamp,dendamp,beta,cosmo_params):
    
    vv_prefac  = 100.*f*mu*veldamp/k
    dd_prefac = ((1.0/(beta**2)) + (2.0*cosmo_params['r_g']*mu**2/beta) + mu**4)*(f**2*dendamp**2)
    dv_prefac = (cosmo_params['r_g']/beta + mu**2)*(f*dendamp)

    return vv_prefac, dd_prefac, dv_prefac



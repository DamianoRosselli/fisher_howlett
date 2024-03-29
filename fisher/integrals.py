#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 11:24:58 2024

@author: Laurent MAGRI-STELLA & DAMIANO ROSSELLI

C to Python translation of PV_fisher code by Cullan HOWLETT
"""


from astropy import constants as cst
from . import utils as ut
import scipy as sc
import numpy as np 


C_LIGHT_KMS = cst.c.to('km/s').value



def zeff(mu,cosmo_params,k,zz,Pmm,Ptt,survey, n_vel, n_red, errors, r,f,beta):

    # compute the power spectra prefactor
    dendamp = ut.compute_dendamp(mu, cosmo_params['sigma_g'],k)     # This is unitless
    veldamp = ut.compute_veldamp(mu,cosmo_params['sigma_u'],k)        # This is unitless

    dVeff =[]
    zdVeff = []

    #for i,z in enumerate(zz):
    vv_prefac,dd_prefac,_ = ut.compute_prefac(mu,f,k,veldamp,dendamp,beta,cosmo_params)

    P_gg = dd_prefac * Pmm
    P_uu = vv_prefac**2 * Ptt
    
    n_g = n_red/(1. + n_red*P_gg)

    error_obs = 100.0*errors['dist']*r    # Percentage error * distance * H0 in km/s (factor of 100.0 comes from hubble parameter)
    error_noise = errors['rand']**2 + error_obs**2   # Error_noise is in km^{2}s^{-2}
    n_u = (n_vel/error_noise) / (1. + P_uu * (n_vel/error_noise) )

    val = (n_g*survey[0])+(n_u*survey[1])+(n_g**2+n_u**2)*survey[2]
    
    return val


def zeff_integral(cosmo_params,k,zz,Pmm_spline,Ptt_spline,survey, n_vel, n_red, errors,r,f,beta):
    mu = np.linspace(0.,1.,50)
    K,MU,R = np.meshgrid(k,mu,r)
    zeff_val = zeff(MU,cosmo_params,K,zz,Pmm_spline,Ptt_spline,survey, n_vel, n_red, errors, R,f,beta)
    integral = np.trapz(zeff_val*zz*r**2, r, axis=2)/np.trapz(zeff_val*r**2, r, axis=2)
    return np.trapz(k**2*np.trapz(integral,mu , axis=0),k,axis=0)/np.trapz(k**2,k)
    



def compute_derivatives(par,cosmo_params,beta,mu,f,dendamp,Pmm,Pmt,P_uu,P_gg,P_ug,vv_prefac,sigma8,k):

    dPdt = np.zeros((2,2),dtype=object)

    if par == 0:  # Differential w.r.t betaA
        dPdt[0, 0] = -2.0*((1.0/beta) + cosmo_params['r_g']*mu**2)*(f**2)*(dendamp**2)*Pmm/(beta**2)
        dPdt[0, 1] = -(vv_prefac*f*cosmo_params['r_g']*dendamp*Pmt)/(beta**2)
        dPdt[1, 0] = -(vv_prefac*f*cosmo_params['r_g']*dendamp*Pmt)/(beta**2)

    elif par == 1:  # Differential w.r.t fsigma8
        dPdt[0, 0] = 2.0*(f/(beta**2) + 2.0*f*cosmo_params['r_g']*mu**2/beta + f*mu**4)*(dendamp**2)*Pmm/sigma8
        dPdt[0, 1] = 2.0*vv_prefac*(cosmo_params['r_g']/beta + mu**2)*dendamp*Pmt/sigma8
        dPdt[1, 0] = 2.0*vv_prefac*(cosmo_params['r_g']/beta + mu**2)*dendamp*Pmt/sigma8
        dPdt[1, 1] = (2.0*P_uu)/(f*sigma8)

    elif par == 2:  # Differential w.r.t r_g
        dPdt[0, 0] = 2.0*(1.0/beta)*mu**2*f**2*(dendamp**2)*Pmm
        dPdt[0, 1] = vv_prefac*(1.0/beta)*f*dendamp*Pmt
        dPdt[1, 0] = vv_prefac*(1.0/beta)*f*dendamp*Pmt
    elif par == 3:  # Differential w.r.t sigma_g
        dPdt[0, 0] = -k**2*mu**2*dendamp**2*cosmo_params['sigma_g']*P_gg
        dPdt[0, 1] = -0.5*k**2*mu**2*dendamp**2*cosmo_params['sigma_g']*P_ug
        dPdt[1, 0] = -0.5*k**2*mu**2*dendamp**2*cosmo_params['sigma_g']*P_ug

    else:  # Differential w.r.t sigma_u
        dPdt[0, 1] = P_ug*(k*np.cos(k*cosmo_params['sigma_u'])/np.sin(k*cosmo_params['sigma_u']) - 1.0/cosmo_params['sigma_u'])
        dPdt[1, 0] = P_ug*(k*np.cos(k*cosmo_params['sigma_u'])/np.sin(k*cosmo_params['sigma_u']) - 1.0/cosmo_params['sigma_u'])
        dPdt[1, 1] = 2.0*P_uu*(k*np.cos(k*cosmo_params['sigma_u'])/np.sin(k*cosmo_params['sigma_u']) - 1.0/cosmo_params['sigma_u'])

    return dPdt




def mu_integrand(mu,k,zz,Pmm,Pmt,Ptt,par1,par2,
                cosmo_params,survey, n_vel, n_red, errors, r,f,beta,sigma8):
    
    # compute the power spectra prefactor
    dendamp = ut.compute_dendamp(mu, cosmo_params['sigma_g'],k)     # This is unitless
    veldamp = ut.compute_veldamp(mu,cosmo_params['sigma_u'],k)        # This is unitless

    
    vv_prefac,dd_prefac,dv_prefac = ut.compute_prefac(mu,f,k,veldamp,dendamp,beta,cosmo_params)
  

    P_gg = dd_prefac*Pmm
    P_ug = vv_prefac*dv_prefac*Pmt
    P_uu = vv_prefac**2*Ptt

    # And now the derivatives. Need to create a matrix of derivatives for each of the two parameters of interest
        

    dPdt1 = compute_derivatives(par1,cosmo_params,beta,mu,f,dendamp,Pmm,Pmt,P_uu,P_gg,P_ug,vv_prefac,sigma8,k)
    dPdt2 = compute_derivatives(par2,cosmo_params,beta,mu,f,dendamp,Pmm,Pmt,P_uu,P_gg,P_ug,vv_prefac,sigma8,k)
        
    surv_sum = []
    for y,s in enumerate(survey):
        if s > 0:
            if y == 0:
                n_g = n_red
                n_u = 0
            elif y == 1:
                n_g = 0
                error_obs = 100.0*errors['dist']*r   # Percentage error * distance * H0 in km/s (factor of 100.0 comes from hubble parameter)
                error_noise = errors['rand']**2 + error_obs**2   # Error_noise is in km^{2}s^{-2}
                n_u = n_vel/error_noise 
            else:
                n_g = n_red
                error_obs = 100.0*errors['dist']*r   # Percentage error * distance * H0 in km/s (factor of 100.0 comes from hubble parameter)
                error_noise = errors['rand']**2 + error_obs**2   # Error_noise is in km^{2}s^{-2}
                n_u = n_vel/error_noise 


            # First we need the determinant.
            det = 1.0 + n_u * n_g * (P_gg * P_uu - P_ug ** 2) + n_u * P_uu + n_g * P_gg

            # Now the inverse matrix.
            iP =  np.zeros((2,2),dtype=object)
            iP[0, 0] = n_u * n_g * P_uu + n_g
            iP[1, 1] = n_g * n_u * P_gg + n_u
            iP[0, 1] = - n_g * n_u * P_ug
            iP[1, 0] = - n_g * n_u * P_ug

            trace = np.trace(dPdt1 @ iP @ dPdt2 @ iP) 
            
            surv_sum.append((trace*s)/(det**2))

    val = np.sum(np.asarray(surv_sum),axis=0) * (r**2)
   
    return val
      
    



def mu_integral(k,zz,Pmm_spline,Pmt_spline,Ptt_spline,par1,par2,
                cosmo_params,survey, n_vel, n_red, errors, r,f,beta,sigma8):
    mu = np.linspace(0.,1.,100)
    K,MU,R = np.meshgrid(k,mu,r)
    mu_val = mu_integrand(MU,K,zz,Pmm_spline,Pmt_spline,Ptt_spline,par1,par2,
                          cosmo_params,survey, n_vel, n_red, errors, R,f,beta,sigma8)
    
    return np.trapz(k**2 * np.trapz(np.trapz(mu_val,r,axis=2), mu, axis=0), k, axis=0)
    


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  8 11:24:58 2024

@author: Laurent MAGRI-STELLA & DAMIANO ROSSELLI

C to Python translation of PV_fisher code by Cullan HOWLETT
"""


import scipy as sc
import os
import abc
import yaml
from astropy import constants as cst
from . import utils as ut
from . import read_file as rfi 
from . import integrals
import numpy as np
from tqdm import tqdm

C_LIGHT_KMS = cst.c.to('km/s').value



class Fisher(abc.ABC):

    """ generic class to handel PV fisher matrix forecast and parameters
    the parametes needed are explained in params.dat
    see also examples """

    def __init__(self, param_dic):
        """Initialise  class."""
        # Load param dict from a yaml file
        if isinstance(param_dic, dict):
            self._config = param_dic
            if 'yaml_path' in param_dic:
                self._yml_path = param_dic['yaml_path']
            else:
                self._yml_path = 'No config file'

        elif isinstance(param_dic, str):
            self._yml_path = param_dic
            with open(self._yml_path, "r") as f:
                self._config = yaml.safe_load(f)

        
        
        self._Data = self._config['Data']  
        self._nparams = len(self._Data)
       
        self._zmin , self._zmax, self._nzbin = self._config['zrange']
        
        self._cosmo_params = self._config['cosmo_params']  
                 
        self._survey_area = self._config['survey_area']  

        # Run some checks
    
        if (self._survey_area[0] == 0.) and (self._survey_area[2] == 0.):
            if 2 in self._Data:
                raise ValueError("ERROR: r_g is a free parameter, but there is no information in the density field (Fisher matrix will be singular)")
            
            if 3 in self._Data:
                raise ValueError("ERROR: sigma_g is a free parameter, but there is no information in the density field (Fisher matrix will be singular)")
            
        if (self._survey_area[1] == 0.) and (self._survey_area[2] == 0.):
            if 4 in self._Data:
                raise ValueError("ERROR: sigma_u is a free parameter, but there is no information in the velocity field (Fisher matrix will be singular)")


        self._errors = self._config['errors']    
        

        self._verbosity = self._config['verbosity']         

        
        self._Pv, self._k = rfi.read_power_spectra_files(self._config['Pvel_file'])   
        

        #run some checks
        for kk in self._Pv.keys():
            if self._cosmo_params['k_max'] > np.max(self._Pv[kk]['k']):
                raise ValueError("ERROR: The maximum k in the input power spectra is less than k_max")

        #spline on nbar
        self._z_vel,self._z_red,self._nbar_vel, self._nbar_red = rfi.read_nz_files( self._config['nbar_file'] ,  self._config['density_unity'])

        #compute some spline for later 
        self._r_spline = ut.compute_r_spline(self._cosmo_params['Om_0'])
        self._growth_spline = ut.compute_growth_spline(self._cosmo_params['Om_0'],self._cosmo_params['gammaval'])
        self._Pmm_spline, self._Pmt_spline, self._Ptt_spline = ut.compute_Pz_spline(self._Pv, np.linspace(self._zmin , self._zmax, self._nzbin))

        self._Fisher_in_bin = None
        self._data = None
        self._Fisher_tot = None
        self._data_tot = None



    def compute_fisher_zbin(self):

        print(f"Evaluating the Fisher Matrix for {self._nzbin -1 } bins between [z_min = {self._zmin}, z_max = {self._zmax}]")

        zbin = np.linspace(self._zmin , self._zmax, self._nzbin)
            
        fisher_tot =[]
        data_tot = {}
        
        for i in tqdm(range(len(zbin)-1)):
            
            rzmax = self._r_spline(zbin[i+1])
            kmin = np.pi/rzmax

            if self._verbosity:
                print(f"Evaluating the Fisher Matrix for [k_min = {kmin}, k_max = {self._cosmo_params['k_max']}] and [z_min = {zbin[i]}, z_max = {zbin[i+1]}]")

            for kk in self._Pv.keys():
                if kmin < np.min(self._Pv[kk]['k']):
                    raise ValueError("ERROR: The minimum k in the input power spectra is more than k_min ")

           
            idk = (self._k > kmin)&(self._k<self._cosmo_params['k_max'])
            zz = np.linspace(zbin[i],zbin[i+1],50)

            sigma_8 = self._cosmo_params['sigma_8_0'] * ut.growthz(zz,self._cosmo_params['Om_0'],self._cosmo_params['gammaval'])
            f, beta,_ = ut.compute_f_beta(zz,self._cosmo_params)

            r = self._r_spline(zz)
             #compute Power spectra at the redshift of interest
            Pmm = np.asarray([spline(zz) for spline in self._Pmm_spline[idk]])
            Ptt = np.asarray([spline(zz) for spline in self._Ptt_spline[idk]])
            Pmt = np.asarray([spline(zz) for spline in self._Pmt_spline[idk]])

            nbar_vel = self._nbar_vel(zz)
            nbar_red = self._nbar_red(zz)
           
            z_eff = integrals.zeff_integral(self._cosmo_params,self._k[idk],zz,
                                          Pmm,Ptt,
                                          self._survey_area, nbar_vel , nbar_red, self._errors,r,
                                          f,beta)
            
            data_tot[i] = {}
            data_tot[i]['zbin'] = [zbin[i],zbin[i+1]]
            data_tot[i]['zeff'] = z_eff

            Fisher = np.zeros((self._nparams, self._nparams))
            
            for t in range(self._nparams):
                for j in range(t, self._nparams):
                   
                    k_sum = integrals.mu_integral(self._k[idk],zz,
                                                Pmm, Pmt, Ptt,
                                                self._Data[t], self._Data[j],self._cosmo_params,
                                                self._survey_area,
                                                nbar_vel , nbar_red, self._errors,
                                                r,
                                                f,beta,sigma_8)

                    Fisher[t, j] = k_sum/(4.0*np.pi)
                    Fisher[j, t] = k_sum/(4.0*np.pi)

    
            fisher_tot.append(Fisher)

            #print some results on the terminal
            if self._verbosity:
                print("Fisher Matrix \n ====================== \n")
                for row in Fisher:
                    print(row , '\n') 
                print(" ====================== \n")
                    
           
            Covariance = sc.linalg.inv(Fisher)
                    
            growth_eff = self._growth_spline(z_eff)  
                        
            sigma8_eff = self._cosmo_params['sigma_8_0'] * growth_eff
            f_eff,_,_ = ut.compute_f_beta(z_eff,self._cosmo_params)
            beta_eff = f_eff*self._cosmo_params['beta_0']*growth_eff/(self._cosmo_params['Om_0']**(self._cosmo_params['gammaval']))

            #save the data
            for w in range(self._nparams):
                if self._Data[w] == 0 :
                    data_tot[i]['beta_eff'] = beta_eff
                    data_tot[i]['beta_eff_err'] = np.sqrt(Covariance[w, w])
                   
                if self._Data[w] == 1 :
                    data_tot[i]['fs8_eff'] = f_eff*sigma8_eff
                    data_tot[i]['fs8_eff_err'] = np.sqrt(Covariance[w, w])
                    
                if self._Data[w] == 2 :
                    data_tot[i]['r_g'] = self._cosmo_params['rg']
                    data_tot[i]['r_g_err'] = np.sqrt(Covariance[w, w])
                    
                if self._Data[w] == 3 :
                    data_tot[i]['sigma_g'] = self._cosmo_params['sigma_g']
                    data_tot[i]['sigma_g_err'] = np.sqrt(Covariance[w, w])
                    
                if self._Data[w] == 3 :
                    data_tot[i]['sigma_u'] = self._cosmo_params['sigma_u']
                    data_tot[i]['sigma_u_err'] = np.sqrt(Covariance[w, w])

            #print some results on the terminal
            if self._verbosity:
                print(f"Effective redshift z_eff = {z_eff}")
                for w in range(self._nparams):
                    if self._Data[w] == 0 :
                        print("beta(z_eff) \t percentage error(z_eff)")
                        print(f'{beta_eff:.6f} \t {100.0*(np.sqrt(Covariance[w, w])/(beta_eff)):.6f}')

                    if self._Data[w] == 1 :
                        print("fsigma8(z_eff) \t percentage error(z_eff)")
                        print(f'{f_eff*sigma8_eff:.6f} \t {100.0*(np.sqrt(Covariance[w, w])/(f_eff*sigma8_eff)):.6f}')

                    if self._Data[w] == 2 :
                        print("r_g \t percentage error")
                        rrg = self._cosmo_params['r_g']
                        print(f'{rg:.6f} \t {100.0*(np.sqrt(Covariance[w, w])/(rg)):.6f}')

                    if self._Data[w] == 3 :
                        print("sigma_g \t percentage error")
                        ssg = self._cosmo_params['sigma_g']
                        print(f'{ssg:.6f} \t {100.0*(np.sqrt(Covariance[w, w])/(ssg)):.6f}')

                    if self._Data[w] == 3 :
                        print("sigma_u \t percentage error")
                        ssu =self._cosmo_params['sigma_u']
                        print(f'{ssu:.6f} \t {100.0*(np.sqrt(Covariance[w, w])/(ssu)):.6f}')
          
          
        print(f"Evaluatiion the Fisher Matrix for {self._nzbin -1 } bins between [z_min = {self._zmin}, z_max = {self._zmax}] is COMPLETE !!")

        self._Fisher_in_bin = np.asarray(fisher_tot)
        self._data = data_tot




    def compute_fisher_tot(self):

        print(f"Evaluating the Fisher Matrix between [z_min = {self._zmin}, z_max = {self._zmax}]")

        rzmax = self._r_spline(self._zmax)
        kmin = np.pi/rzmax

        for kk in self._Pv.keys():
            if kmin < np.min(self._Pv[kk]['k']):
                raise ValueError("ERROR: The minimum k in the input power spectra is more than k_min ")

           
        idk = (self._k > kmin)&(self._k<self._cosmo_params['k_max'])
        zz = np.linspace(self._zmin,self._zmax,100)

        sigma_8 = self._cosmo_params['sigma_8_0'] * ut.growthz(zz,self._cosmo_params['Om_0'],self._cosmo_params['gammaval'])
        f, beta,_ = ut.compute_f_beta(zz,self._cosmo_params)

        r = self._r_spline(zz)
             #compute Power spectra at the redshift of interest
        Pmm = np.asarray([spline(zz) for spline in self._Pmm_spline[idk]])
        Ptt = np.asarray([spline(zz) for spline in self._Ptt_spline[idk]])
        Pmt = np.asarray([spline(zz) for spline in self._Pmt_spline[idk]])

        nbar_vel = self._nbar_vel(zz)
        nbar_red = self._nbar_red(zz)
           
        z_eff = integrals.zeff_integral(self._cosmo_params,self._k[idk],zz,
                                        Pmm,Ptt,
                                        self._survey_area, nbar_vel , nbar_red, self._errors,r,
                                        f,beta)


        if self._Fisher_in_bin is not None:
            Fisher  = np.sum(self._Fisher_in_bin,axis=0)
        
        else:
            Fisher = np.zeros((self._nparams, self._nparams))
            
            for t in range(self._nparams):
                for j in range(t, self._nparams):
                   
                    k_sum = integrals.mu_integral(self._k[idk],zz,
                                                Pmm, Pmt, Ptt,
                                                self._Data[t], self._Data[j],self._cosmo_params,
                                                self._survey_area,
                                                nbar_vel , nbar_red, self._errors,
                                                r,
                                                f,beta,sigma_8)

                    Fisher[t, j] = k_sum/(4.0*np.pi)
                    Fisher[j, t] = k_sum/(4.0*np.pi)


        #print some results on the terminal
        if self._verbosity:
            print("Fisher Matrix \n ====================== \n")
            for row in Fisher:
                    print(row , '\n') 
            print(" ====================== \n")     
           
        Covariance = sc.linalg.inv(Fisher)
                    
        growth_eff = self._growth_spline(z_eff)  
                        
        sigma8_eff = self._cosmo_params['sigma_8_0'] * growth_eff
        f_eff,_,_ = ut.compute_f_beta(z_eff,self._cosmo_params)
        beta_eff = f_eff*self._cosmo_params['beta_0']*growth_eff/(self._cosmo_params['Om_0']**(self._cosmo_params['gammaval']))

        #save the data
        data_tot = {}
        for w in range(self._nparams):
                if self._Data[w] == 0 :
                    data_tot['beta_eff'] = beta_eff
                    data_tot['beta_eff_err'] = np.sqrt(Covariance[w, w])
                   
                if self._Data[w] == 1 :
                    data_tot['fs8_eff'] = f_eff*sigma8_eff
                    data_tot['fs8_eff_err'] = np.sqrt(Covariance[w, w])
                    
                if self._Data[w] == 2 :
                    data_tot['r_g'] = self._cosmo_params['rg']
                    data_tot['r_g_err'] = np.sqrt(Covariance[w, w])
                    
                if self._Data[w] == 3 :
                    data_tot['sigma_g'] = self._cosmo_params['sigma_g']
                    data_tot['sigma_g_err'] = np.sqrt(Covariance[w, w])
                    
                if self._Data[w] == 3 :
                    data_tot['sigma_u'] = self._cosmo_params['sigma_u']
                    data_tot['sigma_u_err'] = np.sqrt(Covariance[w, w])

        data_tot['zeff'] = z_eff
        #print some results on the terminal
        if self._verbosity:
                print(f"Effective redshift z_eff = {z_eff}")
                for w in range(self._nparams):
                    if self._Data[w] == 0 :
                        print("beta(z_eff) \t percentage error(z_eff)")
                        print(f'{beta_eff:.6f} \t {100.0*(np.sqrt(Covariance[w, w])/(beta_eff)):.6f}')

                    if self._Data[w] == 1 :
                        print("fsigma8(z_eff) \t percentage error(z_eff)")
                        print(f'{f_eff*sigma8_eff:.6f} \t {100.0*(np.sqrt(Covariance[w, w])/(f_eff*sigma8_eff)):.6f}')

                    if self._Data[w] == 2 :
                        print("r_g \t percentage error")
                        rrg = self._cosmo_params['r_g']
                        print(f'{rg:.6f} \t {100.0*(np.sqrt(Covariance[w, w])/(rg)):.6f}')

                    if self._Data[w] == 3 :
                        print("sigma_g \t percentage error")
                        ssg = self._cosmo_params['sigma_g']
                        print(f'{ssg:.6f} \t {100.0*(np.sqrt(Covariance[w, w])/(ssg)):.6f}')

                    if self._Data[w] == 3 :
                        print("sigma_u \t percentage error")
                        ssu =self._cosmo_params['sigma_u']
                        print(f'{ssu:.6f} \t {100.0*(np.sqrt(Covariance[w, w])/(ssu)):.6f}')

        self._Fisher_tot = Fisher
        self._data_tot = data_tot
          
          

    @property
    def data(self):
        if self._data is not None:
            return self._data
        else:
            raise ValueError('run the method "compute_fisher" to compute the fisher forecast')

    @property
    def Fisher_zbin(self):
        if self._Fisher_in_bin is not None:
            return self._Fisher_in_bin
        else:
            raise ValueError('run the method "compute_fisher" to compute the fisher forecast')


    @property
    def data_tot(self):
        if self._data_tot is not None:
            return self._data_tot
        else:
            raise ValueError('run the method "compute_fisher_tot" to compute the fisher forecast')

    @property
    def Fisher_tot(self):
        if self._Fisher_tot is not None:
            return self._Fisher_tot
        else:
            raise ValueError('run the method "compute_fisher_tot" to compute the fisher forecast')

    @property
    def cosmo_params(self):
        return self._cosmo_params
       


        

        

    


        
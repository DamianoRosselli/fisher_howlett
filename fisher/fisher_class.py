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
from . import integrals as inte
import numpy as np
from tqdm import tqdm

C_LIGHT_KMS = cst.c.to('km/s').value


class Fisher(abc.ABC):

    """Generic class to handle PV Fisher matrix forecast and parameters.
    The parameters needed are explained in params.dat. See also examples."""

    def __init__(self, param_dic):
        """Initialize the class with parameters provided in a dictionary or a YAML file."""
        # Load the parameter dictionary from either a dictionary or a YAML file
        if isinstance(param_dic, dict):
            self._config = param_dic
            # Check if 'yaml_path' is in the dictionary and store it, otherwise set to 'No config file'
            if 'yaml_path' in param_dic:
                self._yml_path = param_dic['yaml_path']
            else:
                self._yml_path = 'No config file'
                
        elif isinstance(param_dic, str):
            # If a string is provided, assume it is a path to a YAML file and load it
            self._yml_path = param_dic
            with open(self._yml_path, "r") as f:
                self._config = yaml.safe_load(f)

        # Extract parameters from the configuration
        self._Data = self._config['Data'] 
        self._nparams = len(self._Data)
        self._zmin, self._zmax, self._nzbin = self._config['zrange']
        self._cosmo_params = self._config['cosmo_params']
        self._survey_area = self._config['survey_area']
        self._errors = self._config['errors']
        self._verbosity = self._config['verbosity']
        
        # Perform checks on the survey area configuration
        if (self._survey_area[0] == 0.) and (self._survey_area[2] == 0.):
            if 2 in self._Data:
                raise ValueError("ERROR: r_g is a free parameter, but there is no information in the density field (Fisher matrix will be singular)")
            if 3 in self._Data:
                raise ValueError("ERROR: sigma_g is a free parameter, but there is no information in the density field (Fisher matrix will be singular)")
        elif (self._survey_area[1] == 0.) and (self._survey_area[2] == 0.):
            if 4 in self._Data:
                raise ValueError("ERROR: sigma_u is a free parameter, but there is no information in the velocity field (Fisher matrix will be singular)")


        # Read power spectra files
        self._Pv, self._k = rfi.read_power_spectra_files(self._config['Pvel_file'])

        # Check if the maximum k value in the input power spectra is less than the required k_max
        for kk in self._Pv.keys():
            if self._cosmo_params['k_max'] > np.max(self._Pv[kk]['k']):
                raise ValueError("ERROR: The maximum k in the input power spectra is less than k_max")

        # Read nbar files and handle cases where z_vel or z_red is None
        self._z_vel, self._z_red, self._nbar_vel, self._nbar_red = rfi.read_nz_files(self._config['nbar_file'], self._config['density_unity'])
        if self._z_vel is None:
            self._z_vel = np.zeros(len(self._z_red))
            self._nbar_vel = np.zeros(len(self._z_red))
        elif self._z_red is None:
            self._z_red = np.zeros(len(self._z_vel))
            self._nbar_red = np.zeros(len(self._z_vel))

        # Compute splines for later use
        self._r_spline = ut.compute_r_spline(self._cosmo_params['Om_0'], self._cosmo_params['Od_0'], self._cosmo_params['w0'],self._cosmo_params['wa'])
        self._growth_spline = ut.compute_growth_spline(self._cosmo_params['Om_0'], self._cosmo_params['Od_0'], self._cosmo_params['w0'],self._cosmo_params['wa'], self._cosmo_params['gammaval'])
        self._Pmm_spline, self._Pmt_spline, self._Ptt_spline = ut.compute_Pz_spline(self._Pv)

        # Initialize Fisher matrix and data attributes to None
        self._Fisher_in_bin = None
        self._data = None
        self._Fisher_tot = None
        self._data_tot = None


    def compute_fisher_zbin(self):
        """Compute the Fisher Matrix for redshift bins and evaluate related parameters."""

        # Print the initial evaluation message indicating the number of bins and the redshift range
        print(f"Evaluating the Fisher Matrix for {self._nzbin - 1} bins between [z_min = {self._zmin}, z_max = {self._zmax}]")

        # Generate redshift bins from zmin to zmax with nzbin number of bins
        zbin = np.round(np.linspace(self._zmin, self._zmax, self._nzbin), 6)
        fisher_tot = []  # List to store the Fisher matrices for each bin
        data_tot = {}  # Dictionary to store data for each bin

        # Loop over each redshift bin (except the last one)
        for i in tqdm(range(len(zbin) - 1)):

            # Calculate the maximum radial distance for the current redshift bin
            rzmax = self._r_spline(zbin[i + 1])
            # Calculate the minimum wave number for the current redshift bin
            kmin = np.pi / rzmax

            # Print more information if verbosity is enabled
            if self._verbosity:
                print(f"Evaluating the Fisher Matrix for [k_min = {kmin}, k_max = {self._cosmo_params['k_max']}] and [z_min = {zbin[i]}, z_max = {zbin[i + 1]}]")

            # Check if the minimum k value in the power spectra is less than kmin
            for kk in self._Pv.keys():
                if kmin < np.min(self._Pv[kk]['k']):
                    raise ValueError("ERROR: The minimum k in the input power spectra is more than k_min")

            # Get the indices of k values within the range [kmin, k_max]
            idk = (self._k > kmin) & (self._k < self._cosmo_params['k_max'])

            # Generate equally spaced points within the current redshift bin
            zz = np.linspace(zbin[i], zbin[i + 1], 50)

            # Compute the growth factor and related parameters
            sigma_8 = self._cosmo_params['sigma_8_0'] * ut.growthz(zz, self._cosmo_params['Om_0'],
                                                                    self._cosmo_params['Od_0'],
                                                                    self._cosmo_params['w0'],
                                                                    self._cosmo_params['wa'],
                                                                    self._cosmo_params['gammaval'])
            f, beta, _ = ut.compute_f_beta(zz, self._cosmo_params)

            # Compute the radial distances for the redshift points
            r = self._r_spline(zz)

            # Compute the power spectra at the redshift points
            Pmm = np.asarray([spline(zz) for spline in self._Pmm_spline[idk]])
            Ptt = np.asarray([spline(zz) for spline in self._Ptt_spline[idk]])
            Pmt = np.asarray([spline(zz) for spline in self._Pmt_spline[idk]])

            # Get the number densities for velocity and redshift-space distortions
            nbar_vel = np.interp(zz, self._z_vel, self._nbar_vel)
            nbar_red = np.interp(zz, self._z_red, self._nbar_red)

            # Compute the effective redshift for the bin
            z_eff = inte.zeff_integral(self._cosmo_params, self._k[idk], zz, Pmm, Ptt,
                                  self._survey_area, nbar_vel, nbar_red, self._errors, r,
                                  f, beta)

            # Store the bin data in the dictionary
            data_tot[i] = {}
            data_tot[i]['zbin'] = [zbin[i], zbin[i + 1]]
            data_tot[i]['zeff'] = z_eff

            # Initialize the Fisher matrix for the current bin
            Fisher = np.zeros((self._nparams, self._nparams))

            # Compute the Fisher matrix elements
            for t in range(self._nparams):
                for j in range(t, self._nparams):
                    k_sum = inte.mu_integral(self._k[idk], zz, Pmm, Pmt, Ptt,
                                        self._Data[t], self._Data[j], self._cosmo_params,
                                        self._survey_area, nbar_vel, nbar_red, self._errors,
                                        r, f, beta, sigma_8)

                    Fisher[t, j] = k_sum / (4.0 * np.pi)
                    Fisher[j, t] = k_sum / (4.0 * np.pi)

            # Append the Fisher matrix for the current bin to the list
            fisher_tot.append(Fisher)

            # Print the Fisher matrix if verbosity is enabled
            if self._verbosity:
                print("Fisher Matrix \n ====================== \n")
                for row in Fisher:
                    print(row, '\n') 
                print(" ====================== \n")
            
            # Compute the covariance matrix from the Fisher matrix
            Covariance = sc.linalg.inv(Fisher)

            # Compute effective growth factor and related parameters
            growth_eff = self._growth_spline(z_eff)  
            sigma8_eff = self._cosmo_params['sigma_8_0'] * growth_eff
            f_eff, _, _ = ut.compute_f_beta(z_eff, self._cosmo_params)
            beta_eff = f_eff * self._cosmo_params['beta_0'] * growth_eff / (self._cosmo_params['Om_0'] ** self._cosmo_params['gammaval'])

            # Save the effective parameters and their errors
            for w in range(self._nparams):
                if self._Data[w] == 0:
                    data_tot[i]['beta_eff'] = beta_eff
                    data_tot[i]['beta_eff_err'] = np.sqrt(Covariance[w, w])
                if self._Data[w] == 1:
                    data_tot[i]['fs8_eff'] = f_eff * sigma8_eff
                    data_tot[i]['fs8_eff_err'] = np.sqrt(Covariance[w, w])
                if self._Data[w] == 2:
                    data_tot[i]['r_g'] = self._cosmo_params['r_g']
                    data_tot[i]['r_g_err'] = np.sqrt(Covariance[w, w])
                if self._Data[w] == 3:
                    data_tot[i]['sigma_g'] = self._cosmo_params['sigma_g']
                    data_tot[i]['sigma_g_err'] = np.sqrt(Covariance[w, w])
                if self._Data[w] == 3:
                    data_tot[i]['sigma_u'] = self._cosmo_params['sigma_u']
                    data_tot[i]['sigma_u_err'] = np.sqrt(Covariance[w, w])

            # Print effective parameter values and their errors if verbosity is enabled
            if self._verbosity:
                print(f"Effective redshift z_eff = {z_eff}")
                for w in range(self._nparams):
                    if self._Data[w] == 0:
                        print("beta(z_eff) \t percentage error(z_eff)")
                        print(f'{beta_eff:.6f} \t {100.0 * (np.sqrt(Covariance[w, w]) / beta_eff):.6f}')
                    if self._Data[w] == 1:
                        print("fsigma8(z_eff) \t percentage error(z_eff)")
                        print(f'{f_eff * sigma8_eff:.6f} \t {100.0 * (np.sqrt(Covariance[w, w]) / (f_eff * sigma8_eff)):.6f}')
                    if self._Data[w] == 2:
                        print("r_g \t percentage error")
                        rrg = self._cosmo_params['r_g']
                        print(f'{rrg:.6f} \t {100.0 * (np.sqrt(Covariance[w, w]) / rrg):.6f}')
                    if self._Data[w] == 3:
                        print("sigma_g \t percentage error")
                        ssg = self._cosmo_params['sigma_g']
                        print(f'{ssg:.6f} \t {100.0 * (np.sqrt(Covariance[w, w]) / ssg):.6f}')
                    if self._Data[w] == 3:
                        print("sigma_u \t percentage error")
                        ssu = self._cosmo_params['sigma_u']
                        print(f'{ssu:.6f} \t {100.0 * (np.sqrt(Covariance[w, w]) / ssu):.6f}')

        # Final completion message
        print(f"Evaluation of the Fisher Matrix for {self._nzbin - 1} bins between [z_min = {self._zmin}, z_max = {self._zmax}] is COMPLETE!!")

        # Store the Fisher matrices and bin data in class attributes
        self._Fisher_in_bin = np.asarray(fisher_tot)
        self._data = data_tot

    def compute_fisher_tot(self):
        """Compute the total Fisher Matrix and evaluate related parameters."""

        # Print the initial evaluation message indicating the redshift range
        print(f"Evaluating the Fisher Matrix between [z_min = {self._zmin}, z_max = {self._zmax}]")

        # Calculate the maximum radial distance for the maximum redshift
        rzmax = self._r_spline(self._zmax)
        # Calculate the minimum wave number for the given redshift range
        kmin = np.pi / rzmax

        # Check if the minimum k value in the power spectra is less than kmin
        for kk in self._Pv.keys():
            if kmin < np.min(self._Pv[kk]['k']):
                raise ValueError("ERROR: The minimum k in the input power spectra is more than k_min")

        # Get the indices of k values within the range [kmin, k_max]
        idk = (self._k > kmin) & (self._k < self._cosmo_params['k_max'])
        # Generate 100 equally spaced points within the redshift range
        zz = np.linspace(self._zmin, self._zmax, 100)

        # Compute the growth factor and related parameters
        sigma_8 = self._cosmo_params['sigma_8_0'] * ut.growthz(zz, self._cosmo_params['Om_0'],
                                                                self._cosmo_params['Od_0'],
                                                                self._cosmo_params['w0'],
                                                                self._cosmo_params['wa'],
                                                                self._cosmo_params['gammaval'])
        f, beta, _ = ut.compute_f_beta(zz, self._cosmo_params)

        # Compute the radial distances for the redshift points
        r = self._r_spline(zz)

        # Compute the power spectra at the redshift points
        Pmm = np.asarray([spline(zz) for spline in self._Pmm_spline[idk]])
        Ptt = np.asarray([spline(zz) for spline in self._Ptt_spline[idk]])
        Pmt = np.asarray([spline(zz) for spline in self._Pmt_spline[idk]])

        # Get the number densities for velocity and redshift-space distortions
        #nbar_vel = self._nbar_vel
        #nbar_red = self._nbar_red
        nbar_vel = np.interp(zz, self._z_vel, self._nbar_vel)
        nbar_red = np.interp(zz, self._z_red, self._nbar_red)

        # Compute the effective redshift for the entire range
        z_eff = inte.zeff_integral(self._cosmo_params, self._k[idk], zz, Pmm, Ptt,
                              self._survey_area, nbar_vel, nbar_red, self._errors, r,
                              f, beta)

        # Check if the Fisher matrix has been computed for bins
        if self._Fisher_in_bin is not None:
            # Sum the Fisher matrices across bins
            Fisher = np.sum(self._Fisher_in_bin, axis=0)
        else:
            # Initialize the Fisher matrix
            Fisher = np.zeros((self._nparams, self._nparams))

            # Compute the Fisher matrix elements
            for t in range(self._nparams):
                for j in range(t, self._nparams):
                    k_sum = inte.mu_integral(self._k[idk], zz, Pmm, Pmt, Ptt,
                                        self._Data[t], self._Data[j], self._cosmo_params,
                                        self._survey_area, nbar_vel, nbar_red, self._errors,
                                        r, f, beta, sigma_8)

                    Fisher[t, j] = k_sum / (4.0 * np.pi)
                    Fisher[j, t] = k_sum / (4.0 * np.pi)

        # Print the Fisher matrix if verbosity is enabled
        if self._verbosity:
            print("Fisher Matrix \n ====================== \n")
            for row in Fisher:
                print(row, '\n') 
            print(" ====================== \n")     
            
        # Compute the covariance matrix from the Fisher matrix
        Covariance = sc.linalg.inv(Fisher)

        # Compute effective growth factor and related parameters
        growth_eff = self._growth_spline(z_eff)  
        sigma8_eff = self._cosmo_params['sigma_8_0'] * growth_eff
        f_eff, _, _ = ut.compute_f_beta(z_eff, self._cosmo_params)
        beta_eff = f_eff * self._cosmo_params['beta_0'] * growth_eff / (self._cosmo_params['Om_0'] ** self._cosmo_params['gammaval'])

        # Save the effective parameters and their errors
        data_tot = {}
        for w in range(self._nparams):
            if self._Data[w] == 0:
                data_tot['beta_eff'] = beta_eff
                data_tot['beta_eff_err'] = np.sqrt(Covariance[w, w])
            if self._Data[w] == 1:
                data_tot['fs8_eff'] = f_eff * sigma8_eff
                data_tot['fs8_eff_err'] = np.sqrt(Covariance[w, w])
            if self._Data[w] == 2:
                data_tot['r_g'] = self._cosmo_params['rg']
                data_tot['r_g_err'] = np.sqrt(Covariance[w, w])
            if self._Data[w] == 3:
                data_tot['sigma_g'] = self._cosmo_params['sigma_g']
                data_tot['sigma_g_err'] = np.sqrt(Covariance[w, w])
            if self._Data[w] == 3:
                data_tot['sigma_u'] = self._cosmo_params['sigma_u']
                data_tot['sigma_u_err'] = np.sqrt(Covariance[w, w])

        data_tot['zeff'] = z_eff

        # Print effective parameter values and their errors if verbosity is enabled
        if self._verbosity:
            print(f"Effective redshift z_eff = {z_eff}")
            for w in range(self._nparams):
                if self._Data[w] == 0:
                    print("beta(z_eff) \t percentage error(z_eff)")
                    print(f'{beta_eff:.6f} \t {100.0 * (np.sqrt(Covariance[w, w]) / beta_eff):.6f}')
                if self._Data[w] == 1:
                    print("fsigma8(z_eff) \t percentage error(z_eff)")
                    print(f'{f_eff * sigma8_eff:.6f} \t {100.0 * (np.sqrt(Covariance[w, w]) / (f_eff * sigma8_eff)):.6f}')
                if self._Data[w] == 2:
                    print("r_g \t percentage error")
                    rg = self._cosmo_params['r_g']
                    print(f'{rg:.6f} \t {100.0 * (np.sqrt(Covariance[w, w]) / rg):.6f}')
                if self._Data[w] == 3:
                    print("sigma_g \t percentage error")
                    ssg = self._cosmo_params['sigma_g']
                    print(f'{ssg:.6f} \t {100.0 * (np.sqrt(Covariance[w, w]) / ssg):.6f}')
                if self._Data[w] == 3:
                    print("sigma_u \t percentage error")
                    ssu = self._cosmo_params['sigma_u']
                    print(f'{ssu:.6f} \t {100.0 * (np.sqrt(Covariance[w, w]) / ssu):.6f}')

        # Store the total Fisher matrix and bin data in class attributes
        self._Fisher_tot = Fisher
        self._data_tot = data_tot

    @property
    def data(self):
        if self._data is not None:
            return self._data
        else:
            raise ValueError('Run the method "compute_fisher" to compute the Fisher forecast.')

    @property
    def Fisher_zbin(self):
        if self._Fisher_in_bin is not None:
            return self._Fisher_in_bin
        else:
            raise ValueError('Run the method "compute_fisher" to compute the Fisher forecast.')

    @property
    def data_tot(self):
        if self._data_tot is not None:
            return self._data_tot
        else:
            raise ValueError('Run the method "compute_fisher_tot" to compute the Fisher forecast.')

    @property
    def Fisher_tot(self):
        if self._Fisher_tot is not None:
            return self._Fisher_tot
        else:
            raise ValueError('Run the method "compute_fisher_tot" to compute the Fisher forecast.')

    @property
    def cosmo_params(self):
        return self._cosmo_params

       


        

        

    


        
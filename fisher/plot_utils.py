#%% PLOT FORECASTS FS8
import numpy as np
import healpy as hp
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
from os import listdir
from os.path import isfile, join
from . import utils as ut
import pandas as pd

def extract_zrange_from_filename(filename, prefix):
    """
    Extracts the zrange from the filename.
    
    Args:
        filename (str): The name of the file.
        prefix (str): The prefix indicating the start of the zrange in the filename.
        
    Returns:
        str: The extracted zrange.
    """
    start = filename.find(prefix) + len(prefix)
    end = filename.find('.pkl')
    return filename[start:end]


def get_all_zrange_files(path, prefix):
    """
    Retrieves all filenames matching the pattern and extracts their zrange values.
    
    Args:
        path (str): The directory path to search for files.
        prefix (str): The prefix indicating the start of the zrange in the filename.
        
    Returns:
        list: List of all extracted zrange values.
    """
    zrange_files = []
    for f in listdir(path):
        if isfile(join(path, f)) and f.startswith(prefix) and f.endswith('.pkl'):
            zrange = extract_zrange_from_filename(f, prefix)
            zrange_files.append(zrange)
    return zrange_files


def get_ffc_files(plot_params):
    """
    Retrieves forecast files corresponding to each zrange value in zrange_list.
    
    Args:
        plot_params (dict): A dictionary containing plot parameters including zrange list, path, and filename.
    
    Returns:
        dict: A dictionary with zrange strings as keys and the loaded forecast data as values.
    """
    zrange_list, path, filename = plot_params['zrange'], plot_params['path'], plot_params['filename']
    
    ffc_result_zrange = {}
    # Iterate over the zrange list
    for zr in zrange_list:
        # Construct the file path
        file_path = os.path.join(path, f'{filename}{zr}.pkl')
        # Open and load the forecast data from the file
        with open(file_path, 'rb') as f:
            ffc_result_zrange[zr] = pickle.load(f)
    return ffc_result_zrange


def get_plot_colors(plot_params):
    """
    Determines the colors to be used for plotting based on the plot_mode.
    
    Args:
        plot_params (dict): Dictionary containing parameters for plotting.
    
    Returns:
        list: List of colors to be used for plotting.
    """
    # Get the necessary plot parameters
    dataset_to_plot = plot_params['dataset']
    ftype_to_plot = plot_params['ftype']
    plot_mode = plot_params['plot_mode']
    
    # Determine the number of configurations to plot
    if plot_mode == 'zrange':
        n_configs = len(dataset_to_plot) * len(ftype_to_plot)
    elif plot_mode == 'dataset':
        n_configs = len(ftype_to_plot)

    # Generate a list of colors using a colormap
    colors = plt.cm.gist_rainbow(np.linspace(0, 1, n_configs))
    return colors


def get_fs8_ffc(z, cosmo_params):
    """
    Computes the growth rate of structure f*sigma_8 at given redshifts.
    
    Args:
        z (array): Array of redshift values.
        cosmo_params (dict): Dictionary containing cosmological parameters.
    
    Returns:
        array: Array of computed f*sigma_8 values.
    """
    Om = cosmo_params['Om_0']
    gammaval = cosmo_params['gammaval']
    Od = cosmo_params['Od_0']
    w0 = cosmo_params['w0']
    wa = cosmo_params['w0']
    
    # Compute the growth factor using a spline function
    growth = ut.compute_growth_spline(Om, Od, w0, wa, gammaval)
    
    # Calculate sigma_8 at each redshift
    sigma8 = ut.cosmo_params['sigma_8_0'] * growth(z)
    
    # Compute the growth rate f
    f, _, _ = ut.compute_f_beta(z, cosmo_params)
    
    # Return the product of f and sigma_8
    return f * sigma8

        
def plot_ffc_values(plot_params):
    """
    Plot the estimated values of fs8_eff with error bars for each dataset and forecast type.

    Args:
        plot_params (dict): Dictionary containing all parameters for plotting.
    """
    if plot_params['plot_mode'] == 'zrange':
        plot_zrange_result(plot_params)
        
    elif plot_params['plot_mode'] == 'dataset':
        plot_dataset_result(plot_params)
        
    elif plot_params['plot_mode'] == 'ftype':
        plot_ftype_result(plot_params)


def plot_values(ffc_values, plot_params, dataset, ftype, color, zorder):
    """
    Plot the values and error bars for a given dataset and forecast type.

    Args:
        ffc_values (dict): Dictionary containing the fisher forecast values.
        plot_params (dict): Dictionary containing all parameters for plotting.
        dataset (str): The dataset name.
        ftype (str): The forecast type.
        color (str): Color for the plot.
    """
    
    vals_zbin = [pd.DataFrame.from_dict(ffc_values['fisher_zbin'], orient='index')[key] for key in ['zeff', 'fs8_eff', 'fs8_eff_err']]
    vals_tot = [ffc_values['fisher_tot'][key] for key in ['zeff', 'fs8_eff', 'fs8_eff_err']]

    if plot_params['values'] == 'fs8':
        
        if plot_params['plot_mode'] == 'ftype' :
            plt.errorbar(*vals_zbin[:2], yerr=vals_zbin[2], fmt='s', mfc=color, mec='k', ecolor='k',
                         ms=12, elinewidth=2, capsize=5, capthick=2, alpha=1, label=f'{dataset} {ftype}', zorder = zorder)
        else : 
            plt.errorbar(*vals_zbin[:2], yerr=vals_zbin[2], fmt='s', mfc=color, mec='k', ecolor= color,
                         ms=12, elinewidth=2, capsize=5, capthick=2, alpha=1, label=f'{dataset} {ftype}', zorder = zorder)
           
        if plot_params['error_bands']:
            
            if plot_params['plot_mode'] == 'ftype' : 
                plt.fill_between(vals_zbin[0], vals_zbin[1] + vals_zbin[2], vals_zbin[1] - vals_zbin[2], color='grey', alpha=0.3)
            else : 
                plt.fill_between(vals_zbin[0], vals_zbin[1] + vals_zbin[2], vals_zbin[1] - vals_zbin[2], color=color, alpha=0.3)

        if plot_params['ztot'] == True :
            
            if plot_params['plot_mode'] == 'ftype' :
                plt.errorbar(*vals_tot[:2], yerr=vals_tot[2], fmt='o', mfc='k', mec='k', ecolor='k',
                     ms=12, elinewidth=2, capsize=5, capthick=2, alpha=1, label=f'{dataset} {ftype} (all)', zorder = zorder)
            else: 
                plt.errorbar(*vals_tot[:2], yerr=vals_tot[2], fmt='o', mfc=color, mec='k', ecolor='k',
                     ms=12, elinewidth=2, capsize=5, capthick=2, alpha=1, label=f'{dataset} {ftype} (all)', zorder = zorder)
    
    elif plot_params['values'] == 'errors':
        
        if plot_params['plot_mode'] == 'ftype' : 

            plt.plot(vals_zbin[0], (vals_zbin[2]/vals_zbin[1])*100, 's-', mfc=color, mec='k', color = 'k', 
                 ms=12, alpha=1, label=f'{dataset} {ftype}', zorder = zorder)
            
        else :
            
             plt.plot(vals_zbin[0], (vals_zbin[2]/vals_zbin[1])*100, 's-', mfc=color, mec='k', color = color, 
                 ms=12, alpha=1, label=f'{dataset} {ftype}', zorder = zorder)

        if plot_params['ztot'] ==True :
            
            if plot_params['plot_mode'] == 'ftype' : 
            
                plt.plot(vals_tot[0], (vals_tot[2]/vals_tot[1])*100, 'o', mfc='k', mec='k', color = 'k', 
                     ms=12, alpha=1, label=f'{dataset} {ftype} (all)', zorder = zorder)
                
            else : 
                plt.plot(vals_tot[0], (vals_tot[2]/vals_tot[1])*100, 'o', mfc=color, mec='k', color = 'k', 
                     ms=12, alpha=1, label=f'{dataset} {ftype} (all)', zorder = zorder)
                
    

def figure_settings(fig_settings):
    plt.ylim(fig_settings.get('ylim'))
    plt.xlabel(fig_settings.get('xlabel'), fontsize=fig_settings.get('fontsize'))
    plt.ylabel(fig_settings.get('ylabel'), fontsize=fig_settings.get('fontsize'))
    plt.title( fig_settings.get('title' ), fontsize=fig_settings.get('fontsize'))
    plt.legend(framealpha=0, fontsize=10, loc='best', ncol=1)
    plt.tick_params(
        labelbottom=True, labeltop=False, labelleft=True, labelright=False,
        bottom=True, top=True, left=True, right=True, 
        direction="in", which='both', labelsize=15, width=1, size=5)
    

def plot_zrange_result(plot_params):
    """
    Plot results for different redshift ranges.
    
    Args:
        plot_params (dict): Dictionary containing parameters for plotting.
    """
    zrange_list = plot_params['zrange']
    dataset_list = plot_params['dataset']
    ftype_list = plot_params['ftype']
    fig_idx = plot_params['fig_settings']['fig_idx']
    
    ffc_result_zrange = get_ffc_files(plot_params)
    common_params = ffc_result_zrange[zrange_list[0]][dataset_list[0]][ftype_list[0]]['parameters']
    cosmo_params = common_params['cosmo_params']
    colors = get_plot_colors(plot_params)
    
    for zr_idx, (n_zrange, ffc_datasets) in enumerate(ffc_result_zrange.items()):
        
        if n_zrange not in zrange_list:
            continue
        
    
        plt.figure(num= plot_params['fig_settings']['fig_idx'], figsize=(12, 8))
        
        zrange = ffc_datasets[dataset_list[0]][ftype_list[0]]['parameters']['zrange']
        color_idx = 0

        if plot_params['values'] == 'fs8':
        
            z_plot = np.linspace(zrange[0], zrange[1], 101)
            
            # Number of iterations based on the product of lengths of w0 and wa lists
            n_w0wa = len(plot_params['new_w0wa']['w0']) * len(plot_params['new_w0wa']['wa'])
            # Define custom colors based on the number of lines to plot
            f_colors = plt.cm.jet(np.linspace(0, 1, n_w0wa))  
            # Create a ListedColormap using the custom colors
            cmap = mcolors.ListedColormap(f_colors)
        
            # Define bounds for the discrete colors
            bounds = np.arange(n_w0wa + 1)
            norm = mcolors.BoundaryNorm(bounds, cmap.N)
            tick_labels = []
            
            # Iterate over each combination of w0 and wa
            for w0_val in plot_params['new_w0wa']['w0']:
                for wa_val in plot_params['new_w0wa']['wa']:
                    cosmo_params['w0'] = w0_val
                    cosmo_params['wa'] = wa_val
                    
                    fs8_z = get_fs8_ffc(z_plot, cosmo_params)
                    
                    # Plot each line with appropriate labels and colors
                    if (w0_val == -1.) and (wa_val == 0.):
                        f_colors[color_idx] = [0,0,0,1]
                        tick_labels.append(r'$\Lambda$CDM')
                        zorder = 20
                    else:
                        tick_labels.append(f"({w0_val}, {wa_val})")
                        zorder = 10
                        
                    plt.plot(z_plot, fs8_z, linestyle='-', c=f_colors[color_idx], zorder = zorder)
                    color_idx += 1
                    
            # Add a colorbar
            cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ticks=np.arange(n_w0wa))
            tick_locs = (np.arange(n_w0wa) + 0.5) * (n_w0wa / n_w0wa)
            cb.set_ticks(tick_locs)
            cb.ax.set_title(r'$(\omega_0, \omega_a)$', fontsize = 15)
            cb.ax.set_yticklabels(tick_labels, fontsize=15)
        
        color_idx = 0
        for dataset, ffc_ftype in ffc_datasets.items():
            if dataset not in dataset_list:
                continue
            for ftype, ffc_values in ffc_ftype.items():
                if ftype not in ftype_list:
                    continue
                color = colors[color_idx]
                plot_values(ffc_values, plot_params, dataset, ftype, color, zorder = 30)
                color_idx += 1

        plt.xlim(zrange[0], zrange[1])
        plt.tight_layout()
        figure_settings(plot_params['fig_settings'])
        if plot_params['fig_settings'].get('save', False):
            save_fig(zr_idx, plot_params)

        #plt.show()
        #fig_idx +=1


def plot_dataset_result(plot_params):
    """
    Plot results for different datasets.
    
    Args:
        plot_params (dict): Dictionary containing parameters for plotting.
    """
    zrange_list = plot_params['zrange']
    dataset_list = plot_params['dataset']
    ftype_list = plot_params['ftype']
    
    ffc_result_zrange = get_ffc_files(plot_params)
    common_params = ffc_result_zrange[zrange_list[0]][dataset_list[0]][ftype_list[0]]['parameters']
    cosmo_params = common_params['cosmo_params']
    colors = get_plot_colors(plot_params)
     
    for n_zrange, ffc_datasets in ffc_result_zrange.items():
        if n_zrange not in zrange_list:
            continue
        zrange = ffc_datasets[dataset_list[0]][ftype_list[0]]['parameters']['zrange']
        for dataset, ffc_ftype in ffc_datasets.items():
            if dataset not in dataset_list:
                continue

            plt.figure(num= plot_params['fig_settings']['fig_idx'], figsize=(12, 8))
            color_idx = 0
            
            if plot_params['values'] == 'fs8':
                    
                z_plot = np.linspace(zrange[0], zrange[1], 101)
                
                # Number of iterations based on the product of lengths of w0 and wa lists
                n_w0wa = len(plot_params['new_w0wa']['w0']) * len(plot_params['new_w0wa']['wa'])
                # Define custom colors based on the number of lines to plot
                f_colors = plt.cm.jet(np.linspace(0, 1, n_w0wa))  
                # Create a ListedColormap using the custom colors
                cmap = mcolors.ListedColormap(f_colors)
            
                # Define bounds for the discrete colors
                bounds = np.arange(n_w0wa + 1)
                norm = mcolors.BoundaryNorm(bounds, cmap.N)
                tick_labels = []
                
                # Iterate over each combination of w0 and wa
                for w0_val in plot_params['new_w0wa']['w0']:
                    for wa_val in plot_params['new_w0wa']['wa']:
                        cosmo_params['w0'] = w0_val
                        cosmo_params['wa'] = wa_val
                        
                        fs8_z = get_fs8_ffc(z_plot, cosmo_params)
                        
                        # Plot each line with appropriate labels and colors
                        if (w0_val == -1.) and (wa_val == 0.):
                            f_colors[color_idx] = [0,0,0,1]
                            tick_labels.append(r'$\Lambda$CDM')
                            zorder = 20
                        else:
                            tick_labels.append(f"({w0_val}, {wa_val})")
                            zorder = 10
                            
                        plt.plot(z_plot, fs8_z, linestyle='-', c=f_colors[color_idx], zorder = zorder)
                        color_idx += 1
                        
                # Add a colorbar
                cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ticks=np.arange(n_w0wa))
                tick_locs = (np.arange(n_w0wa) + 0.5) * (n_w0wa / n_w0wa)
                cb.set_ticks(tick_locs)
                cb.ax.set_title(r'$(\omega_0, \omega_a)$', fontsize = 15)
                cb.ax.set_yticklabels(tick_labels, fontsize=15)
            
            color_idx = 0
            
            for ftype, ffc_values in ffc_ftype.items():
                if ftype not in ftype_list:
                    continue
                color = colors[color_idx]
                plot_values(ffc_values, plot_params, dataset, ftype, color, zorder = 30)
                color_idx += 1
                
            plt.xlim(zrange[0], zrange[1])
            figure_settings(plot_params['fig_settings'])
            plt.tight_layout()

            if plot_params['fig_settings'].get('save', False):
                save_fig(zrange_list.index(n_zrange), plot_params)

            #plt.show()
            #fig_idx +=1
            
            
def plot_ftype_result(plot_params):
    """
    Plot results for different forecast types.
    
    Args:
        plot_params (dict): Dictionary containing parameters for plotting.
    """
    zrange_list = plot_params['zrange']
    dataset_list = plot_params['dataset']
    ftype_list = plot_params['ftype']
    
    ffc_result_zrange = get_ffc_files(plot_params)

    for n_zrange, ffc_datasets in ffc_result_zrange.items():
        if n_zrange not in zrange_list:
            continue
        for dataset, ffc_ftype in ffc_datasets.items():
            if dataset not in dataset_list:
                continue
            for ftype, ffc_values in ffc_ftype.items():
                if ftype not in ftype_list:
                    continue

                plt.figure(num= plot_params['fig_settings']['fig_idx'], figsize=(12, 8))
                ffc_params = ffc_ftype[ftype]['parameters']
                cosmo_params = ffc_params['cosmo_params']
                zrange = ffc_params['zrange']
                z_plot = np.linspace(zrange[0], zrange[1], 100)
                
                if plot_params['values'] == 'fs8':
                    
                    z_plot = np.linspace(zrange[0], zrange[1], 101)
                    
                    # Number of iterations based on the product of lengths of w0 and wa lists
                    n_w0wa = len(plot_params['new_w0wa']['w0']) * len(plot_params['new_w0wa']['wa'])
                    # Define custom colors based on the number of lines to plot
                    f_colors = plt.cm.jet(np.linspace(0, 1, n_w0wa))  
                    # Create a ListedColormap using the custom colors
                    cmap = mcolors.ListedColormap(f_colors)
                
                    # Define bounds for the discrete colors
                    bounds = np.arange(n_w0wa + 1)
                    norm = mcolors.BoundaryNorm(bounds, cmap.N)
                    tick_labels = []
                    color_idx = 0
                    # Iterate over each combination of w0 and wa
                    for w0_val in plot_params['new_w0wa']['w0']:
                        for wa_val in plot_params['new_w0wa']['wa']:
                            cosmo_params['w0'] = w0_val
                            cosmo_params['wa'] = wa_val
                            
                            fs8_z = get_fs8_ffc(z_plot, cosmo_params)
                            
                            # Plot each line with appropriate labels and colors
                            if (w0_val == -1.) and (wa_val == 0.):
                                f_colors[color_idx] = [0,0,0,1]
                                tick_labels.append(r'$\Lambda$CDM')
                                zorder = 20
                            else:
                                tick_labels.append(f"({w0_val}, {wa_val})")
                                zorder = 10
                                
                            plt.plot(z_plot, fs8_z, linestyle='-', c=f_colors[color_idx], zorder = zorder)
                            color_idx += 1
                            
                    # Add a colorbar
                    cb = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ticks=np.arange(n_w0wa))
                    tick_locs = (np.arange(n_w0wa) + 0.5) * (n_w0wa / n_w0wa)
                    cb.set_ticks(tick_locs)
                    cb.ax.set_title(r'$(\omega_0, \omega_a)$', fontsize = 15)
                    cb.ax.set_yticklabels(tick_labels, fontsize=15)
                
                color = 'w'
                plot_values(ffc_values, plot_params, dataset, ftype, color, zorder = 30)
    
                plt.xlim(zrange[0], zrange[1])
                figure_settings(plot_params['fig_settings'])
                plt.tight_layout()
                
                if plot_params['fig_settings'].get('save', False):
                    save_fig(zrange_list.index(n_zrange), plot_params)
                
                #plt.show()
                #fig_idx +=1

def save_fig(idx, plot_params):
    """
    Save the current figure with a filename based on plot parameters.

    Args:
        idx (int): The index of the current zrange in the zrange list.
        plot_params (dict): Dictionary containing parameters for plotting.
    """
    dir_path = plot_params['path']
    fig_format = plot_params['fig_settings']['fig_format']
    ffc_name = plot_params['filename'] + plot_params['zrange'][idx]
    fig_name = f"{dir_path}/{ffc_name}_{plot_params['plot_mode']}_{plot_params['values']}.{fig_format}"
    plt.savefig(fig_name, format=fig_format)


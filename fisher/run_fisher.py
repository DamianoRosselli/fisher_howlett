#%% CREATE DICTIONARY FOR FISHER FORECASTS CONFIGURATION AND RUN MULTIPLE PROCEDURES

from . import fisher_class as fcla
import time

def generate_filenames(zrange_list, fc_file_start):
    """
    Generate filenames associated with different zranges.
     
    Args:
    - zrange_list (list): List of zranges.
    - fc_file_start (str): Base filename prefix.
    
    Returns:
    - List of filenames associated with the zranges.
    """
    filenames = []
    
    for zrange in zrange_list:
        # Convert each element in zrange to string
        zrange_str = '_'.join([str(int(z * 100)).zfill(3) 
                               if isinstance(z, float) 
                               else str(z) for z in zrange])
        
        filename = f"{fc_file_start}{zrange_str}"
        filenames.append(filename)
        
    return filenames


def run_ffc_all_configs(zrange_list, cosmo_params, errors, verbosity, Pvel_file, data_surveys, path, ffc_zbin, ffc_tot, fc_file_name):
    """
    Run Fisher Forecast (FFC) for all configurations within specified zranges.
    
    Args:
    - zrange_list (list): List of zranges. Each zrange is represented as a list 
                          containing three elements: [start, end, step].
    - cosmo_params (dict): Dictionary containing cosmological parameters.
    - errors (dict): Dictionary containing error parameters.
    - verbosity (bool): Verbosity flag.
    - Pvel_file (str): Path to the peculiar velocity file.
    - data_surveys (dict): Dictionary containing survey data.
    - path (str): Path to the directory where results will be saved.
    
    Returns:
    - Dictionary containing FFC results for each zrange.
    """
    # Define the prefix for forecast filenames
    fc_file_name = fc_file_name
    
    # Generate a list of forecast filenames associated with each zrange
    fc_filenames_list = generate_filenames(zrange_list, fc_file_name)
      
    # Check if the directory to save results exists, if not, create it
    results_dir = os.path.join(path, 'forecasts results w0wa')
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Initialize an empty dictionary to store FFC results for each zrange
    ffc_result_zrange = {}

    # Iterate over each zrange in the zrange_list
    for i, zrange in tqdm(enumerate(zrange_list)):
        
        # List of common parameters shared among all configurations
        common_params = [zrange, cosmo_params, errors, verbosity, Pvel_file]

        # Print the zrange for which the forecast is being performed
        print('Fisher Forecast on zrange = ', zrange)
        
        # Get configurations for the current zrange
        configs = get_configs(common_params, data_surveys, path)
        
        # Print the configurations for the current zrange
        print(pd.DataFrame(configs),'\n')

        # Run forecasts for the current zrange and store the results in ffc_result_zrange
        ffc_result_zrange[i] = run_forecasts(configs, path, fc_filenames_list[i], ffc_zbin, ffc_tot)


# need to implement after pessimistic vs optimistic errors choice in get_configs change errors and kmax
def get_configs(common_params, data_params, path):
    """
    Create parameter dictionaries for different survey configurations, combining common and survey-specific parameters.

    Args:
        common_params (list): List containing zrange and the parameters all configurations have in common.
        data_params (dict): Dictionary containing survey-specific parameters.
        path (str): Base path for the project files.

    Returns:
        dict: A dictionary containing all parameter configurations for each survey.
    """    
    
    # Define flags for different forecast types
    ffc_type = {
        'density_only': [0, 1, 3],   # Flags for density-only forecasts
        'velocity_only': [1, 4],     # Flags for velocity-only forecasts
        'combined': [0, 1, 3, 4]     # Flags for combined forecasts
    }
    
    all_configs = {}  # Initialize an empty dictionary to store all configurations
    
    # Iterate over each survey in data_surveys
    for data, params in data_params.items():
        
        # Define survey areas for different forecast types
        area = {
            'density_only': [params['area_density'], 0, 0],  # Area configuration for density-only forecasts
            'velocity_only': [0, params['area_velocity'], 0],  # Area configuration for velocity-only forecasts
            'combined': [0, 1., params['area_density']]  # Area configuration for combined forecasts
        }
        
        # Create parameter dictionaries for each forecast type
        data_all_params = {
            ftype: create_params_dict(
                *common_params,              # Common parameters shared across all configurations
                params['nz_files'],          # File paths for number density data
                params['density_units'],     # Unit conversion factors for densities
                area[ftype],                 # Survey areas for the current forecast type
                ffc_type[ftype]              # Flags for the current forecast type
            ) 
            for ftype in ffc_type  # Iterate over each forecast type
        }
        
        all_configs[data] = data_all_params  # Store the parameter dictionaries for the current survey
    
    return all_configs  # Return the dictionary containing all parameter configurations


def run_forecasts(configs, path, namefile, ffc_zbin, ffc_tot):
    """
    Run forecasts for each configuration and save the results.

    Args:
        configs (dict): Dictionary containing the forecast configurations.
        namefile (str): Name of the file to save the results.
        path (str): Base path for the project files.

    Returns:
        dict: Dictionary containing the forecast results.
    """
    start_all = time.time()  # Record the start time for measuring execution time
    all_forecasts = {}  # Initialize an empty dictionary to store all forecast results

    # Iterate over each survey in the forecast dictionary
    for data, types in configs.items():
        print('# =============================================================================', 
              '\n   ', data, 'forecast',
              '\n# =============================================================================\n')

        ftype_results = {}  # Initialize a dictionary to store results for each forecast type

        # Iterate over each forecast type for the current survey
        for ftype, params in types.items():
            c = fcla.Fisher(params)  # Initialize the Fisher object with the current parameters
            
            if (ffc_zbin == True) and (ffc_tot == False) :
                print('\n• ', ftype, 'forecast on redshift bins\n')
                c.compute_fisher_zbin()  # Compute the Fisher matrix for redshift bins
                
                # Store the results for the current forecast type
                ftype_results[ftype] = {
                'parameters': params,          # Parameters used in the forecast
                'fisher_zbin': c.data,         # Fisher data for redshift bins
                'matrix_zbin': c.Fisher_zbin,  # Fisher matrix for redshift bins
                }
                
            elif (ffc_zbin == False) and (ffc_tot == True):
                print('\n• ', ftype, 'forecast on all z range\n')
                c.compute_fisher_tot()   # Compute the Fisher matrix for the entire redshift range
                
                # Store the results for the current forecast type
                ftype_results[ftype] = {
                'parameters': params,          # Parameters used in the forecast
                'fisher_tot': c.data_tot,      # Fisher data for the entire redshift range
                'matrix_tot': c.Fisher_tot     # Fisher matrix for the entire redshift range
                }
                
            elif (ffc_zbin == True) and (ffc_tot == True) :
                
                print('\n• ', ftype, 'forecast on redshift bins\n')
                c.compute_fisher_zbin()  # Compute the Fisher matrix for redshift bins
                print('\n• ', ftype, 'forecast on all z range\n')
                c.compute_fisher_tot()   # Compute the Fisher matrix for the entire redshift range

                # Store the results for the current forecast type
                ftype_results[ftype] = {
                'parameters': params,          # Parameters used in the forecast
                'fisher_zbin': c.data,         # Fisher data for redshift bins
                'matrix_zbin': c.Fisher_zbin,  # Fisher matrix for redshift bins
                'fisher_tot': c.data_tot,      # Fisher data for the entire redshift range
                'matrix_tot': c.Fisher_tot     # Fisher matrix for the entire redshift range
                }

            
            print('\n ------------------------------------------------------------------------------')
            
        all_forecasts[data] = ftype_results  # Store the results for the current survey

    # Save the forecast results to a pickle file
    with open(os.path.join(path, 'forecasts results w0wa', namefile + '.pkl'), 'wb') as f:
        pickle.dump(all_forecasts, f)
                
    print('\n ALL FORECASTS DONE! Results saved in :', namefile)
        
    end_all = time.time()  # Record the end time for measuring execution time
    exec_time_all = end_all - start_all  # Calculate the total execution time
    minutes = int(exec_time_all // 60)  # Calculate minutes part of the execution time
    seconds = exec_time_all % 60  # Calculate seconds part of the execution time

    # Print the total execution time
    print("\nTime of execution: ", f"{minutes} minutes and {seconds:.3f} seconds\n")
    

def create_params_dict(zrange, cosmo_params, errors, verbosity, Pvel_file, nz_files, density_units, survey_area, flags):
    """
    Create a parameters dictionary with inputs from the user, files, and computed values.
    
    Args:
        files (list of str): List of file paths to read data from.
        cosmo_params (dict): Dictionary containing cosmological parameters.
        survey_area (list of float): List containing survey areas for each survey and overlap area between surveys.
        errors (dict): Dictionary containing error parameters.
        verbosity (bool): Boolean flag for output verbosity on terminal.
        Pvel_file (str): File path containing the velocity divergence power spectrum.
        density_units (list of float): Unit conversion factors for densities.
    
    Returns:
        dict: Dictionary containing all parameters needed for the calculations.
    """

    # Define the parameters dictionary
    parameters = {
        'Data': flags,  # A vector of flags for the parameters of interest (0=beta, 1=fsigma8, 2=r_g, 3=sigma_g, 4=sigma_u).
        'zrange': zrange, # Redshift range with min, max, and number of bins.
        'cosmo_params': cosmo_params,  # Dictionary containing cosmological parameters.
        'survey_area': survey_area,    # Survey areas for each survey and overlap area between surveys.
        'errors': errors,              # Dictionary containing error parameters.
        'verbosity': verbosity,        # Boolean flag for output verbosity on terminal.
        'Pvel_file': Pvel_file,    # File path containing the velocity divergence power spectrum.
        'nbar_file': nz_files,     # List of new files with overlapping redshift values.
        'density_unity': density_units, # Unit conversion factor for density (to maintain precision).
    }
    
    return parameters

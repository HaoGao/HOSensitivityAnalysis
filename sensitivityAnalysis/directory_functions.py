    
import os


def setup_save_dir(start_index: int, num_train: int, log_unif_prior: bool, sa_type: str):
    
    # base directory where both SA1 and SA2 results are stored
    base_save_dir = f'results/sa_{start_index}_{num_train}_{log_unif_prior}'      
    
    # directory where plots are stored
    save_dir = f'{base_save_dir}/{sa_type}'
    
    # directory where sampled SA index values are stored
    data_save_dir = f'{save_dir}/resultsData'
    
    # create these directories if they don't exist
    if not os.path.exists(base_save_dir):  os.makedirs(base_save_dir)
    if not os.path.exists(save_dir):  os.makedirs(save_dir) 
    if not os.path.exists(data_save_dir):  os.makedirs(data_save_dir) 
    
    return save_dir, data_save_dir
  
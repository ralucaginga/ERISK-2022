import os
from glob import glob

data_folder = 'data'
logs_folder = 'logs'

data_json_path = os.path.join(data_folder, 'full_data.json')
individual_train_path = os.path.join('data', 'individual_train_vec.npz')
individual_dev_path = os.path.join('data', 'individual_dev_vec.npz')
individual_labels_path = os.path.join('data', 'individual_labels.npy')

def get_new_experiment_folder(model_name):
    folder_re = os.path.join(logs_folder, f'{model_name}*')
    experiment_id = len(glob(folder_re))
    experiment_folder = os.path.join(logs_folder, f'{model_name}_{experiment_id}')
    os.makedirs(experiment_folder)
    return experiment_folder

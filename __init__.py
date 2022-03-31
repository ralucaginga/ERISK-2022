import os

data_folder = 'data'

data_json_path = os.path.join(data_folder, 'full_data.json')
individual_train_path = os.path.join('data', 'individual_train_vec.npz')
individual_dev_path = os.path.join('data', 'individual_dev_vec.npz')
individual_labels_path = os.path.join('data', 'individual_labels.npy')
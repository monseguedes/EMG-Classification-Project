"""
@author: Monse Guedes Ayala
@project: Bioliberty EMG Classification Algorithm

This is the main file to control the experiments.
"""

# Python imports
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

#Self-created imports
import pre_processing as prep
import feature_functions as ffun
import modelling as model
import lists_features as feat

#Plots parameters
sns.set_style('darkgrid')
plt.rcParams['font.size'] = '20'


# Choose process variables
#-----------------------------------------------
dataset_name = 'limbposition'

channels_used = [1,2,3,4,5,6,7]    # Channels to be used (smaller than total no.channels)

window_size = 300
window_overlap = 150

wavelet_type = 'db1'
wavelet_level = 4

model_name = 'svc'    # Choose: svc, lda, random_forest


# Create dataset class and save only if necessary 
# ----------------------------------------------
dataset = model.Dataset(dataset_name=dataset_name)
dataset.create_save_complete_dataframe(no_subject=1, window_size=window_size, overlap= window_overlap, features= feat.time_features + feat.time_freq_features, 
                                    channels_used=channels_used, wavelet_level=wavelet_level, wavelet_type=wavelet_type, overwrite=False)


# Import dataset object, which was either just created, or was already stored in main file
# ----------------------------------------------
filehandler = open(dataset.file_name, 'rb') 
subject = pickle.load(filehandler)


# Set hyperparameters for models
# ----------------------------------------------
if model_name == 'svc':
    hyperparameters_dict = {'kernel': 'linear', 'C': 10}
elif model_name == 'random_forest':
    hyperparameters_dict = {'n_estimators': 100, 'max_depth': 10, 'min_samples_leaf': 1}
else:
    hyperparameters_dict = []
      

# Create machine learning model
# ----------------------------------------------
mlmodel = model.ML_Model(subject= subject, model_name= model_name, hyperparameters= hyperparameters_dict)   


# Create and store results table for some list of feature sets (which is taken from the list_features file)
# ----------------------------------------------
mlmodel.create_featuresets_dataframe(featuresets_dict=feat.feature_sets_dict, scale=True, pca=False, n_components=0.95, plots=True, cross_validation=False)   
# mlmodel.store_results_dataframe()    # Store these results


# Plot functions
# ----------------------------------------------
# mlmodel.plot_pca(featuresets_dict=feat.feature_sets_dict)    # Plot PCA plots for desired feature sets, this is independent of other processes
# mlmodel.feature_selection_plots()    # Use only for individual channels, otherwise too slow and plots have too much information


# Wrapper feature selection
# ----------------------------------------------
# mlmodel.forward_feature_selection(35, 'all_features')[0]
# mlmodel.store_forward_selection_results()

# mlmodel.backward_feature_elimination(256, 'all_features')[0]    # This is quite computationally expensive
# mlmodel.store_backward_elimination_results()

# Use feature sets from forward selection to get other results
# ----------------------------------------------
# mlmodel.plot_forward_selection_true_results(no_features=35, feature_set_name = 'all features')
# mlmodel.plot_PCA_improvement(no_features=35, feature_set_name = 'all features')


# Set hyperparameters for grid search table
# ----------------------------------------------
if model_name == 'svc':
    param_grid = [
            {'C': [0.1, 1, 10, 100], 'kernel': ['linear']},
            {'C': [1, 10], 'gamma': [0.0001, 0.001], 'kernel': ['rbf']},
            {'C': [1, 10], 'gamma': [0.0001, 0.001], 'degree': [2,6], 'kernel': ['poly']}
            ]
elif model_name == 'random_forest':
    param_grid = [
                {'n_estimators': [100, 500, 750, 1000], 'max_depth': [2,4,5,10], 'min_samples_leaf': [1,2,4]},
                ]    

# Create and store grid search table (computationally expensive for some models such as svm)
# ----------------------------------------------
mlmodel.store_best_hyperparameter_selection(featuresets_dict=feat.feature_sets_dict, param_grid=param_grid, scale=True)
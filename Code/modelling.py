"""
@author: Monse Guedes Ayala
@project: Bioliberty EMG Classification Algorithm

Modelling part of the process. Includes classes for dataset, model, subject, and ML model.
Here, a subject can be defined from a dataset, and then a ml model can be built and applied
to the subject (including feature selection and similar procedures). 
"""

# Self-created imports 
from concurrent.futures import thread
from turtle import forward
import pre_processing as prep

# Python imports
from os import path
from os import listdir
from pandas import read_csv
from pandas import concat
import pandas as pd
import numpy as np
from itertools import product
from enum import Enum
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from time import process_time
import ast
import os

# ML imports
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn import metrics 
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import mutual_info_classif
from skfeature.function.similarity_based import fisher_score
from sklearn.feature_selection import SequentialFeatureSelector
import scikitplot as skplt
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold


class Dataset:
    """
    Class representing a whole dataset to be used.
   
    Attributes
    ----------
    folder_name: str
        name of the folder where the dataset is at
    path: str
        full relative path of the studied dataset
    file_name
        name of file where dataset is stored

    subjects_list
        list of all the unique subjects (by number)
    movements_list
        list of all the unique movements (by number)
    trials_list
        list of all the unique trials (by number)
    
    no_subjects : int
        the number assiged to the subject for which this data was collected
    no_trials : int
        the number of the trial of the data present in the file. There are several trials for same movement-subject
    no_movements : int
        the number assiged to the movement that the subject performed when collecting the file data
    no_channels : int
        number of channels recorded in this data file (usually the same for all subjects and movements)

    window_size
        size of windows
    overlap
        size of overlap
    channels_names
        names of all the channels
    channels_used
        channels that are actually used, this can be smaller than the number of channels if trying to simplify the model
    wavelet_level
        level of wavelet decomposition
    level_type
        type of wavelet used for decomposition

    Methods
    -------
    create_save_complete_dataframe(no_subject: int, window_size: int, overlap: int, features: list)
        creates and saves the dataset object, including a dataframe integrating all the files of a subject, and the chosen features (with a window size and overlap)

    """    

    def __init__(self,  dataset_name: str):
        """
        Creates an object for the dataset, and defines several attributes.

        dataset_name: folder name of the stantard dataset folder. This is inside datasets/standard_datasets, 
        and there is a folder for each dataset in standard format.
        """

        # Path-related attributes
        self.path = 'Datasets/standard_datasets'
        self.folder_name = dataset_name 
        self.file_name = None

        # Get files and folders names
        list_folders = [ name for name in path.join(self.path, self.folder_name) 
                        if path.isdir(path.join(self.path, self.folder_name, name))]    # Get all folders
        list_files = listdir(path.join(self.path, self.folder_name, 'S1'))    # Get all files names for first subject
        example_file = read_csv(path.join(self.path, self.folder_name, 'S1', list_files[0]))    # Get first file for first subject

        # Get list of all movements and trials
        all_movements_list = [int(''.join(filter(lambda i: i.isdigit(), file.split('_')[0]))) for file in list_files]    # Divide movement part from trial part in name and get movement numbers
        all_trials_list = [int(''.join(filter(lambda i: i.isdigit(), file.split('_')[1]))) for file in list_files]    # Divide movement part from trial part in name and get trial numbers

        # Store list of unique subjects, movements, and trials of the dataset.
        self.subjects_list = list(range(1,len(list_folders)))    #Get no. subjects i.e. no. folders 
        self.movements_list = list(set(all_movements_list))    #Get unique movements
        self.trials_list = list(set(all_trials_list))    #Get unique trials
        
        # Store attributes in numerical format
        self.no_subjects = len(self.subjects_list)    #no. of subject folders
        self.no_trials = len(self.movements_list)    #no. distinct values after '_' for any folder (first one)
        self.no_movements = len(self.trials_list)    #no. distinct numbers after 'mov' for any folder (first one)
        self.no_channels = len(example_file.columns)

        self.window_size = None
        self.overlap = None
        self.channels_names = None
        self.channels_used = None
        self.wavelet_level = None
        self.wavelet_type = None

    def create_save_complete_dataframe(self, no_subject: int, window_size: int, overlap: int, features: list, 
                                    channels_used: list, wavelet_level=4, wavelet_type='db1', overwrite=False):
        """
        Process dataset and store object.

        The reason behind this is that this is a long process with current code, and it is more efficient 
        for further analysis to do it only once and store results.
    
        no_subject: number of subject for which data will be stored
        window_size: size of window when windowing
        overlap: overlap of windows
        features: features that are to be generated for each window
        channels_used: list of positive integers, smaller than no_channels [1,2,3]
        wavelet_level: level of deomposition
        wavelet_type: type of wavelet
        overwrite: True if existing file is to be overwrite
        """

        # Update attributes
        self.window_size = window_size
        self.overlap = overlap
        self.channels_used = channels_used
        self.wavelet_level = wavelet_level
        self.wavelet_type = wavelet_type

        if len(channels_used) == self.no_channels:   
            self.channels_names = 'all_channels'    # Name when all channels are considered
        elif len(channels_used) == 1:    
            self.channels_names = ''.join(['channel_', str(channels_used[0])])    # Name when only one channel is considered, ex: channel_1
        else:
            self.channels_names = ''.join(['channels_', str(channels_used)])    # Name when a subset of channels is considered, ex: channels_[1,2,3]
        
        self.file_name = ''.join([self.channels_names, '_', 'subject_', str(no_subject), '_(', str(window_size), ',', str(overlap), 
                                ')_(',  str(self.wavelet_level), ',', self.wavelet_type, ')_data', '.obj'])   
        # example: all_channels_subject_1_(200,100)_(4,db1)_data.obj
        
        if path.exists(self.file_name) == False or overwrite == True:    # Create and store only if there is not already a file with same name, or if specifically indicated to overwrite current file
            subject = Subject(dataset= self, no_subject= no_subject)    # Create subject object with current dataset
            subject.prepare_data_model(window_size=window_size, overlap=overlap, features=features)    # Prepare the data for that subject
            filehandler = open(self.file_name, 'wb') 
            pickle.dump(subject, filehandler)    # Store the data in a file as an object 


class Subject:
    """
    Class representing a specific subject within the dataset.
    (it is for a single subject that the ML models are built and evaluated).
   
    Attributes
    ----------
    no_subject
        the number assosiated with the chosen subject
        
    dataset
        the dataset object from which the subject is taken 

    feature_dataframes_list
        list of a dataframe fr each file in a subject, including the extracted features
    complete_dataframe
        dataframe with all the files of a subject joint together (can have a selection of features, but currrently run with all possible features in main.py) 
    selected_dataframe
        a selection of features from complete dataframe 

    target_names
        name of the targets, in array format
    features_names 
        name of the features of the data formated for sklearn
    data 
        numpy array with feature matrix (correct format for sklearn)
    target 
        numpy array with targets (correct format for sklearn)

    Methods
    -------
    create_dataframes_list(window_size: int, overlap: int, features: list)
        creates a dataframe for each movement-trial pair within the studied subject
    concatenate_dataframes
        joins all the dataframes of each file together
    convert_data_format(features: list)
        converts the main dataframe (selecting only a subset of features) to sklearn format
    prepare_data_model(window_size: int, overlap: int, features: list)
        for this subject, a given window size, a given overlap, and a given set of features, create all dataframes and concatenate them, resulting in complete dataframe

    """    

    def __init__(self, dataset, no_subject: int):
        """
        dataset: dataset object
        no_subject: subject we want to model
        """

        self.no_subject = no_subject
        
        # Dataset
        self.dataset = dataset

        # Dataframes
        self.feature_dataframes_list = []
        self.complete_dataframe = None
        self.selected_dataframe = None

        # Main data attributes for scikit-learn
        self.target_names = None    # Array of strings with target labels
        self.features_names = None
        self.data = None            # numpy array with festure values
        self.target = None          # array with targets of each observation

    def create_dataframes_list(self, window_size: int, overlap: int, features: list):
        """
        Creates a dataframe for each movement-trial pair within the studied subject.

        It takes as inputs the window size, overlap, and the features that are to be
        extracted.
        """
        
        for trial, movement in product(self.dataset.trials_list, self.dataset.movements_list):    # Examine all possible files
            file = prep.File_data(self.dataset, self.no_subject, movement, trial, self.dataset.channels_used)    # Create file class object for each file
            # Create the file dataframe, normalize it, perform windowing, and extract the chosen features
            file.process_file(window_size= window_size, overlap= overlap, features= features, wavelet_level=self.dataset.wavelet_level, wavelet_type=self.dataset.wavelet_type)    
            self.feature_dataframes_list.append(file.feature_dataframe)    # Add the dataframe to a list of the subject's dataframes
            print('The file for movement {} and trial {} is ready'.format(movement, trial))    # To check approximate speed of the process.

        return self.feature_dataframes_list

    def concatenate_dataframes(self):
        """
        Horizobtally joins all the dataframes that are stored in the subject class (they
        already have a target assigned to them)
        """

        self.complete_dataframe = concat(self.feature_dataframes_list, ignore_index=True)
        
        return self.complete_dataframe

    def convert_data_format(self, features: list):
        """
        Converts the complete dataframe, after reducing it to the desired features, to the standard 
        format accepted by sklearn, which is: numpy matrix with all features called data, and numpy 
        array with all targets called target.
        """
        
        # Get a list of all the columns that contain any of the desired features in their name
        # (recall that columns are currently of the form channel_1_variance, so the feature name 
        # is cotained in the column name):
        features_names = [column for column in self.complete_dataframe.columns if any((feature == column.split('_', maxsplit=2)[2])    # Either the input feature is cointained in the column name
                        or (feature == column) for feature in features)]    # or it is exactly the column name
        column_names= features_names + ['target']    # Include the target column into extacted columns again
        
        self.selected_dataframe = self.complete_dataframe[column_names]    #Filter the complete dataframe to include only desired features
        self.features_names = features_names
        self.data = self.selected_dataframe.drop('target', axis=1).to_numpy()    # Get feature matrix in correct format
        self.target = np.array(self.selected_dataframe['target'])    # Get target array in correct format

        return self.data, self.target

    def prepare_data_model(self, window_size: int, overlap: int, features: list):    # Only prepare whole dataframe
        """
        For the class subject, this function creates the list with a dataframe per file, and 
        concatenates it to get the complete dataframe.
        """
        
        self.create_dataframes_list(window_size= window_size, overlap= overlap, features= features)
        self.concatenate_dataframes()   # This will give us the 'complete_dataframe' attribute. This is a dataframe ordered as follows:
                                        # as columns: all features for all channels, as rows: all windows for trial 1 movement 1, all 
                                        # windows for trial 1 and the rest of movements (ordered), all windows for trial 2 and all movents 
                                        # (ordered), and so on for all the trials. This is built in such a way than we are then able to 
                                        # split the training and test sets into the first four trials for training and the last two for testing. 
                        
        return self.complete_dataframe


class ML_Model:
    """
    Class representing a machine learning model.
   
    Attributes
    ----------
    type
        name of the ML model 
    subject
        subject class object 
    hyperparameters
        dictonary with hyperparameters to be used for the model 
    channels_used
        channels that are used for this model 

    dataframe 
        dataframe for experiment results: feature set used, scores, and hyperparameters information
    hyperparameter_dataframe
        dataframe with grid search results 

    featuresets_names
        names of the different feature sets considered 
    featuresets_dict
        dictionary that includes the name of the feature sets, and the list of features they are associated to
    forward_selection
        dictionary with the results of forward selection
    backward_elimination
        dictionary with the results of backward elimination  

    pca
        string stating whether PCA was used or not, to late rinclude in files names 
    scale
        string stating whether data was scaled or not, to late rinclude in files names
    

    Methods
    -------
    build_model
    fit_model(feature_set, pca= False, n_components=2, scale= True, plots=False)
    create_featuresets_dataframe(featuresets_dict:dict, pca= False, n_components=2, scale= True, plots=False)
    store_results_dataframe
    store_best_hyperparameter_selection(featuresets_dict:dict, param_grid, window_size: int, overlap: int, featuresets_file_name: str, pca= False, scale= False)

    ROC_curve( X_train, y_train, X_test, y_test)
    pca_variance(feature_set)
    pca_2d_projection(feature_set)
    confusion_matrix(y_test, y_predict, feature_set)
    plot_pca

    information_gain_selection(self, feature_set)
    filtered_information_gain_selection(self, feature_set, no_filter=10)
    fisher_score(self, feature_set)
    filtered_fisher_score(self, feature_set, no_filter=10)
    correlation_matrix(self, feature_set: list)
    rf_feature_importance(self, feature_set)
    filtered_correlation_matrix(self, feature_set, threshold=0.5)

    feature_selection_plots(self)
    forward_feature_selection(self)
    backward_feature_selection(self)
    """    

    def __init__(self, subject: Subject, model_name: str, hyperparameters: dict):
        """
        subject: subject object from Subject class
        model_name: either svc, lda, or random_forest
        hyperparameyers: dictionary with hyperparameters
        """

        self.type = model_name
        self.subject = subject
        self.hyperparameters = hyperparameters
        self.channels_used = subject.dataset.channels_used

        self.dataframe = None
        self.hyperparameter_dataframe = None

        self.featuresets_names = None
        self.featuresets_dict = None
        self.forward_selection = None
        self.backward_elimination = None

        self.pca = 'no_pca'
        self.scale = 'no_scale'

        # Always build the model if the object is created
        self.build_model()
    
    def build_model(self):
        """
        Builds the ML model with desired parameters
        """

        #SVC model
        if self.type == 'svc':
            if self.hyperparameters['kernel'] == 'linear':
                svc = SVC(kernel= self.hyperparameters['kernel'], C= self.hyperparameters['C'])    # Parameters for linear kernel
            else:
                svc = SVC(kernel= self.hyperparameters['kernel'], C= self.hyperparameters['C'], gamma= self.hyperparameters['gamma'])    # Parameters for rest of kernels
            self.model = svc

        #LDA model
        if self.type == 'lda':
            lda = LinearDiscriminantAnalysis()
            self.model = lda

        #Random Forest model
        if self.type == 'random_forest':
            rf = RandomForestClassifier(n_estimators=self.hyperparameters['n_estimators'], 
                            max_depth=self.hyperparameters['max_depth'],
                            min_samples_leaf=self.hyperparameters['min_samples_leaf'] , random_state=0)
            self.model = rf

        return self.model  

    def fit_model(self, feature_set, pca= False, n_components=0.95, scale= True, plots=False, cross_validation=False):
        """
        Fits the machine learning model into the desired data.

        Depends on the feature set, whether PCA is applied or not, whether data is scaled
        ot not, whether cross validation is applied, and whether plots (confusion matrix) 
        are to be generated.
        """
                
        #Prepare data with specific features
        self.subject.convert_data_format(feature_set)

        #Split training and testing
        X_train, X_test, y_train, y_test = train_test_split(
                self.subject.data, self.subject.target, train_size=0.666666666, shuffle=False)      # There are 4 trials for training and 2 for testing (ordered), 
                                                                                                    # therefore the 0.66666... However, there are more efficnet ways 
                                                                                                    # to divide the trials into training and testing, I just this this 
                                                                                                    # one because was straightforward.

        if scale == True: # In reality, scale is almost always preferred. 
            # Scale the data using MinMaxScaler (other scalers could also be applied)
            scaler = MinMaxScaler()
            scaler.fit(X_train)     # The scaler is fitted into the X_train data instead of whole data so that the test data does not have information from 
                                    # the training data (this is something to keep in ming when preprocessing the signal as well, since it can slightly skew 
                                    # classification results).
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
            self.scale = 'scale'    # Change scale attribute to reflect that data is scaled when storing files

        if pca == True:    # Apply PCA
            fit_time_start = process_time()    # Start counting PCA + fitting time
            pca_reduction = PCA(n_components=n_components, random_state=0)
            pca_reduction.fit(X_train)
            X_train = pca_reduction.transform(X_train)    # Transform train data to principal components
            X_test = pca_reduction.transform(X_test)    # Transform test data to principal components
            self.pca = 'pca'    # Change PCA attribute to reflect that data is scaled when storing files

            # Fit the model with PCA and predict test data
            self.model.fit(X_train, y_train)
            fit_time_end = process_time()    # End counting CPU time
            y_pred = self.model.predict(X_test)    # Get predicted values
            
            if plots== True:
                #Create and store confusion matrix
                self.confusion_matrix(y_test, y_pred, feature_set)
                
            # Get scores of model with PCA
            training_score = round(self.model.score(X_train, y_train), 4)    # Round the training score to 4 decimal places
            test_score = round(self.model.score(X_test, y_test), 4)    # Round the test score to 4 decimal places
            score = {'test_score': test_score, 'training_score': training_score}

        else:    # This is the case without PCA
            fit_time_start = process_time()    # Start counting CPU time
            self.model.fit(X_train, y_train)    # Fit the model 
            fit_time_end = process_time()    # Stop counting time
            y_pred = self.model.predict(X_test)    # Predict all test points

            if plots== True:
                self.confusion_matrix(y_test, y_pred, feature_set)    # Create and store confusion matrix
                
            # Get scores 
            training_score = round(self.model.score(X_train, y_train), 4)
            test_score = round(self.model.score(X_test, y_test), 4)

            if cross_validation == True:    
                # In this case a cross validation score is added to the dictionary of scores. Now, there are several 
                # problems with this that I didn't have time to account for. First, for SVC and other models the data 
                # must be scaled. However, the scaler depends on the trainig data, which means that scaling must be 
                # done at each fold. The problem with the built in functions is that you need a pipeline for this. 
                # Here there is an example of how a pipeline would look like. The other problem is that the training 
                # and testing data must be divided in such a way that different trials are not mixed, and there is 
                # equal training for each movement. Possible solutions to this are to manually build a cross validation 
                # function. I didn't have time to do this, but it shouldn't be very complicated. It is always a good 
                # practice to use cross-validation for model evaluation. Also, cross-validation slows down this function,
                # so should only be used when actually evaluating performance of the model.

                # (nonworking) Example of pipeline:
                scalar = MinMaxScaler()
                pipeline = Pipeline([('transformer', scalar), ('estimator', self.model)])
                cv = KFold(n_splits=3)
                cross_score = cross_val_score(pipeline, self.subject.data, self.subject.target, cv=cv)
                score = {'cross_val_score': cross_score.mean(), 'test_score': test_score, 'training_score': training_score}
            
            else:    # When cross-validation is not applied
                score = {'test_score': test_score, 'training_score': training_score}

        return score, fit_time_end - fit_time_start    # Returns score dictionary and CPU time of fitting model

    def create_featuresets_dataframe(self, featuresets_dict: dict, pca=False, n_components=0.95, scale= True, plots=False, cross_validation=True):
        """
        Create dataframe with results of the model.

        featuresets_dict: distionary with the names of the feature sets, and the features they contain.
        pca: whether PCA is applied or not
        n_components: components of PCA
        scale: whether data is scaled before fitting
        plots: whether confusion matrices are generated 
        cross-validation: whether cross-validation is used 
        """

        self.featuresets_names = featuresets_dict.keys()    # Get and add list of feature sets names to attribute
        self.featuresets_dict = featuresets_dict    # Add the featureset dict to attribute
    
        scores, cpu = map(list,zip(*[self.fit_model(feature_set, pca=pca, n_components=n_components, scale=scale, plots=plots, cross_validation=cross_validation) 
                                    for feature_set in featuresets_dict.values()]))    # Get scores and CPU when running the fit function with all the feature sets
        hyperparameters = [str(self.hyperparameters) for feature_set in featuresets_dict.values()]    #List of hyperparameters dictionaries
        
        #Create dataframe with results to store it in excel file with store_results_dataframe function, or use for something else
        self.dataframe = pd.DataFrame(data= {'Scores': scores, 'Hyperparameters': hyperparameters, 'cpu': cpu}, index= self.featuresets_names)

        return self.dataframe
        
    def store_results_dataframe(self):
        """
        Stores dataframe with results of different featuresets.
        """

        path_name = ''.join(['Code/Results/', str(self.subject.dataset.window_size), '_', str(self.subject.dataset.overlap), '_window/', 
                            str(self.subject.dataset.wavelet_level), '_', self.subject.dataset.wavelet_type, '/'])
        # format example: Code/Results/200_100_window/2_db1/

        file_name = ''.join([self.type, '_', self.subject.dataset.channels_names,  '_', str(self.featuresets_names), '_', 
                str(self.hyperparameters), '_', self.pca, '_',  self.scale, '_results.xlsx'])    
        # Output format example: svc_channels_[time,time_freq,all]_{C: 10, kernel: 'linear'}_no_pca_scale_results.xlsx

        os.makedirs(os.path.dirname(path_name + file_name), exist_ok=True)    # If such directory does not exist, create it

        # Save dataframe to excel
        writer = pd.ExcelWriter(''.join([path_name, file_name]))    
        self.dataframe.to_excel(writer)
        writer.save()

    def store_best_hyperparameter_selection(self, featuresets_dict:dict, param_grid, window_size: int, overlap: int, featuresets_file_name: str, pca= False, scale= True):
        """
        Performs grid search to find the best possible hyperparameters, 
        and stores results in an excel.
        TODO: fix, remove window,etc, compact lists
        """

        self.featuresets_names = featuresets_dict.keys() #List of features names
        self.featuresets_dict = featuresets_dict
        scores = [] #List of misclassification scores
        hyperparameters = [] #List of dictionaries
        
        for feature_set in featuresets_dict.values():
            #print('Preparing data for feature set {}'.format(feature_set))
            #Prepare data with specific features
            self.subject.convert_data_format(feature_set)

            X_train, X_test, y_train, y_test = train_test_split(
                    self.subject.data, self.subject.target, random_state=0)    

            if self.type == 'svc':
                #cv=[(slice(None), slice(None))] for no cross-validation
                grid_search = GridSearchCV(SVC(), param_grid, scoring='accuracy', cv=2)          
            elif self.type == 'lda':
                grid_search = GridSearchCV(LinearDiscriminantAnalysis(), param_grid, scoring='accuracy', cv=2)
            elif self.type == 'random_forest':
                grid_search = GridSearchCV(RandomForestClassifier(), param_grid, scoring='accuracy', cv=2)
            
            if scale == True:
                # scale the data using MinMaxScaler
                scaler = MinMaxScaler()
                scaler.fit(X_train)
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)
                self.scale = 'scale'

            if pca== True:
                pca = PCA(n_components=5, whiten=True, random_state=0).fit(X_train)
                X_train = pca.transform(X_train)
                X_test = pca.transform(X_test)
                self.pca = 'pca'

            grid_search.fit(X_train, y_train)

            #Create dictionary with best hyperparameters and best score for cross-validation
            best_hyperparameter = grid_search.best_params_
            best_hyperparameter['best_score'] = grid_search.best_score_
            hyperparameters.append(str(best_hyperparameter)) #Add to empty list

            #Create dictionary with training score and tests core
            y_pred = grid_search.predict(X_test) #Call predict on the estimator with the best found parameters
            training_score = round(grid_search.score(X_train, y_train), 4)
            test_score = round(grid_search.score(X_test, y_test), 4)
            score = {'test_score': test_score, 'training_score': training_score}
            scores.append(str(score)) #Add to empty list

        #Create dataframe with results and store it in excel file.
        self.hyperparameter_dataframe = pd.DataFrame(data= {'Scores': scores, 'Hyperparameters': hyperparameters}, index= self.featuresets_names)
            
        path_name = 'Code/Results/' + str(window_size) + '_' + str(overlap) + '_window/'
        os.makedirs(os.path.dirname(path_name), exist_ok=True)
        writer = pd.ExcelWriter(path_name + self.type + '_best_hyperparameters' + featuresets_file_name + self.pca + self.scale + '_results.xlsx')
        self.hyperparameter_dataframe.to_excel(writer)
        writer.save()
    

    # PCA plots
    def pca_variance(self, feature_set):
            """
            Plots the explained variance of the PCA.

            Explained variance refers to the variance explained by each of 
            the principal components (eigenvectors).
            """
            
            pca = PCA(random_state=1)    # Create PCA object
            pca.fit(self.subject.data)    # Fit PCA to data
            skplt.decomposition.plot_pca_component_variance(pca, figsize=(8,6))    # Plot PCA component variance

            featureset_name = list(self.featuresets_dict.keys())[list(self.featuresets_dict.values()).index(feature_set)]   # Get name of the feature set for which the PCA plot was produced by looking 
                                                                                                                            # its position in dictionary with sets and their names
            
            full_path = ''.join(['Plots/', str(self.subject.dataset.window_size), '_', str(self.subject.dataset.overlap), '_window/',
                            str(self.subject.dataset.wavelet_level), '_', self.subject.dataset.wavelet_type, '/variance_PCA/'])   
            # Example: Plots/200_100_window/2_db1/variance_PCA/
            
            file_name = ''.join(['pca_variance_', self.subject.dataset.channels_names, '_', featureset_name, '_', self.scale, '.png'])
            # Example: pca_variance_[1,2,3,4,5,6,7]_featureset_scale.png

            os.makedirs(os.path.dirname(full_path), exist_ok=True)    # If path does not exist, create it

            plt.savefig(full_path + file_name, bbox_inches='tight', dpi=200)    # Save figure
            plt.clf()

    def pca_2d_projection(self, feature_set):
        """
        Plots the projection of the first two components,
        in relation to the classes.
        """
        
        pca = PCA(random_state=1)
        pca.fit(self.subject.data)
        skplt.decomposition.plot_pca_2d_projection(pca, self.subject.data, self.subject.target,    # Create 2D PCA projection for data
                                        figsize=(10,10),
                                        cmap="tab10")

        featureset_name = list(self.featuresets_dict.keys())[list(self.featuresets_dict.values()).index(feature_set)]    # Get the name of the feaature set (features dict key for a given list of features)
        
        full_path = ''.join(['Plots/', str(self.subject.dataset.window_size), '_', str(self.subject.dataset.overlap), '_window/',
                            str(self.subject.dataset.wavelet_level), '_', self.subject.dataset.wavelet_type, '/2d_PCA/'])    
        # Example: Plots/200_100_window/2_db1/2d_PCA/

        file_name = ''.join(['pca_2d_', self.subject.dataset.channels_names, '_', featureset_name, '_', self.scale, '.png'])
        # Example: pca_2d_coefficients_channels_featureset_scale.png
        
        os.makedirs(os.path.dirname(full_path), exist_ok=True)    # Create directory if necessary

        plt.savefig(full_path + file_name, bbox_inches='tight', dpi=200)    # Save figure
        plt.clf()
    
    def plot_pca(self, featuresets_dict:dict):
            """
            To plot all PCA plots for all the feature sets. 
            """

            for feature_set in featuresets_dict.values():
                # Convert data to include only selected features
                self.subject.convert_data_format(feature_set)
                # Feature selection plots
                self.pca_variance(feature_set=feature_set)
                self.pca_2d_projection(feature_set=feature_set)

    def plot_PCA_improvement(self, no_features, feature_set_name, n_components):
        """
        TODO: fill
        """
        full_path = ''.join(['Code/Results/', str(self.subject.dataset.window_size), '_', str(self.subject.dataset.overlap), '_window/', 
                            str(self.subject.dataset.wavelet_level), '_', self.subject.dataset.wavelet_type, '/'])  
        # Example: Code/Results/200_100_window/2_db1/
        
        file_name = ''.join([str(no_features), '_forwardselection_lda_', self.subject.dataset.channels_names, '_',    # This is the name of the file containing forward selection results
                '[]', '_', self.pca, '_',  self.scale, '_results.xlsx'])
        # Example: 35_forwardselection_lda_all_channels_[]_pca_scale_results.xlsx            

        forward_dataframe = pd.read_excel(full_path + file_name)    # Make a dataframe with forward selection results (which were obtained usding forward selection function)

        pca_scores, cpu = map(list,zip(*[self.fit_model(list(ast.literal_eval(feature_set)), pca= True, n_components=n_components, cross_validation=False)    # Get the scores for the model after applying PCA using the features that were selected during forward selection
                            for feature_set in forward_dataframe['features']]))

        forward_dataframe['no_features'] = forward_dataframe['no_features'].astype(int)    # Convert column with number of features to integer
        pca_scores = [score_dict['test_score'] for score_dict in pca_scores]    # Create list with all the scores of the model after PCA

        plt.figure(figsize=(20,15))
        plt.plot(forward_dataframe['no_features'], pca_scores, marker= 'o', color='blue')    # Plots scores of PCA
        plt.plot(forward_dataframe['no_features'],cpu, marker='x', color='darkgreen')    # Plot cPU time of PCA + fitting
        plt.title('Evolution PCA (0.95 variance components) for ' + str(self.subject.dataset.window_size) + '-' + str(self.subject.dataset.overlap) + ' Windowing')
        plt.legend(['Accuracy Score', 'CPU time fitting PCA and the model'])
        plt.xticks(forward_dataframe['no_features'])    # x ticks as no. of features
        plt.axhline(y=pca_scores[-1], color='r', linestyle='dashed')    # Add horizontal line for last value of the plot (maximum number of features considered)
        plt.yticks([0.1 * i for i in range(1,11)])

        full_path = ''.join(['Plots/', str(self.subject.dataset.window_size), '_', str(self.subject.dataset.overlap), '_window/', 
                            str(self.subject.dataset.wavelet_level), '_', str(self.subject.dataset.wavelet_type) ,'/PCA_evolution/'])
        # Example: Plots/200_100_window/2_db1/PCA_evolution/

        file_name = ''.join([str(no_features), '_PCA_evolution_', self.type, '_', self.subject.dataset.channels_names, '_', feature_set_name, '.png'])
        # Example: 35_PCA_evolution_lda_channels_featureset.png
        
        os.makedirs(os.path.dirname(full_path), exist_ok=True)    # Create dictionary if not already there

        plt.savefig(full_path + file_name, bbox_inches='tight', dpi=200)    # Save figure
        plt.clf()

        pca_dataframe = pd.DataFrame({'no_features': forward_dataframe['no_features'],    # Create dataframe with the results from the plot to store as excel file
                                        'score': pca_scores,
                                        'cpu': cpu,
                                        'features': forward_dataframe['features']
                                        })

        path_name = ''.join(['Code/Results/', str(self.subject.dataset.window_size), '_', str(self.subject.dataset.overlap), '_window/', 
                            str(self.subject.dataset.wavelet_level), '_', self.subject.dataset.wavelet_type, '/'])
        # Example: Code/Results/200_100_window/2_db1/
        
        file_name = ''.join([str(len(forward_dataframe['no_features'])), '_pcaresults_', self.type, '_', self.subject.dataset.channels_names,  '_', str(self.featuresets_names), '_', 
                str(self.hyperparameters), '_',  self.scale, '_results.xlsx'])    
        # Example: no_features_pcaresults_svc_[1,2,3]_all_features_[]_scale_results.xlsx

        os.makedirs(os.path.dirname(path_name + file_name), exist_ok=True)    # Create directory if not already there

        writer = pd.ExcelWriter(''.join([path_name, file_name]))    # Save dataframe as excel file
        pca_dataframe.to_excel(writer)
        writer.save()    
    
    
    #Plotting results
    def confusion_matrix(self, y_test, y_predict, feature_set):
        """
        Plots two different confusion matrices plot:
        one showing count of misclassified points, and 
        another one showing percentages.
        """
        
        fig = plt.figure(figsize=(15,6))
        ax1 = fig.add_subplot(121)
        skplt.metrics.plot_confusion_matrix(y_test, y_predict,    # Confusion matrix with count of values
                                            title="Confusion Matrix",
                                            cmap="Oranges",
                                            ax=ax1)

        ax2 = fig.add_subplot(122)
        skplt.metrics.plot_confusion_matrix(y_test, y_predict,     # Confusion matrix with percentages
                                            normalize=True,
                                            title="Confusion Matrix",
                                            cmap="Purples",
                                            ax=ax2)

        featureset_name = list(self.featuresets_dict.keys())[list(self.featuresets_dict.values()).index(feature_set)]    # Get the name of the used feature set from directory
        
        full_path = ''.join(['Plots/', str(self.subject.dataset.window_size), '_', str(self.subject.dataset.overlap), '_window/',
                            str(self.subject.dataset.wavelet_level), '_', self.subject.dataset.wavelet_type, '/confusion_matrix/', self.type, '/'])
        # Example: Plots/200_100_window/2_db1/confusion_matrix/svc/

        file_name = ''.join(['matrix_', self.subject.dataset.channels_names, '_', featureset_name, '_', str(self.hyperparameters), '_', self.pca, '_', self.scale, '.png'])
        # Example: matrix_channels_featureset_{C: 10, kernel: 'linear'}_no_pca_scale.png

        os.makedirs(os.path.dirname(full_path), exist_ok=True)    # Create directory if not there

        plt.savefig(full_path + file_name, bbox_inches='tight', dpi=200)    # Save figure
        plt.clf()

    def plot_forward_selection_true_results(self, no_features, feature_set_name):
        """TODO: fill and comment
        """
        
        full_path = ''.join(['Code/Results/', str(self.subject.dataset.window_size), '_', str(self.subject.dataset.overlap), '_window/', 
                            str(self.subject.dataset.wavelet_level), '_', self.subject.dataset.wavelet_type, '/'])  
        # Example: Code/Results/200_100_window/2_db1/

        file_name = ''.join([str(no_features), '_forwardselection_lda_', self.subject.dataset.channels_names, '_', 
                '[]', '_', self.pca, '_',  self.scale, '_results.xlsx'])  
        # Example: 35_forwardselection_lda_all_channels_[]_pca_scale_results.xlsx

        forward_dataframe = pd.read_excel(full_path + file_name)    # Load dataframe with forward selection results

        true_scores, cpu = map(list,zip(*[self.fit_model(list(ast.literal_eval(feature_set)), pca= False, cross_validation=False)    # Fit the model with feature set and get score and CPU time of fitting
                            for feature_set in forward_dataframe['features']]))

        forward_dataframe['no_features'] = forward_dataframe['no_features'].astype(int)    # Convert no_features column to integers
        true_scores = [score_dict['test_score'] for score_dict in true_scores]    # Make list of scores

        plt.figure(figsize=(20,15))
        plt.plot(forward_dataframe['no_features'], true_scores, marker= 'o', color='blue')
        plt.plot(forward_dataframe['no_features'],cpu, marker='x', color='darkgreen')
        plt.title('Test set scores for ' + str(self.subject.dataset.window_size) + '-' + str(self.subject.dataset.overlap) + ' windowing')
        plt.legend(['Accuracy Score', 'CPU time fitting the model'])
        plt.xticks(forward_dataframe['no_features'])
        plt.axhline(y=true_scores[-1], color='r', linestyle='dashed')
        plt.yticks([0.1 * i for i in range(1,11)])

        full_path = ''.join(['Plots/', str(self.subject.dataset.window_size), '_', str(self.subject.dataset.overlap), '_window/', 
                            str(self.subject.dataset.wavelet_level), '_', str(self.subject.dataset.wavelet_type) ,'/test_scores/'])
        # Example: Plots/200_100_window/2_db1/test_scores/

        file_name = ''.join([str(no_features), '_test_score_', self.type, '_', self.subject.dataset.channels_names, '_', feature_set_name, '.png'])
        # Example: 35_test_score_svc_all_channels_all_features.png
        
        os.makedirs(os.path.dirname(full_path), exist_ok=True)    # Create directory if not there

        plt.savefig(full_path + file_name, bbox_inches='tight', dpi=200)    # Save figure
        plt.clf()

        pca_dataframe = pd.DataFrame({'no_features': forward_dataframe['no_features'],    # Create dataframe with values from plot to store in excel file
                                        'score': true_scores,
                                        'cpu': cpu,
                                        'features': forward_dataframe['features']
                                        })

        path_name = ''.join(['Code/Results/', str(self.subject.dataset.window_size), '_', str(self.subject.dataset.overlap), '_window/', 
                            str(self.subject.dataset.wavelet_level), '_', self.subject.dataset.wavelet_type, '/'])
        # Example: Code/Results/200_100_window/2_db1/
        
        file_name = ''.join([str(len(forward_dataframe['no_features'])), '_test_scores_', self.type, '_', self.subject.dataset.channels_names,  '_', str(self.featuresets_names), '_', 
                str(self.hyperparameters), '_',  self.scale, '_results.xlsx'])    
        # Example: 35_test_scores_svc_all_channels_all_features_[]_scale_results.xlsx

        os.makedirs(os.path.dirname(path_name + file_name), exist_ok=True)    # Create directory if not already there

        writer = pd.ExcelWriter(''.join([path_name, file_name]))     # Save file
        pca_dataframe.to_excel(writer)
        writer.save()    


    # Feature selection plots
    def information_gain_selection(self, feature_set):
        """
        Information gain calculates the reduction in entropy from the transformation 
        of a dataset. It can be used for feature selection by evaluating the Information 
        gain of each variable in the context of the target variable.
        """
        
        # Get reduction in entropy for all features as importane indicator
        importances = mutual_info_classif(self.subject.data, self.subject.target)
        #Create a pandas series with importances as values, indexed by feature name
        feat_importances = pd.Series(importances, self.subject.selected_dataframe.columns[0:len(self.subject.selected_dataframe.columns)-1])

        no_features = len(self.subject.dataset.channels_used) * len(feature_set)    # Get the total number of features to scale plot size by this
        featureset_name = list(self.featuresets_dict.keys())[list(self.featuresets_dict.values()).index(feature_set)]    # Get name of feature set to add to title

        # Create plot
        feat_importances.plot(kind='barh', color='lightcoral', figsize=(1 * no_features, 1 * no_features))
        plt.title('Information Gain of ' + featureset_name.replace('_',' ').title())

        full_path = ''.join(['Plots/', str(self.subject.dataset.window_size), '_', str(self.subject.dataset.overlap), '_window/', 
                            str(self.subject.dataset.wavelet_level), '_', self.subject.dataset.wavelet_type, '/information_gain/'])
        # Example: Plots/200_200_window/2_db2/information_gain/
        
        file_name = ''.join(['information_gain_', self.subject.dataset.channels_names, '_', featureset_name, '_', self.pca, '_', self.scale, '.png'])    
        # Example: information_gain_channels_featureset_no_pca_scale.png

        os.makedirs(os.path.dirname(full_path), exist_ok=True)    # Create dictionary if not already there

        plt.savefig( full_path + file_name, bbox_inches='tight', dpi=200)    # Save plot
        plt.clf()

    def filtered_information_gain_selection(self, feature_set, no_filter=10):
        """
        Information gain calculates the reduction in entropy from the transformation 
        of a dataset. It can be used for feature selection by evaluating the information 
        gain of each variable in the context of the target variable.
        This version returns top 20, or other chosen number, features.
        """

        # Get reduction in entropy for all features as importane indicator
        importances = mutual_info_classif(self.subject.data, self.subject.target)
        # Create a pandas series with importances as values, indexed by feature name
        feat_importances = pd.Series(importances, self.subject.selected_dataframe.columns[0:len(self.subject.selected_dataframe.columns)-1])
        # Select only the largest values
        feat_importances =  feat_importances.nlargest(no_filter)
        featureset_name = list(self.featuresets_dict.keys())[list(self.featuresets_dict.values()).index(feature_set)]

        # Create plot
        feat_importances.plot(kind='barh', color='brown', figsize=(10, 10))
        plt.title('Top ' + str(no_filter) +  ' Information Gain of ' + featureset_name.replace('_',' ').title())
    
        full_path = ''.join(['Plots/', str(self.subject.dataset.window_size), '_', str(self.subject.dataset.overlap), '_window/', 
                            str(self.subject.dataset.wavelet_level), '_', self.subject.dataset.wavelet_type, '/information_gain/'])
        # Example: Plots/200_100_window/2_db1/information_gain/

        file_name = ''.join(['filtered_information_gain_', self.subject.dataset.channels_names, '_', featureset_name, '_', self.pca, '_', self.scale, '.png'])
        # Example: filtered_information_gain_channels_featureset_no_pca_scale.png

        os.makedirs(os.path.dirname(full_path), exist_ok=True)    # Create directory if not already there

        plt.savefig(full_path + file_name, bbox_inches='tight', dpi=200)    # Save figure
        plt.clf()

    def fisher_score(self, feature_set):
        """
        Returns the ranks of the variables based on the fisher’s score in descending 
        order. We can then select the variables as per the case.
        """

        # Calculate fisher's scores
        ranks = fisher_score.fisher_score(self.subject.data, self.subject.target)
        # Create a pandas series with scores as values and feature names as index
        feat_importances = pd.Series(ranks, self.subject.selected_dataframe.columns[0:len(self.subject.selected_dataframe.columns)-1])

        no_features = len(self.subject.dataset.channels_used) * len(feature_set)
        featureset_name = list(self.featuresets_dict.keys())[list(self.featuresets_dict.values()).index(feature_set)]

        # Create plot
        feat_importances.plot(kind='barh', color='royalblue', figsize=(1 * no_features, 1 * no_features))
        plt.title('Fisher\'s Score of ' + featureset_name.replace('_',' ').title())

        full_path = ''.join(['Plots/', str(self.subject.dataset.window_size), '_', str(self.subject.dataset.overlap), '_window/', 
                            str(self.subject.dataset.wavelet_level), '_', self.subject.dataset.wavelet_type, '/fisher_score/'])
        # Example: Plots/200_100_window/2_db1/fisher_score/
        
        file_name = ''.join(['fisher_score_', self.subject.dataset.channels_names, '_', featureset_name, '_', self.pca, '_', self.scale, '.png'])
        # Example: fisher_score_channels_featureset_no_pca_scale

        os.makedirs(os.path.dirname(full_path), exist_ok=True)    # Create directory if not already there

        plt.savefig(full_path + file_name, bbox_inches='tight', dpi=200)    # Save figure
        plt.clf()

    def filtered_fisher_score(self, feature_set, no_filter=10):
        """
        Returns the ranks of the variables based on the fisher’s score in descending 
        order. We can then select the variables as per the case. This version returns 
        top 20, or other chosen number, features.
        """

        # Calculate fisher's scores
        ranks = fisher_score.fisher_score(self.subject.data, self.subject.target)
        # Create a pandas series with scores as values and feature names as index
        feat_importances = pd.Series(ranks, self.subject.selected_dataframe.columns[0:len(self.subject.selected_dataframe.columns)-1])
        # Select largest values
        feat_importances =  feat_importances.nlargest(no_filter)

        featureset_name = list(self.featuresets_dict.keys())[list(self.featuresets_dict.values()).index(feature_set)]

        # Create plot
        feat_importances.plot(kind='barh', color='midnightblue', figsize=(10, 10))
        plt.title('Top ' + str(no_filter) +  ' Fisher\'s Score of ' + featureset_name.replace('_',' ').title())

        full_path = ''.join(['Plots/', str(self.subject.dataset.window_size), '_', str(self.subject.dataset.overlap), '_window/', 
                            str(self.subject.dataset.wavelet_level), '_', self.subject.dataset.wavelet_type, '/fisher_score/'])
        # Example: Plots/200_100_window/2_db1/fisher_score/

        file_name = ''.join(['filtered_fisher_score_', self.subject.dataset.channels_names, '_', featureset_name, '_', self.pca, '_', self.scale, '.png'])
        # Example: filtered_fisher_score_channels_featureset_no_pca_scale

        os.makedirs(os.path.dirname(full_path), exist_ok=True)    # Create directory if not already there

        plt.savefig(full_path + file_name, bbox_inches='tight', dpi=200)    # Save figure
        plt.clf()

    def correlation_matrix(self, feature_set: list):
        """
        Correlation matrix of the features
        """
        
        # Create correlation matrix
        corr = self.subject.selected_dataframe.corr()

        no_features = len(self.subject.dataset.channels_used) * len(feature_set)
        featureset_name = list(self.featuresets_dict.keys())[list(self.featuresets_dict.values()).index(feature_set)]

        # Plotting heatmap
        plt.figure(figsize=(1.6 * no_features, 1.2 * no_features))
        sns.heatmap(corr, annot= True)
        plt.title('Correlation Matrix of ' + featureset_name.replace('_',' ').title())

        full_path = ''.join(['Plots/', str(self.subject.dataset.window_size), '_', str(self.subject.dataset.overlap), '_window/', 
                            str(self.subject.dataset.wavelet_level), '_', self.subject.dataset.wavelet_type, '/correlation_matrix/'])
        # Example: Plots/200_100_window/2_db1/correlation_matrix/

        file_name = ''.join(['correlation_matrix_', self.subject.dataset.channels_names, '_', featureset_name, '_', self.pca, '_', self.scale, '.png'])
        # Example: correlation_matrix_channels_featureset_no_pca_scale.png

        os.makedirs(os.path.dirname(full_path), exist_ok=True)    # Create directory if not already there

        plt.savefig(full_path + file_name, bbox_inches='tight', dpi=200)    # Save figure
        plt.clf()
    
    def filtered_correlation_matrix(self, feature_set, threshold=0.5):
            """
            Correlation matrix of the features, showing only those bigger than a 
            threshold.
            """
            
            # Create correlation matrix
            corr = self.subject.selected_dataframe.corr().abs()    # Get absolute value matrix

            # Get features that are highly correlated to each other
            highly_correlated = corr[corr > threshold]

            no_features = len(self.subject.dataset.channels_used) * len(feature_set)
            featureset_name = list(self.featuresets_dict.keys())[list(self.featuresets_dict.values()).index(feature_set)]
            
            # Plotting heatmap
            plt.subplots(figsize=(1.6 * no_features, 1.2 * no_features))
            sns.heatmap(highly_correlated, annot=True, fmt='.2f')
            plt.title('Correlation Matrix of Scores Greater than ' + str(threshold) +  ' for ' + featureset_name.replace('_',' ').title())

            full_path = ''.join(['Plots/', str(self.subject.dataset.window_size), '_', str(self.subject.dataset.overlap), '_window/', 
                            str(self.subject.dataset.wavelet_level), '_', self.subject.dataset.wavelet_type, '/correlation_matrix/'])
            # Example: Plots/200_100_window/2_db1/correlation_matrix/

            file_name = ''.join(['filtered_correlation_matrix_', self.subject.dataset.channels_names, '_', featureset_name, '_', self.pca, '_', self.scale, '.png'])
            # Example: importance_channels_featureset_no_pca_scale

            os.makedirs(os.path.dirname(full_path), exist_ok=True)

            plt.savefig(full_path + file_name, bbox_inches='tight', dpi=200)
            plt.clf() 
            
    def rf_feature_importance(self, feature_set):
        """
        Plots feature importance of RF model.

        The feature importance (variable importance) describes which features are 
        relevant. It can help with better understanding of the solved problem and 
        sometimes lead to model improvements by employing the feature selection.
        """
        
        skplt.estimators.plot_feature_importances(self.model, feature_names=self.subject.selected_dataframe.columns[0:len(self.subject.selected_dataframe.columns)-1],
                                         title="Random Forest Classification Feature Importance",
                                         x_tick_rotation=90, order="ascending")    # Generate plot for rf feature importance

        featureset_name = list(self.featuresets_dict.keys())[list(self.featuresets_dict.values()).index(feature_set)]
        
        full_path = ''.join(['Plots/', str(self.subject.dataset.window_size), '_', str(self.subject.dataset.overlap), '_window/', 
                            str(self.subject.dataset.wavelet_level), '_', self.subject.dataset.wavelet_type, '/rf_feature_importance/'])
        # Example: Plots/200_100_window/2_db1/rf_feature_importance/
    
        file_name = ''.join(['importance_', self.subject.dataset.channels_names, '_', featureset_name, '_', str(self.hyperparameters), '_', self.pca, '_', self.scale, '.png'])
        # Example: importance_channels_featureset_{n_estimators: 10}_no_pca_scale.png

        os.makedirs(os.path.dirname(full_path), exist_ok=True)    # Create directory if not there

        plt.savefig(full_path + file_name, bbox_inches='tight', dpi=200)    # Save figure
        plt.clf()
   

    # Feature selection 
    def feature_selection_plots(self):
        """
        Combines all the functions for plotting plots for feature selection.

        These plots can be slow and have too much information for a big number of features.
        """
        
        for feature_set in self.featuresets_dict.values():
            # Convert data to include onnly selected features
            self.subject.convert_data_format(feature_set)
            self.fit_model(feature_set=feature_set, cross_validation=False)

            # Feature selection plots
            self.information_gain_selection(feature_set=feature_set)
            self.filtered_information_gain_selection(feature_set=feature_set)
            self.fisher_score(feature_set=feature_set)
            self.filtered_fisher_score(feature_set=feature_set)
            self.correlation_matrix(feature_set=feature_set)
            self.filtered_correlation_matrix(feature_set=feature_set)

            if self.type == 'random_forest':
                self.rf_feature_importance(feature_set=feature_set)

    def forward_feature_selection(self, no_features: int, feature_set_name: str):
        """
        Performs forward feature selection.

        Forward selection is an iterative method in which we start with having no feature 
        in the model. In each iteration, we keep adding the feature which best improves our 
        model until an addition of a new variable does not improve the performance of the model.
        """
        
        self.subject.convert_data_format(self.featuresets_dict[feature_set_name])    # Get only those features that are going to be used (by their name), and convert data to right format

        X_train, X_test, y_train, y_test = train_test_split(
                self.subject.data, self.subject.target, shuffle=False, train_size=0.66666666)    # Split data into training and test 

        sfs = SFS(self.model,    # Make object for forward feature selection
                forward= True,
                k_features=no_features,
                scoring= 'accuracy',
                verbose=2,    # This prints the process, it is very useful to keep track of this since sometimes it is just too slow
                cv=3)

        sfs.fit(X_train, y_train, custom_feature_names= self.subject.features_names)    # Now, even if the training and testing data were divided by trial, SFS is going to 
                                                                                        # perform cross validation without this condition. This means that the scores used 
                                                                                        # for evaluating the features are not the real ones (they are actually smaller). A 
                                                                                        # possible solution would be to set cross validation to 0, or to do this process manually.
                                                                                        # Moreover, it also faces the same scaling problem that cross-validations.

        features_list = sfs.k_feature_names_   # Get best features that were chosen
        score = sfs.k_score_   # Get best score

        results_dict = sfs.subsets_   # Get dictionary with all chosen features and scores
        self.forward_selection = results_dict    # Save this dictionary as attribute

        cpu = [self.fit_model(list(results_dict[iteration]['feature_names']))[1] for iteration in results_dict.keys()]    # Get list of CPU times of fitting the model with these features

        # Generate plot
        fig1 = plot_sfs(sfs.get_metric_dict(), kind='std_dev', figsize=(20,15))
        plt.plot(results_dict.keys(),cpu, marker='x', color='darkgreen')
        plt.title('Evolution of Forward Selection for ' + str(self.subject.dataset.window_size) + '-' + str(self.subject.dataset.overlap) + ' Windowing')
        plt.legend(['0.95 Confidence Interval', 'Accuracy Score', 'CPU time fitting the model', 'Last score'])
        plt.axhline(y=sfs.get_metric_dict()[list(sfs.get_metric_dict())[-1]]['avg_score'], color='r', linestyle='dashed')
        plt.yticks([0.1 * i for i in range(1,11)])

        full_path = ''.join(['Plots/', str(self.subject.dataset.window_size), '_', str(self.subject.dataset.overlap), '_window/', 
                            str(self.subject.dataset.wavelet_level), '_', str(self.subject.dataset.wavelet_type) ,'/forward_selection/'])
        # Example: Plots/200_100_window/2_db1/forward_selection/

        if self.type == 'lda':
            file_name = ''.join([str(no_features), '_forward_selection_evolution_', self.type, '_', self.subject.dataset.channels_names, '_', feature_set_name, '.png'])
            # Example: 35_forward_selection_evolution_lda_all_channels_all_features.png
        else:    # All other models (which have hyperparameters)
            file_name = ''.join([str(no_features), '_forward_selection_evolution_', self.type, str(self.hyperparameters), '_', self.subject.dataset.channels_names, '_', feature_set_name, '.png'])
            # Example: 35_forward_selection_evolution_svc_{'kernel':...}_all_channels_all_features.png
        
        os.makedirs(os.path.dirname(full_path), exist_ok=True)    # Make directory if not already there

        plt.savefig(full_path + file_name, bbox_inches='tight', dpi=200)    # Save figure
        plt.clf()
        
        return features_list, score, results_dict    # Return best feature set and best score

    def backward_feature_selection(self, no_features: int, feature_set_name: str):
        """
        Performs backward elimination.

        In backward elimination, we start with all the features and remove 
        the least significant feature at each iteration which improves the 
        performance of the model.
        """
        
        self.subject.convert_data_format(self.featuresets_dict[feature_set_name])

        X_train, X_test, y_train, y_test = train_test_split(
                self.subject.data, self.subject.target, random_state=0)     

        sfs = SFS(self.model, 
                forward= False,
                k_features=no_features,
                scoring= 'accuracy',
                verbose=2, 
                cv=3)

        sfs.fit(X_train, y_train, custom_feature_names= self.subject.features_names)

        features_list = sfs.k_feature_names_
        score = sfs.k_score_

        results_dict = sfs.subsets_
        self.backward_elimination = results_dict    # Save this dictionary as attribute

        cpu = [self.fit_model(list(results_dict[iteration]['feature_names']))[1] for iteration in results_dict.keys()]

        # Generate plot
        fig1 = plot_sfs(sfs.get_metric_dict(), kind='std_dev', figsize=(20,15))
        plt.plot(results_dict.keys(),cpu, marker='x', color='darkgreen')
        plt.title('Evolution of Backward Elimination')
        plt.legend(['0.95 Confidence Interval', 'Accuracy Score', 'CPU time fitting the model'])

        full_path = ''.join(['Plots/', str(self.subject.dataset.window_size), '_', str(self.subject.dataset.overlap), '_window/', 
                            str(self.subject.dataset.wavelet_level), '_', str(self.subject.dataset.wavelet_type) ,'/backward_elimination/'])
        # Example: Plots/200_100_window/2_db1/backward_elimination/
        
        if self.type == 'lda':
            file_name = ''.join([str(no_features), '_backward_elimination_evolution_', self.subject.dataset.channels_names, '_', feature_set_name, '.png'])
            # Example: 35_backward_selection_evolution_lda_all_channels_all_features.png
        else:
            file_name = ''.join([str(no_features), '_backward_elimination_evolution_', self.subject.dataset.channels_names, '_', feature_set_name, '.png'])
        
        full_path = ''.join(['Plots/', str(self.subject.dataset.window_size), '_', str(self.subject.dataset.overlap), '_window/', 
                            str(self.subject.dataset.wavelet_level), '_', str(self.subject.dataset.wavelet_type) ,'/backward_elimination/'])
        # Example: 35_backward_selection_evolution_svc_{'kernel':...}_all_channels_all_features.png
        
        os.makedirs(os.path.dirname(full_path), exist_ok=True)    # Create directory if not already there

        plt.savefig(full_path + file_name, bbox_inches='tight', dpi=200)    # Save figure
        plt.clf()
        
        return features_list, score, results_dict    # Returns best feature set and best score

    def store_forward_selection_results(self):
        """
        TODO: fill
        """

        forward_dataframe = pd.DataFrame({'no_features': list(self.forward_selection),    # Make dictionary with all the results from feature selection plots
                                        'score': [self.forward_selection[i]['avg_score'] for i in list(self.forward_selection)],
                                        'features': [self.forward_selection[i]['feature_names'] for i in list(self.forward_selection)]
                                        })

        path_name = ''.join(['Code/Results/', str(self.subject.dataset.window_size), '_', str(self.subject.dataset.overlap), '_window/', 
                            str(self.subject.dataset.wavelet_level), '_', self.subject.dataset.wavelet_type, '/'])
        # Example: Code/Results/200_200_window/2_db1/
        
        file_name = ''.join([str(list(self.forward_selection)[-1]), '_forwardselection_', self.type, '_', self.subject.dataset.channels_names, '_', 
                str(self.hyperparameters), '_', self.pca, '_',  self.scale, '_results.xlsx'])    
        # Example: 10_forwardselection_10_svc_channels_[time,time_freq,all]_{C: 10, kernel: 'linear'}_no_pca_scale_results.xlsx

        os.makedirs(os.path.dirname(path_name + file_name), exist_ok=True)    # Create directory if not already there

        writer = pd.ExcelWriter(''.join([path_name, file_name]))    # Save file
        forward_dataframe.to_excel(writer)
        writer.save()

    def store_backward_elimination_results(self):
        """
        TODO: fill
        """

        backward_dataframe = pd.DataFrame({'no_features': list(self.backward_elimination),    # Make dictionary with all the results from feature selection plots
                                        'score': [self.backward_elimination [i]['avg_score'] for i in list(self.backward_elimination)],
                                        'features': [self.backward_elimination[i]['feature_names'] for i in list(self.backward_elimination)]
                                        })

        path_name = ''.join(['Code/Results/', str(self.subject.dataset.window_size), '_', str(self.subject.dataset.overlap), '_window/', 
                            str(self.subject.dataset.wavelet_level), '_', self.subject.dataset.wavelet_type, '/'])
        # Example: Code/Results/200_200_window/2_db1/
        
        file_name = ''.join([str(list(self.backward_elimination)[-1]), '_backwardelimination_', self.type, '_', self.subject.dataset.channels_names, '_', 
                str(self.hyperparameters), '_', self.pca, '_',  self.scale, '_results.xlsx'])    
        # Example: 10_backwardelimination_10_svc_channels_[time,time_freq,all]_{C: 10, kernel: 'linear'}_no_pca_scale_results.xlsx

        os.makedirs(os.path.dirname(path_name + file_name), exist_ok=True)    # Create directory if not already there

        writer = pd.ExcelWriter(''.join([path_name, file_name]))    # Save file
        backward_dataframe.to_excel(writer)
        writer.save()



def classification_scores(y_test,y_pred):
    """
    Some extra scores for classification models.
    """
    
    accuracy = metrics.accuracy_score(y_test,y_pred)
    balanced_accuracy = metrics.balanced_accuracy_score(y_test,y_pred)
    top_k_accuracy = metrics.top_k_accuracy_score(y_test,y_pred)
    average_precision = metrics.average_precision_score(y_test,y_pred)
    neg_brier_score = metrics.brier_score_loss(y_test,y_pred)

def manual_lda():
    """
    TODO: fill
    """
    print('hello')
"""
@author: Monse Guedes Ayala
@project: Bioliberty EMG Classification Algorithm

This code is the core of the preprocessing steps of the signals. It includes one class for data files, which allows us
to create and process a dataframe for each file. The folders and files need to be in the standard format (explained in
confluence), so that all the functions work.
"""

# Python libraries
from numpy.lib.stride_tricks import as_strided
import numpy as np
from scipy import signal
from itertools import product
from pandas import read_csv
from pandas import MultiIndex
from pandas import DataFrame
from pandas import concat
from pandas import DataFrame

# Self-created inports 
import feature_functions as ffun

def column_windowing(column, window_size: int, overlap: int):
    """Takes a column and returns its windows.

    Takes a column, window size, and overlap as input, and 
    returns an array with all the windows as inner arrays.
    """

    # Convert column to array
    column = np.asarray(column)
    window_step = window_size - overlap 

    # Define shape of returning array
    new_shape = column.shape[:-1] + ((column.shape[-1] - overlap) // window_step,
                                window_size)
    new_strides = (column.strides[:-1] + (window_step * column.strides[-1],) +
                column.strides[-1:])

    # Fill in the array with desires values
    windowed_column = as_strided(column, shape=new_shape, strides=new_strides)
    
    return windowed_column

class File_data:
    """
    Class representing a data file.
    
    Attributes
    ----------
    no_subject : int
        the number assiged to the subject for which this data was collected
    no_movement : int
        the number assiged to the movement that the subject performed when collecting the file data
    no_trial : int
        the number of the trial of the data present in the file. There are several trials for same movement-subject
    no_channels : int
        number of channels recorded in this data file (usually the same for all subjects and movements)
    no_windows: int 
        number of windows that are applied to the data (obtained after applying method for windowing)
    
    file_channels: list of strings
        name of each channel/column, in the form of 'channel_4'
    window_names: list of strings
        name of each window row, in the form of 'window_9'
    channels_used:
        list of only teh channels that are currently
    
    dataframe: dataframe
        original (processed) dataframe, with channels as columns, and original time periods as rows
    windowed: dataframe
        windowed dataframe, just as dataframe but with a multiindex representing different windows (this is obtained after 
        applying windowing method)
    feature_dataframe: dataframe
        dataframe where features are extracted fro each window, returning to a single index (window) (it is obtained after 
        feature extraction method, and depends on the list of features to be extracted)

    file_name: str
        name of the file used to create the class
    folder_name: str
        name of the folder where the file is at
    path: str
        full relative path of the studied file


    Methods
    -------
    create_dataframe
        created dataframe from file
    remove_mean
        removes mean from the signal
    rectify_signal
        rectifies signal (makes negative values positive)
    filter_signal(low_pass=10, sfreq=1000, high_band=20, low_band=450)
        filters data within band
    standard_moving_average(window_size)
        finds the standard moving average of the signal
    dataframe_windowing
        windows the dataframe
    mean_normalization(before=True)
        normalizes data by mean
    min_normalization(before=True)
        normalizes data by min
    feature_extraction(features: list, wavelet_level= 4, wavelet_type= 'db1')
        extracts desired features from windowed dataframe
    process_file(window_size: int, overlap: int, features: list, wavelet_level= 4, wavelet_type= 'db1')
        performs all the necessary methods to prepare file

    """    

    def __init__(self, dataset, no_subject: int, no_movement: int, no_trial: int, channels_used: list):
        # Numbers
        self.no_subject = no_subject 
        self.no_movement = no_movement
        self.no_trial = no_trial
        self.no_channels = None
        self.no_windows = None
        
        # Names
        self.file_channels = None
        self.window_names = None
        self.channels_used = [''.join(['channel_', str(channel)]) for channel in channels_used]

        # Dataframes
        self.dataframe = None
        self.windowed = None
        self.feature_dataframe = None

        # Path-related attributes
        self.file_name = ''.join(['mov', str(no_movement), '_', str(no_trial), '.csv'])
        self.folder_name = ''.join(['S', str(no_subject)])
        self.path = ''.join(['Datasets/standard_datasets/', dataset.folder_name, '/', self.folder_name, '/'])

    def create_dataframe(self):
        """ Converts file to dataframe.
        
        Creates dataframe, sets index to be 'time', and stores 
        the dataframe, no.channels, and channels'names as class 
        attributes.
        """

        # Read data file and create dataframe
        dataframe = read_csv(''.join([self.path, self.file_name]), index_col=0)

        self.file_channels = list(dataframe.columns)    # Create attribute with number of channels in file

        # Get only used channels
        dataframe = dataframe[self.channels_used]
        
        # Set index as time
        dataframe.reset_index()
        dataframe.index.rename('time')

        # Define attributes
        self.dataframe = dataframe
        self.no_channels = len(dataframe.columns)
        
        return dataframe

    def remove_mean(self):
        """Removes the mean value from the signal.

        Subtracts the mean of the signal from all the individual 
        signal values, in order to center it around 0.
        """

        # Loop over all channels to subtract the mean
        for channel in self.channels_used:
            self.dataframe[channel] = self.dataframe[channel] - self.dataframe[channel].mean()
        
        return self.dataframe

    def rectify_signal(self):
        """Rectifies the signal.

        Converts all negative values to positive values.
        It substitutes previous dataframe attribute, so 
        once this function is applied the data is permanently
        modified. 
        """

        self.dataframe = self.dataframe.abs()

        return self.dataframe 

    def filter_signal(self, low_pass=10, sfreq=1000, high_band=20, low_band=450):
        """Filters the data within a band.

        Considers only values that are within a desired band. 
        It must be applied after removing mean.
        """

        # Normalise cut-off frequencies to sampling frequency
        high_band = high_band/(sfreq/2)
        low_band = low_band/(sfreq/2)
        
        # Create bandpass filter for EMG
        b1, a1 = signal.butter(4, [high_band,low_band], btype='bandpass')

        for channel in self.channels_used:

            # Process EMG signal: filter EMG
            self.dataframe[channel] = signal.filtfilt(b1, a1,self.dataframe[channel])    
            
            # Create lowpass filter and apply to rectified signal to get EMG envelope
            #low_pass = low_pass/(sfreq/2)
            #b2, a2 = sp.signal.butter(4, low_pass, btype='lowpass')
            #emg_envelope = sp.signal.filtfilt(b2, a2, emg)

        return self.dataframe
    
    def standard_moving_average(self, window_size: int):
        """Finds the standard moving average of the signal.

        window_size: number of time period (window) for which the moving average 
        is calculated.
        """

        # Loop over all channels, and find their moving average
        for channel in self.channels_used:
            self.dataframe['moving_average'] = self.dataframe[channel].rolling(window_size, min_periods=1).mean()
        
        return self.dataframe
    
    def dataframe_windowing(self, window_size: int, overlap: int):
        """Windows the dataframe.

        Creates a multiindex for windowing (depending on input parameters),
        windows each channel individually, and then joins all channels to 
        create main windowed dataframe. The goal of this function is to go 
        from a SxC matrix (dataframe), where S is the total number of samples, 
        and C is the total number of channels, to a Wx(MxC) matrix, where W is 
        the number of windows, and M the number of periods per window. 
        
        """
        # Create multiindex for windowing
        no_windows = len(column_windowing(self.dataframe[self.channels_used[0]], window_size, overlap))    # Find no. of windows
        window_names = [ ''.join(['window_', str(i)]) for i in range(1, no_windows + 1)]    # Create list of window names, e.g. 'window_1'
        times_per_window = list(range(window_size))    # Get no. periods per window
        multi_index = MultiIndex.from_product([window_names, times_per_window], names=["window", "time"])    #Create multiiindex
        
        # Assign values to attributes
        self.no_windows = no_windows
        self.window_names = window_names

        dataframes_list = [] # Empty list to later add the windowed dataframe of each channel

        # Loop over all channels and window them
        for channel in self.channels_used: 
            window_channel = column_windowing(column=self.dataframe[channel], window_size=window_size, overlap=overlap).flatten()    # Apply windowing function to channel column
            window_dataframe = DataFrame(window_channel, index= multi_index, columns=[channel])    # Create dataframe for each channel where column-channel, rows-multiindex(window/time)
            dataframes_list.append(window_dataframe)    # Add channel windowed dataframe to empty list
        
        # Store the windowed dataframes for all channels
        self.windowed = concat(dataframes_list, axis=1)
        
        return self.windowed

    def mean_normalization(self, before= True):
        """Normalizes data by mean.

        It subtracts the mean and divides by standard deviation. 
        Can be applied either before of after windowing. This 
        function replaces previous values of the dataframe and
        windowed dataframe attributes.
        """

        if before == True:    # Applied before windowing
            for channel in self.channels_used:    # Loop over all channels and subtract mean and dividing by standard deviation
                self.dataframe[channel] = (self.dataframe[channel] - self.dataframe[channel].mean()) / np.std(self.dataframe[channel])
        
            return self.dataframe
        
        else:    # Applied after windowing (normalizes each window)
            for channel, window in product(self.channels_used, self.window_names):    # Loop over all channel-window combinations, and subtract mean and divide by standard deviation of window
                column_to_normalize = self.windowed.iloc[self.windowed.index.get_level_values('window')== window, self.windowed.columns.get_loc(channel)] # Get the column to normalize
                column_to_normalize = (column_to_normalize - np.mean(column_to_normalize)) / np.std(column_to_normalize) # Normalize the column
                self.windowed.iloc[self.windowed.index.get_level_values('window')== window, self.windowed.columns.get_loc(channel)] = column_to_normalize # Assign new values to window

            return self.windowed
               
    def min_normalization(self, before= True):
        """Normalizes data by min.

        It subtracts the min value and divides by range (max-min). 
        Can be applied either before of after windowing. This 
        function replaces previous values of the dataframe and
        windowed dataframe attributes.
        """

        if before == True:    # Applied before windowing
            for channel in self.channels_used:    # Loop over all channels and subtract min and dividing by range
                self.dataframe[channel] = (self.dataframe[channel] - np.min(self.dataframe[channel])
                ) / (np.max(self.dataframe[channel]) - np.min(self.dataframe[channel]))

            return self.dataframe        

        else:    # Applied after windowing (normalize each window)
            for channel, window in product(self.channels_used, self.window_names): # Loop over all channel-window combinations, and subtract mean and divide by standard deviation of window
                column_to_normalize = self.windowed.iloc[self.windowed.index.get_level_values('window')== window, self.windowed.columns.get_loc(channel)]
                column_to_normalize = (column_to_normalize - np.min(column_to_normalize)) / (np.max(column_to_normalize) - np.min(column_to_normalize))
                self.windowed.iloc[self.windowed.index.get_level_values('window')== window, self.windowed.columns.get_loc(channel)] = column_to_normalize

            return self.windowed

    def feature_extraction(self, features: list, wavelet_level= 4, wavelet_type= 'db1'):
        """Extracts desired features from windowed dataframe.
        
        It takes as input a list with the set of features to extract, the wavelet level
        and wavelet type for time-frequency domain features. It produces a new dataframe
        with features as columns and windows as rows.
        """

        def feature_vector(channel: str, window: str):
            """Creates a feature vector for a specific channel and window.

            It includes all the desired features ordered as they are ordered
            in the input list of the main function.
            """

            # Create array object for specific window and channel
            array = ffun.Dataframe_column(self.windowed, self.window_names.index(window), self.file_channels.index(channel))

            # If time-frequency domain features are involved, then decompose the array and store decomposition
            if any('wavelet' in feature for feature in features):
                array.wavelet_decomposition(wavelet_level= wavelet_level, wavelet_type= wavelet_type)

            # Create the feature vector by extracting each feature from array
            feature_vector = [array.extract_feature(feature) for feature in features]

            return feature_vector    # Example: for window 1 and channel 1, this is [feature 1, feature 2, ...]

        def column_name(channel: str, feature: str):
            """Gets the names of all columns for given channel.

            It names the columns of the feature dataframe of each channel.
             To be used later in the function 'channel_dataframe'.
            """

            column_name = ''.join([channel, '_', feature]) # Make column name as 'channel_1_feature_function_name' 

            return column_name    # Example: channel_1_zero_crossing

        def channel_dataframe(channel: str):
            """Creates a feature dataframe for a given channel. 

            Takes as input the desires channel name and produces its
            feature dataframe.
            """

            column_names = [column_name(channel, feature) for feature in features]   # Define the name of the columns ex: [channel_1_feature_1, channel_1_feature_2, channel_1_feature_3, ...]
            feature_matrix = [feature_vector(channel, window) for window in self.window_names]    # List of vectors, so a matrix with the following form: [[chanel_1_feature_1, channel_1_feature_2,...](window 1), [chanel_1_feature_1, channel_1_feature_2,...](window 2), ...]
            channel_dataframe = DataFrame(np.array(feature_matrix), columns=column_names)    # Convert this matric as a dataframe with column names

            return channel_dataframe    # Example: for channel 1, a dataframe with columns [channel_1_feature_1, channel_1_feature_2, channel_1_feature_3, ...], and then one row per window with extracted features
        
        # List of dataframes for all channels
        list_channels_dataframes = [channel_dataframe(channel) for channel in self.channels_used]  

        # Store dataframe with features
        self.feature_dataframe = concat(list_channels_dataframes, axis=1)    # Horizontally join all dataframes

        return self.feature_dataframe
                
    def process_file(self, window_size: int, overlap: int, features: list, wavelet_level= 4, wavelet_type= 'db1'):
        """Performs all the necessary methods to prepare file.

        It creates the dataframe, normalizes it, windows it, and
        extracts desired features.
        
        """

        # Apply methods
        self.create_dataframe()    # Add dataframe to class
        self.mean_normalization()    # Normalize dataframe
        self.dataframe_windowing(window_size= window_size, overlap= overlap)    # Windowing the dataframe
        self.feature_extraction(features, wavelet_level= wavelet_level , wavelet_type= wavelet_type)    # Extract features
        
        #Add target label to feature dataframe
        self.feature_dataframe['target'] = self.no_movement
        
        return self.feature_dataframe
        





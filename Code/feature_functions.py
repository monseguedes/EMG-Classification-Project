"""
@author: Monse Guedes Ayala
@project: Bioliberty EMG Classification Algorithm

Functions for feature extraction. The input is an array.
"""

# Python libraries
import numpy as np
from pandas import DataFrame
from pywt import wavedec
from pywt import WaveletPacket

class Dataframe_column:
    """
    Class representing a column of a given dataframe.
    
    Attributes
    ----------
    channel:
        no. of of channel for which the array is extracted
    channel_name:
        name of channel for which the array is extracted
    window_name:
        name of window for which the array is extracted, only exists if the dataframe is windowed
    column_array:
        array associeted with the desired column for a channel and window
    wavelet_array:
        wavelet decomposition of array
    wavelet_matrix:
        numpy matrix with all the wavelet decomposition coefficients by level
    wavelet_packet:
        packet decomposition of array
    packet_maxlevel:
        maximum level for the wavelet packet decomposition

    Methods
    -------
    extract_feature(feature)
        extracts a chosen feature from an array, returning a single value

    - Time domain functions
    - Time-frequency domain functions

    """    
    def __init__(self, dataframe: DataFrame, window: int, channel: int): 
        self.channel = channel
        self.channel_name = ''.join(['channel_', str(channel + 1)]) 

        if not window == None: #There is a window
            self.window_name = dataframe.index.unique(level= 'window')[window]
            self.column_array = dataframe.iloc[dataframe.index.get_level_values('window')== self.window_name, dataframe.columns.get_loc(self.channel_name)]
        else: #There is no window
            self.column_array = dataframe[self.channel_name]
        
        # Wavelet attributes
        self.wavelet_array = None
        self.wavelet_matrix = None
        self.wavelet_packet =  None
        self.packet_maxlevel = None

    # Feature extraction
    def extract_feature(self, feature: str):
        """Extracts a feature from an array
        """
        
        function = getattr(self, feature) # Get method for specific function
        feature_value = function() 

        return feature_value

    # Time domain features
    def mean_absolute_value(self): 
        """
        MAV is obtained by averaging the absolute value of the EMG signal in a window. 
        A large increase occurs in the value of this feature at onset and remains high 
        during the contraction.
        """

        MAV = np.mean(abs(self.column_array))

        return MAV

    def standard_deviation(self):
        """
        STD represents the difference between each sample of EMG and its mean value
        """

        STD = np.std(self.column_array)

        return STD

    def variance (self):
        """
        Variance represents the power of the EMG signal and helps to determine onset 
        and contraction.
        """

        Var = np.var(self.column_array)

        return Var

    def waveform_length(self):
        """
        Waveform length of the signal gives information about complexity of the signal 
        in a window by summing the numerical derivative of the sample window.
        """

        WL = sum(abs(np.diff(self.column_array)))

        return WL

    def zero_crossing(self): 
        """
        ZC counts the number of times that the sign of the amplitude of the signal changes. 
        """

        ZC = sum(np.diff(np.sign(self.column_array)) != 0)

        return ZC

    def root_mean_square(self):
        """
        The square root of the mean square.
        """

        RMS = np.sqrt(np.mean(abs(self.column_array) ** 2))

        return RMS

    def number_of_peaks(self):
        """
        Number of peaks is the number of values that are higher than their RMS value. 
        """

        RMS = self.root_mean_square()
        NP = (abs(self.column_array) > RMS).sum()

        return NP

    def mean_of_peak_values(self):
        """
        Mean of peak values is the average of the peak values.
        """

        RMS = self.root_mean_square()
        peaks = np.array([peak for peak in self.column_array if abs(peak) > RMS])
        MPV = np.mean(abs(peaks))

        return MPV

    def mean_firing_velocity(self): 
        """
        Mean firing velocity is the difference or velocity of the peak values.
        """

        RMS = self.root_mean_square()
        peaks = np.array([peak for peak in self.column_array if abs(peak) > RMS])
        MFV = sum(np.diff(peaks))

        return MFV

    def slope_sign_changes(self):
        """
        SSC represents the frequency properties of the EMG signal
        and it counts the number of times the slope of the EMG signal
        in a time window changes sign. 
        """

        SSC = sum(np.diff(np.sign(np.diff(self.column_array))) != 0)
        
        return SSC

    def difference_absolute_mean_value(self):
        """
        DAMV calculates the absolute value of the difference between signals for
        consecutive time periods, and takes the average. 
        """
        
        DAMV  = np.mean(abs(np.diff(self.column_array)))
        
        return DAMV

    def M_i(self, i= 2):
        """(Second) order moment
        """
        M_i = np.mean((self.column_array - np.mean(self.column_array)) ** i )
        
        return M_i

    def skewness(self):
        """
        The Skewness describes asymmetry in a statistical distribution 
        around the mean value.
        """

        M_2 = self.M_i(2)
        M_3 = self.M_i(3)
        Skew = (M_3 / M_2) * np.sqrt(M_2)

        return Skew

    def integrated_absolute_value(self):
        """
        The integral of absolute value is a summation of absolute values 
        of the EMG signal in a time window of N samples.
        """

        IAV = sum(abs(self.column_array))

        return IAV

    def hjorth_mobility_parameter(self):
        """
        Three parameters were introduced by Hjorth: activity, mobility, 
        and complexity (HCom). The mobility parameter is proportional 
        to the standard deviation of the power spectrum.
        """

        HMob = np.sqrt(np.var(np.diff(self.column_array) / np.var(self.column_array)))
        
        return HMob

    def hjorth_complexity_parameter(self):
        """
        Complexity is the third feature of the Hjorth parameters which 
        compares the similarity of the shape of a signal with a pure 
        sine wave.
        """

        derivative = np.diff(self.column_array)
        numerator = np.sqrt(np.var(np.diff(derivative) / np.var(derivative)))
        denominator = self.hjorth_mobility_parameter()
        HCob = numerator / denominator
        
        return HCob

    def difference_absolute_standard_deviation_value(self):
        """
        DASDV is a standard deviation value of the difference between the 
        adjacent samples.
        """
        
        DASDV = np.sqrt(np.mean(np.diff(self.column_array) ** 2))
        
        return DASDV

    def willison_amplitude(self):
        """
        WAM in a time window counts the number of times the absolute value 
        of the difference between two adjacent samples exceeds a predefined 
        threshold.
        """

        threshold = 0
        WAM = (abs(np.diff(self.column_array)) > threshold).sum()
        
        return WAM

    def kurtosis(self):
        """
        Kurtosis describes the shape of a statistical distribution compared 
        with the normal distribution.
        """

        M_2 = self.M_i(2)
        M_4 = self.M_i(4)
        Kurt = M_4 / (M_2 * M_2)

        return Kurt

    def simple_square_integrated(self):
        """
        Takes the square of each element, and sums them all.
        """
        
        SSI = (self.column_array ** 2).sum()
       
        return SSI

    def myopulse_percentage_rate(self):
        """
        The myopulse percentage rate (MYOP) is an average value of myopulse 
        output. It is defined as one absolute value of the EMG signal exceed 
        a pre-defined thershold value.
        """

        threshold = 0
        MYOP = np.mean(self.column_array > threshold)
        
        return MYOP

    def difference_variance_version(self):
        """
        Takes the difference between consecutive signal values,
        squares it, and divides it by the number of differences.
        """
        
        DVARV = np.mean(np.diff(self.column_array) ** 2)
        
        return DVARV


    #Time-frequency domain features
    def wavelet_decomposition(self, wavelet_level: int, wavelet_type: str):
        """Applies wavelet decomposition to signal.
        
        Takes as input the desires wavelet level and the type of wavelet
        used for the decomposition, and returns an array with the desired
        coefficients.

        wavedec 
        returns: [cA_n, cD_n, cD_n-1, …, cD2, cD1] : list
        ordered list of coefficients arrays where n denotes the level of 
        decomposition. The first element (cA_n) of the result is approximation 
        coefficients array and the following elements (cD_n - cD_1) are details 
        coefficients arrays.
        """
        
        coefficients = wavedec(self.column_array, wavelet=wavelet_type, level=wavelet_level)   # Approximation coefficients and detail coefficients
        flat_list = [item for array in coefficients for item in array] # Flatten all coefficients
        self.wavelet_matrix = np.array(coefficients, dtype=object) # Store all coefficients as wavelet_matrix
        self.wavelet_array = np.array(flat_list)
        
        return self.wavelet_array

    def obtain_wavelet_packet(self, max_level= 4, wavelet_type= 'db1'):
        """Applies wavelet packet decomposition to signal.
        
        Takes as input the desires wavelet level and the type of wavelet
        used for the decomposition, and returns an array with the desired
        coefficients.
        """
        
        self.packet_maxlevel = max_level
        packet = WaveletPacket(self.column_array, wavelet= wavelet_type, maxlevel= max_level) #Creates packet object
        paths = [node.path for node in packet.get_level(max_level, 'natural')]
        data = [node.data for node in packet.get_level(max_level, 'natural')]
        self.wavelet_packet = data

    def standard_deviation_wavelet_coefficients(self):
        """
        After applying DWT to each segment of the EMG signal, standard deviation 
        of the wavelet coefficients in the last level is calculated.
        """
        
        #STD_array = np.apply_along_axis(np.std, 1, self.wavelet_matrix)
        STD_wavelet = np.std(self.wavelet_array)
        
        return STD_wavelet

    def variance_wavelet_coefficients(self):
        """
        First, each segment of the EMG signal was decomposed using DWT; then, 
        variance of the wavelet coefficients in the last level was calculated.
        """
        
        #Var_array = np.apply_along_axis(np.var, 1, self.wavelet_matrix)
        var_wavelet = np.var(self.wavelet_array)
        
        return var_wavelet

    def waveform_length_wavelet_coefficients(self):
        """
        After applying wavelet transform to each window of the EMG signal, waveform 
        length of the wavelet coefficients in the last level was calculated.
        """
        
        #WL_array = np.apply_along_axis(lambda x: sum(abs(np.diff(x))), 1, self.wavelet_matrix)
        WL_wavelet = sum(abs(np.diff(self.wavelet_array)))
        
        return WL_wavelet

    def energy_wavelet_coefficients(self):
        """
        The EMG signal is decomposed by wavelet transform into levels; then, 
        the energy of the wavelet coefficients is determined in the last level 
        as components of the feature vector. 
        """

        E_wavelet = sum((abs(self.wavelet_array) ** 2))
        
        return E_wavelet

    def maximum_absolute_value_wavelet_coefficients(self):
        """
        The maximum absolute value (MaxAV) of the wavelet coefficients in the 
        last level was calculated as the feature vector of EMG signals.
        """
        
        MAV_wavelet = np.max(abs(self.wavelet_array))
        
        return MAV_wavelet

    def zero_crossing_wavelet_coefficients(self):
        """
        After decomposing the EMG signal using DWT, the number of ZC of the wavelet 
        coefficients in the last level is evaluated.
        """
        
        ZC_wavelet = sum(np.diff(np.sign(self.wavelet_array)) != 0)
        
        return ZC_wavelet

    def mean_wavelet_coefficients(self):
        """
        The EMG signal was decomposed by DWT into four levels; then, the mean of 
        the wavelet coefficients in the last level was calculated.
        """
        
        mean_wavelet = np.mean(self.wavelet_array)
        
        return mean_wavelet

    def mean_absolute_value_wavelet_coefficients(self):
        """
        After decomposing the EMG signal using DWT into four levels, the mean absolute 
        value of the wavelet coefficients in the last level was calculated.
        """
        
        MAV_wavelet = np.mean(abs(self.wavelet_array))
        
        return MAV_wavelet

    def logarithmic_RMS_wavelet_packet_coefficients(self, wavelet= 'db1', maxlevel= 4):
        """
        After applying the wavelet packet transform (WPT) and decomposing the EMG into 
        levels, the logarithmic RMS (LogRMS) of the coefficient in the last subspace was 
        calculated.
        """
        
        if self.wavelet_packet == None:
            self.obtain_wavelet_packet(max_level= maxlevel, wavelet_type= wavelet)
        last_subspace = self.wavelet_packet[-1]
        LRMS = np.sqrt(np.mean(np.log(abs(last_subspace + 0.00000001)) ** 2)) #Does log go here or applied ot everything?
        
        return LRMS

    def relative_energy_wavelet_packet_coefficients(self, wavelet= 'db1', maxlevel= 4, subspace=0):
        """
        After the EMG had been decomposed by WPT, the relative energy (RE) of the 
        coefficients in every subspace was employed as the signal feature set.
        """
        
        if self.wavelet_packet == None:
            self.obtain_wavelet_packet(max_level= maxlevel, wavelet_type= wavelet)
        subspace_energy = np.array([sum(abs(subspace) ** 2) for subspace in self.wavelet_packet])
        total_energy = sum(subspace_energy)
        relative_energy = subspace_energy / total_energy
        
        return relative_energy[subspace] #This output is an array

    def normalized_logarithmic_energy_wavelet_packet_coefficients(self, wavelet= 'db1', maxlevel= 4):
        """
        WPT was applied to the EMG signals to generate wavelet coefficients up to 
        a level of decomposition. The logarithmic operator was then applied to the
        accumulation of the squares of the coefficients divided by the number of 
        coefficients in the subspace.
        """
        
        if self.wavelet_packet == None:
            self.obtain_wavelet_packet(max_level= maxlevel, wavelet_type= wavelet)
        last_subspace = self.wavelet_packet[-1]
        NLE = np.log(sum(last_subspace ** 2) / (len(last_subspace) / 2 ** self.packet_maxlevel))
        
        return NLE

    def integrated_absolute_second_derivative_wavelet(self):
        """
        This feature captures the relative changes of the second derivative 
        of a signal which behaves like a filter to reduce the noise.

        IASD = ∑|x'[n+1]−x'[n]| where x'[n] = x[n+1] − x[n]
        """

        IASD = sum(abs(np.diff(self.wavelet_array, n=2)))
        
        return IASD
    
    def integrated_absolute_third_derivative_wavelet(self):
        """
        This feature captures the relative changes of the third derivative of a 
        signal. Similar to IASD, the third derivative also filters out the noise.

        IATD = ∑|x"[n + 1] − x"[n]| where  x"[n] = x'[n + 1] − x'[n]
        """

        IATD = sum(abs(np.diff(self.wavelet_array, n=3)))
        
        return IATD

    def integrated_exponential_absolute_values_wavelet(self):
        """
        This function amplifies the samples that are large and suppresses the 
        samples that are small for all positive and negative samples.
        """

        IEAV = sum(np.exp(abs(self.wavelet_array)))
        
        return IEAV

    def integrated_absolute_log_values_wavelet(self, T= 1): #T is a threshold that needs to be tuned
        """
        This function suppresses the samples that are large and amplifies 
        the samples that are small.
        """

        IALV = sum(abs(np.log(abs(self.wavelet_array + T))))
        
        return IALV

    def integrated_exponential_wavelet(self):
        """
        It is similar to IEAV while distinguishes between positive and negative 
        samples, i.e., generally amplifies positive samples and suppresses 
        negative ones.
        """

        IE = sum(np.exp(self.wavelet_array))
        
        return IE

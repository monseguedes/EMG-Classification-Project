"""
@author: Monse Guedes Ayala
@project: Bioliberty EMG Classification Algorithm

This file has several lists of possible feature sets that are used 
in the main file for the analysis. 
"""

# Lists of features names
# ----------------------------------------------
time_features = ['mean_absolute_value', 'standard_deviation', 'variance', 'waveform_length', 'zero_crossing', 'root_mean_square', 
'number_of_peaks', 'mean_of_peak_values', 'mean_firing_velocity', 'slope_sign_changes', 'difference_absolute_mean_value', 'M_i', 'skewness',
'integrated_absolute_value', 'hjorth_mobility_parameter', 'hjorth_complexity_parameter', 'difference_absolute_standard_deviation_value', 
'willison_amplitude', 'kurtosis', 'simple_square_integrated', 'myopulse_percentage_rate', 'difference_variance_version']

time_freq_features = ['standard_deviation_wavelet_coefficients', 'variance_wavelet_coefficients', 'waveform_length_wavelet_coefficients', 'energy_wavelet_coefficients',
'maximum_absolute_value_wavelet_coefficients', 'zero_crossing_wavelet_coefficients', 'mean_wavelet_coefficients', 'mean_absolute_value_wavelet_coefficients',
'logarithmic_RMS_wavelet_packet_coefficients', 'relative_energy_wavelet_packet_coefficients', 'normalized_logarithmic_energy_wavelet_packet_coefficients',
'integrated_absolute_second_derivative_wavelet','integrated_absolute_third_derivative_wavelet','integrated_exponential_absolute_values_wavelet',
'integrated_absolute_log_values_wavelet', 'integrated_exponential_wavelet']

# Lists of features after forward selection for different windows
# ----------------------------------------------
lda_forward_selection_25_features_200_100_4db1 = ['channel_1_slope_sign_changes', 'channel_1_kurtosis', 'channel_1_integrated_absolute_log_values_wavelet', 'channel_2_waveform_length', 
                                            'channel_2_root_mean_square', 'channel_2_slope_sign_changes', 'channel_2_difference_absolute_standard_deviation_value', 'channel_3_variance', 
                                            'channel_3_waveform_length', 'channel_3_slope_sign_changes', 'channel_3_skewness', 'channel_3_difference_absolute_standard_deviation_value', 
                                            'channel_4_waveform_length', 'channel_4_slope_sign_changes', 'channel_4_hjorth_mobility_parameter', 'channel_4_hjorth_complexity_parameter', 
                                            'channel_4_difference_variance_version', 'channel_4_normalized_logarithmic_energy_wavelet_packet_coefficients', 'channel_5_waveform_length', 
                                            'channel_5_hjorth_mobility_parameter', 'channel_5_hjorth_complexity_parameter', 'channel_5_difference_absolute_standard_deviation_value', 
                                            'channel_6_standard_deviation', 'channel_6_waveform_length', 'channel_6_slope_sign_changes']

lda_forward_selection_27_features_300_150_4db1 = ['channel_1_number_of_peaks', 'channel_1_mean_of_peak_values', 'channel_1_slope_sign_changes', 'channel_1_integrated_absolute_log_values_wavelet', 
                                                'channel_2_standard_deviation', 'channel_2_waveform_length', 'channel_2_slope_sign_changes', 'channel_2_hjorth_complexity_parameter', 
                                                'channel_3_waveform_length', 'channel_3_slope_sign_changes', 'channel_3_hjorth_mobility_parameter', 'channel_3_difference_variance_version',
                                                'channel_4_waveform_length', 'channel_4_slope_sign_changes', 'channel_4_skewness', 'channel_4_difference_variance_version', 
                                                'channel_4_relative_energy_wavelet_packet_coefficients', 'channel_5_waveform_length', 'channel_5_hjorth_mobility_parameter', 
                                                'channel_5_hjorth_complexity_parameter', 'channel_5_kurtosis', 'channel_6_waveform_length', 'channel_6_slope_sign_changes', 
                                                'channel_6_hjorth_mobility_parameter', 'channel_6_myopulse_percentage_rate', 'channel_7_slope_sign_changes', 'channel_7_integrated_absolute_third_derivative_wavelet']

lda_forward_selection_25_features_365_10_4db1 = ['channel_1_slope_sign_changes', 'channel_1_difference_variance_version', 'channel_1_integrated_absolute_log_values_wavelet', 
                                                'channel_2_waveform_length', 'channel_2_mean_firing_velocity', 'channel_2_slope_sign_changes', 'channel_2_hjorth_complexity_parameter', 
                                                'channel_2_kurtosis', 'channel_2_mean_absolute_value_wavelet_coefficients', 'channel_3_waveform_length', 'channel_3_mean_of_peak_values', 
                                                'channel_3_slope_sign_changes', 'channel_3_myopulse_percentage_rate', 'channel_4_slope_sign_changes', 'channel_4_hjorth_mobility_parameter', 
                                                'channel_4_kurtosis', 'channel_4_normalized_logarithmic_energy_wavelet_packet_coefficients', 'channel_5_mean_absolute_value', 
                                                'channel_5_waveform_length', 'channel_5_hjorth_complexity_parameter', 'channel_6_waveform_length', 'channel_6_slope_sign_changes', 
                                                'channel_6_hjorth_mobility_parameter', 'channel_7_slope_sign_changes', 'channel_7_skewness']

lda_forward_selection_25_features_100_50_4db1 = ['channel_1_waveform_length', 'channel_1_slope_sign_changes', 'channel_1_hjorth_complexity_parameter', 'channel_1_kurtosis',
                                                'channel_1_simple_square_integrated', 'channel_1_difference_variance_version', 'channel_1_integrated_absolute_log_values_wavelet', 
                                                'channel_2_waveform_length', 'channel_2_mean_of_peak_values', 'channel_2_slope_sign_changes', 'channel_2_hjorth_complexity_parameter', 
                                                'channel_2_difference_absolute_standard_deviation_value', 'channel_2_myopulse_percentage_rate', 'channel_2_integrated_absolute_log_values_wavelet', 
                                                'channel_2_integrated_exponential_wavelet', 'channel_3_variance', 'channel_3_waveform_length', 'channel_3_slope_sign_changes', 'channel_3_skewness', 
                                                'channel_3_hjorth_mobility_parameter', 'channel_3_difference_absolute_standard_deviation_value', 'channel_3_waveform_length_wavelet_coefficients', 
                                                'channel_4_waveform_length', 'channel_4_mean_of_peak_values', 'channel_4_slope_sign_changes', 'channel_4_hjorth_mobility_parameter', 
                                                'channel_4_difference_absolute_standard_deviation_value', 'channel_4_variance_wavelet_coefficients', 'channel_4_integrated_absolute_log_values_wavelet', 
                                                'channel_5_mean_absolute_value', 'channel_5_waveform_length', 'channel_5_root_mean_square', 'channel_5_slope_sign_changes', 
                                                'channel_5_hjorth_mobility_parameter', 'channel_5_hjorth_complexity_parameter', 'channel_6_waveform_length', 'channel_6_root_mean_square', 
                                                'channel_6_slope_sign_changes', 'channel_6_skewness', 'channel_6_hjorth_complexity_parameter', 'channel_6_difference_absolute_standard_deviation_value', 
                                                'channel_7_waveform_length', 'channel_7_slope_sign_changes', 'channel_7_difference_variance_version', 'channel_7_normalized_logarithmic_energy_wavelet_packet_coefficients']

# Lists of features after forward selection for different wavelets
# ----------------------------------------------
lda_forward_selection_21_features_300_150_4db4 = ['channel_1_number_of_peaks', 'channel_1_slope_sign_changes', 'channel_2_waveform_length', 'channel_2_slope_sign_changes', 'channel_2_skewness', 
                                                'channel_2_hjorth_complexity_parameter', 'channel_3_mean_absolute_value', 'channel_3_waveform_length', 'channel_3_slope_sign_changes', 
                                                'channel_3_skewness', 'channel_3_difference_absolute_standard_deviation_value', 'channel_4_slope_sign_changes', 'channel_4_difference_variance_version', 
                                                'channel_4_normalized_logarithmic_energy_wavelet_packet_coefficients', 'channel_4_integrated_absolute_log_values_wavelet', 'channel_5_waveform_length', 
                                                'channel_5_hjorth_mobility_parameter', 'channel_5_hjorth_complexity_parameter', 'channel_6_mean_absolute_value', 'channel_6_waveform_length', 
                                                'channel_6_slope_sign_changes']

lda_forward_selection_27_features_300_150_4db2 =  ['channel_1_number_of_peaks', 'channel_1_slope_sign_changes', 'channel_2_mean_absolute_value', 'channel_2_waveform_length', 'channel_2_number_of_peaks', 
                                                    'channel_2_slope_sign_changes', 'channel_2_skewness', 'channel_2_hjorth_complexity_parameter', 'channel_2_difference_absolute_standard_deviation_value', 
                                                    'channel_3_waveform_length', 'channel_3_number_of_peaks', 'channel_3_slope_sign_changes', 'channel_3_logarithmic_RMS_wavelet_packet_coefficients', 
                                                    'channel_4_waveform_length', 'channel_4_slope_sign_changes', 'channel_4_skewness', 'channel_4_hjorth_complexity_parameter', 'channel_4_zero_crossing_wavelet_coefficients', 
                                                    'channel_4_normalized_logarithmic_energy_wavelet_packet_coefficients', 'channel_5_hjorth_complexity_parameter', 'channel_5_difference_variance_version', 
                                                    'channel_6_waveform_length', 'channel_6_slope_sign_changes', 'channel_6_hjorth_mobility_parameter', 'channel_7_slope_sign_changes']

# lda_forward_selection_27_features_300_150_4db1 is in previous list

lda_forward_selection_27_features_300_150_2db4 =  ['channel_1_mean_absolute_value', 'channel_1_difference_absolute_standard_deviation_value', 'channel_1_integrated_absolute_third_derivative_wavelet', 
                                                    'channel_2_waveform_length', 'channel_2_slope_sign_changes', 'channel_3_number_of_peaks', 'channel_3_difference_absolute_standard_deviation_value', 
                                                    'channel_3_logarithmic_RMS_wavelet_packet_coefficients', 'channel_3_integrated_absolute_third_derivative_wavelet', 'channel_4_mean_of_peak_values', 
                                                    'channel_4_skewness', 'channel_4_waveform_length_wavelet_coefficients', 'channel_4_integrated_absolute_third_derivative_wavelet', 
                                                    'channel_5_hjorth_mobility_parameter', 'channel_5_hjorth_complexity_parameter', 'channel_5_integrated_absolute_second_derivative_wavelet', 
                                                    'channel_6_slope_sign_changes', 'channel_6_standard_deviation_wavelet_coefficients', 'channel_6_integrated_absolute_third_derivative_wavelet', 
                                                    'channel_7_mean_absolute_value', 'channel_7_integrated_absolute_third_derivative_wavelet']

lda_forward_selection_27_features_300_150_2db2 =  ['channel_1_hjorth_mobility_parameter', 'channel_1_hjorth_complexity_parameter', 'channel_1_difference_absolute_standard_deviation_value', 'channel_1_myopulse_percentage_rate', 
                                                    'channel_1_integrated_absolute_third_derivative_wavelet', 'channel_2_waveform_length', 'channel_2_slope_sign_changes', 'channel_2_logarithmic_RMS_wavelet_packet_coefficients', 
                                                    'channel_3_difference_absolute_standard_deviation_value', 'channel_3_integrated_absolute_third_derivative_wavelet', 'channel_4_mean_absolute_value', 'channel_4_skewness', 
                                                    'channel_4_zero_crossing_wavelet_coefficients', 'channel_4_integrated_absolute_third_derivative_wavelet', 'channel_5_slope_sign_changes', 'channel_5_hjorth_mobility_parameter', 
                                                    'channel_5_integrated_absolute_second_derivative_wavelet', 'channel_6_slope_sign_changes', 'channel_6_integrated_absolute_third_derivative_wavelet', 
                                                    'channel_7_integrated_absolute_third_derivative_wavelet']

lda_forward_selection_27_features_300_150_2db1 =  ['channel_1_mean_absolute_value', 'channel_1_skewness', 'channel_1_difference_absolute_standard_deviation_value', 'channel_1_myopulse_percentage_rate', 
                                                    'channel_1_normalized_logarithmic_energy_wavelet_packet_coefficients', 'channel_1_integrated_absolute_third_derivative_wavelet', 
                                                    'channel_2_slope_sign_changes', 'channel_2_integrated_absolute_third_derivative_wavelet', 'channel_3_difference_absolute_standard_deviation_value', 
                                                    'channel_3_integrated_absolute_third_derivative_wavelet', 'channel_4_mean_of_peak_values', 'channel_4_slope_sign_changes', 'channel_4_hjorth_mobility_parameter', 
                                                    'channel_4_hjorth_complexity_parameter', 'channel_4_integrated_absolute_third_derivative_wavelet', 'channel_5_hjorth_complexity_parameter', 'channel_5_kurtosis', 
                                                    'channel_5_integrated_absolute_third_derivative_wavelet', 'channel_6_slope_sign_changes', 'channel_6_maximum_absolute_value_wavelet_coefficients', 
                                                    'channel_6_normalized_logarithmic_energy_wavelet_packet_coefficients', 'channel_6_integrated_absolute_third_derivative_wavelet', 'channel_7_mean_wavelet_coefficients', 
                                                    'channel_7_integrated_absolute_second_derivative_wavelet']

# List of top 20 channel features
# ----------------------------------------------
channel_1_2db1_300_150 = ['channel_1_mean_absolute_value', 'channel_1_waveform_length', 'channel_1_zero_crossing', 'channel_1_number_of_peaks', 
                            'channel_1_mean_of_peak_values', 'channel_1_slope_sign_changes', 'channel_1_difference_absolute_mean_value', 'channel_1_skewness', 
                            'channel_1_integrated_absolute_value', 'channel_1_hjorth_mobility_parameter', 'channel_1_hjorth_complexity_parameter', 
                            'channel_1_myopulse_percentage_rate', 'channel_1_difference_variance_version', 'channel_1_waveform_length_wavelet_coefficients', 
                            'channel_1_maximum_absolute_value_wavelet_coefficients', 'channel_1_zero_crossing_wavelet_coefficients', 'channel_1_mean_wavelet_coefficients', 
                            'channel_1_integrated_absolute_third_derivative_wavelet', 'channel_1_integrated_exponential_absolute_values_wavelet', 
                            'channel_1_integrated_exponential_wavelet']

# Rest of the channels are in files

channel_4_2db1_300_150 = ['channel_4_standard_deviation', 'channel_4_variance', 'channel_4_waveform_length', 'channel_4_root_mean_square', 'channel_4_number_of_peaks', 
                        'channel_4_mean_of_peak_values', 'channel_4_mean_firing_velocity', 'channel_4_slope_sign_changes', 'channel_4_difference_absolute_mean_value', 
                        'channel_4_M_i', 'channel_4_hjorth_complexity_parameter', 'channel_4_difference_absolute_standard_deviation_value', 'channel_4_simple_square_integrated', 
                        'channel_4_difference_variance_version', 'channel_4_energy_wavelet_coefficients', 'channel_4_zero_crossing_wavelet_coefficients', 
                        'channel_4_normalized_logarithmic_energy_wavelet_packet_coefficients', 'channel_4_integrated_absolute_third_derivative_wavelet', 
                        'channel_4_integrated_exponential_absolute_values_wavelet', 'channel_4_integrated_exponential_wavelet']

all_channels_2db1_300_150 = ['channel_1_mean_absolute_value', 'channel_1_difference_absolute_standard_deviation_value', 'channel_1_myopulse_percentage_rate', 
                            'channel_1_integrated_absolute_third_derivative_wavelet', 'channel_2_slope_sign_changes', 'channel_2_integrated_absolute_third_derivative_wavelet', 
                            'channel_3_difference_absolute_standard_deviation_value', 'channel_3_integrated_absolute_third_derivative_wavelet', 'channel_4_mean_of_peak_values', 
                            'channel_4_slope_sign_changes', 'channel_4_hjorth_mobility_parameter', 'channel_4_integrated_absolute_third_derivative_wavelet', 
                            'channel_5_hjorth_complexity_parameter', 'channel_5_kurtosis', 'channel_5_integrated_absolute_third_derivative_wavelet', 'channel_6_slope_sign_changes', 
                            'channel_6_maximum_absolute_value_wavelet_coefficients', 'channel_6_integrated_absolute_third_derivative_wavelet', 'channel_7_mean_wavelet_coefficients', 
                            'channel_7_integrated_absolute_second_derivative_wavelet']

# Dictionary with all individual features separated
# ----------------------------------------------
individual_time_features = {feature: [feature] for i, feature in enumerate(time_features)}
individual_time_freq_features = {feature: [feature] for i, feature in enumerate(time_freq_features)}

# List of possible input feature sets (dictionaries) (use control + / to comment and uncomment)
# ----------------------------------------------
# feature_sets_dict = individual_time_features | individual_time_freq_features | {'time_features': time_features, 'time_freq_features': time_freq_features, 'all_features': time_features + time_freq_features}

# feature_sets_dict = {'time_features': time_features, 'time_freq_features': time_freq_features, 'all_features': time_features + time_freq_features}

# feature_sets_dict = {'200-100': lda_forward_selection_25_features_200_100_4db1, '300-150': lda_forward_selection_27_features_300_150_4db1,    # Used to compare windows
#                     '256-10': lda_forward_selection_25_features_365_10_4db1, '100-50': lda_forward_selection_25_features_100_50_4db1,
#                     'all_features': time_features + time_freq_features}

# feature_sets_dict = {'4db1':lda_forward_selection_27_features_300_150_4db1, '4db4': lda_forward_selection_21_features_300_150_4db4,    # Used to compare wavelets
#                     '4db2': lda_forward_selection_27_features_300_150_4db2, '2db1': lda_forward_selection_27_features_300_150_2db1,
#                     '2db4': lda_forward_selection_27_features_300_150_2db4, '2db2': lda_forward_selection_27_features_300_150_2db2,
#                     'all_features': time_features + time_freq_features}

# feature_sets_dict = {'channel_1': channel_1_2db1_300_150,    # Used to compare channels
#                     'channel_4': channel_4_2db1_300_150,
#                     'all_channels': all_channels_2db1_300_150,
#                     'all_features': time_features + time_freq_features}

feature_sets_dict = {'selected_features': lda_forward_selection_27_features_300_150_2db1,    # Used for general comparison
                    'all_features': time_features + time_freq_features}
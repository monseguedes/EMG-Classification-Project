"""
@author: Monse Guedes Ayala
@project: Bioliberty EMG Classification Algorithm

This file is for preparing the limb position dataset for the rest of the analysis. If other datasets were to be used,
they would need similar files to convert the data into the standard format described in Confluence.
"""

import pandas as pd
import matplotlib.pyplot as plt
import os 
import itertools

# Dictionary of equivalence between movement name and number (useful for future analysis)
movements_dic = {'WristFlex': 1, 'WristExten': 2, 'WristPron': 3, 'WristSupi': 4, 
            'ObjectGrip': 5, 'PichGrip': 6, 'HandOpen': 7, 'HandRest': 8}

class File_data:
    """
    Class representing the data file to be converted to standard format.
    
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

    movement: str
        name of the movement associated with the file
    dataframe: dataframe
        original dataframe
    column_names: list
        list of the column names (channel_1, channel_2,...)

    file_name: str
        name of the file used to create the class
    folder_name: str
        name of the folder where the file is at

    Methods
    -------
    create_dataframe
        created dataframe from file
    plot_dataframe
        plots the signals in three ways: all channels, all channels closer look, each channel.
    save_dataframe
        saves dataframe with desired standard format in standard folders.

    """    
    def __init__(self,  no_subject: int, movement: str, no_trial: int, limb_position=2):
        # Representative numbers
        self.no_subject = no_subject 
        self.no_movement = movements_dic[movement]
        self.no_trial = no_trial
        self.no_channels = None

        self.movement = movement
        self.dataframe = None
        self.column_names = None

        # Directory attributes
        self.file_name = 'Pos' + str(limb_position) + '_' + movement + '_-' + str(no_trial) + '.txt'
        if no_subject == 9:
            self.folder_name = 'S' + str(no_subject) + '_Female'
        else:
            self.folder_name = 'S' + str(no_subject) + '_Male'

    def create_dataframe(self):
        """
        Adds column names for channel, and sets index to be 'time', changes 
        file name, and stores the file in a new folder.
        """

        path = 'Datasets/original_datasets/Limb Position/LimbPosition_EMG_dataset/' + self.folder_name + '/'    # Define path
        data = pd.read_csv( path + self.file_name , delimiter='\s+', header=None, on_bad_lines='skip')  # Read file

        self.no_channels = len(data.columns)
        self.columns_name = ['channel_' + str(i) for i in range(1, self.no_channels + 1)]
        
        # Set index and columns names
        data.columns = self.columns_name
        data.reset_index()
        data.index.rename('time')

        self.dataframe = data   # Store dataframe attribute

        return data

    def plot_dataframe(self):
        """
        Plots the EMG signals all together, and sperately for each channel.
        """

        path = 'Datasets/original_datasets/Limb Position/plots/'
        
        #Plot all channels together
        legend_names = ['Channel ' + str(i) for i in range(1, self.no_channels + 1)]
        self.dataframe.plot(alpha= 0.75, linewidth=0.5)
        plt.title('EMG Signal Amplitude All Channels')
        plt.legend(legend_names, bbox_to_anchor=(1, -0.11),fancybox=True, ncol=4)
        plt.xlabel('Time')
        plt.ylabel('EMG Amplitude')
        file_name = 'raw_signal.png'
        plt.savefig(path + file_name, dpi=300, bbox_inches='tight')
        plt.close()

        #Plot all channels together closer look
        legend_names = ['Channel ' + str(i) for i in range(1, self.no_channels + 1)]
        self.dataframe.head(400).plot(alpha= 0.75, linewidth=0.5)
        plt.title('Zoomed In EMG Signal Amplitude All Channels')
        plt.legend(legend_names, bbox_to_anchor=(1, -0.11),fancybox=True, ncol=4)
        plt.xlabel('Time')
        plt.ylabel('EMG Amplitude')
        file_name = 'raw_signal_closer.png'
        plt.savefig(path + file_name, dpi=300, bbox_inches='tight')
        plt.close()

        #Plot channels separately
        colors = ['orange', 'dodgerblue', 'forestgreen', 'orangered', 
                'crimson', 'blue', 'royalblue']
        for i in self.dataframe.columns:
            channel = self.dataframe.columns.get_loc(i) + 1
            self.dataframe[i].plot(linewidth=0.75, color=colors[channel - 1])
            plt.title('EMG Signal Amplitude Channel' + str(channel))
            plt.legend(['Channel ' +  str(channel)])
            plt.xlabel('Time')
            plt.ylabel('EMG Amplitude')
            file_name = 'raw_signal_' + str(channel)+ '.png'
            plt.savefig(path + file_name, dpi=300, bbox_inches='tight')
            plt.close()

            #Plot channels closer look
            self.dataframe[i].head(300).plot(alpha= 0.75, linewidth=0.5, color=colors[channel - 1])
            plt.title('Zoomed In EMG Signal Amplitude Channel' + str(channel))
            plt.legend(['Channel ' +  str(channel)])
            plt.xlabel('Time')
            plt.ylabel('EMG Amplitude')
            file_name = 'raw_signal_closer_' + str(channel) + '.png'
            plt.savefig(path + file_name, dpi=300, bbox_inches='tight')
            plt.close()

    def save_dataframe(self):
        """
        Saves the new dataframe in correct format.

        Note: this function is only for when limb position is equal to 2, since it ignores the limb postion
        """

        path = 'Datasets/standard_datasets/limbposition'
        new_folder_name = 'S' + str(self.no_subject)   # S1, S2,...
        if not os.path.isdir(os.path.join(path, new_folder_name)):  # If there is not a filder with subject number, create it
            os.mkdir(os.path.join(path,new_folder_name))
        new_file_name = 'mov' + str(self.no_movement) + '_' + str(self.no_trial) + '.csv' # mov1_1.csv,...
        
        # Save dataframe inside standard folder, with standard name
        self.dataframe.to_csv(os.path.join(path, new_folder_name, new_file_name))

#Possible variables for movement type, trial number, and subject number.
movements_list = ['WristFlex', 'WristExten', 'WristPron', 'WristSupi', 
    'ObjectGrip', 'PichGrip', 'HandOpen', 'HandRest']
trials_list = range(1,7)
subjects_list = range(1,12)

file = File_data(no_subject=1, movement='ObjectGrip', no_trial=1)
file.create_dataframe()
file.plot_dataframe()

#Converting and saving all dataframes.
for no_subject, movement, no_trial in itertools.product(subjects_list, movements_list, trials_list):
    file = File_data(no_subject=no_subject, movement=movement, no_trial=no_trial)
    file.create_dataframe()
    file.save_dataframe()


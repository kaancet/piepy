import datajoint as dj
import PySimpleGUI as sg
import numpy as np
# from suite2p.gui import gui2p
import os
import subprocess
from datajoint_pipeline.intermediaries.run_fissa_correction import run_fissa_correction
from natsort import natsorted
from behavior_python.detection.wheelDetectionSession import WheelDetectionSession
import matplotlib.pyplot as plt

# TODO automatize / batch the suite2p preprocessing steps
# TODO make the cell selection available to batch processing


schema = dj.schema('two_p')

dj.config['stores'] = {
'nerffs' : dict( # The default storage for this pipeline for now
            protocol='file',
            location= r'E:\Users\Dylan\2p_test') # r"\\nerffs17\boninlabwip2023\data\datajoint_2p")
}

# TODO Consider whether to make this lookup table variable depending on the user
@schema
class DataPaths(dj.Lookup):
    definition = """
    data_type : varchar(50)    # The data the path is for 
    ---
    path_to_data : longblob # The path the largest common folder for the data
    """
    
    contents = [
        ['server_local_23', r"\\nerffs17\boninlabwip2023\data"],
        ['server_nerfcluster_23', "/mnt/boninlab/boninlabwip2023/data"],
        ['server_local_24', r"\\nerfceph01\nerfceph\boninlabwip2024\data"],
        ['server_nerfcluster_24', "/mnt/boninlab/boninlabwip2024/data"],
        ['scratch_dir', "/scratch/dylanm-boninlab/suite2p_Kaan_tests"],
        ['behave_data', r'\\10.86.3.20\data_on_50TB\presentation'] #r'C:\Users\dylan\Documents\Repositories\temp_data\presentation']
    ]


@schema
class Mouse(dj.Manual):
    definition = """
    # Experimental animals
    mouse_id        : varchar(10)                  # Unique animal ID
    ---
    dob=null        : date                         # date of birth
    sex="unknown"   : enum('M', 'F', 'unknown')    # sex
    """


# Could set up a script for automatically generating suggested entries for the tables - or could use Helium to make it easy to enter the data at the time.
# Probably better if it's done along the way
@schema
class Session(dj.Manual):
    definition = """
    # Experimental sessions
    -> Mouse
    session_date        : date              # date
    ---
    experimental_rig    : int           # Experimental rig ID 
    experimenter        : varchar(100)  # Experimenters name
    """

    #Outdated fields to clear away
    # img_data_path       : varchar(250)  # Path to the 2p data
    # behaviour_path      : varchar(250)  # Path to the behavioural data folder
    
# Choosing not to include the data location in Session table - but there could be multiple recordings or behavioural sessions in the same day

# Currently leaving recording and behaviour tables as manual, later can make them imported
@schema
class Recording(dj.Manual):
    definition = """
    # Two-photon recordings taken within an experimental session (day)
    -> Session    
    rec_idx     : int           # index of the recordings on the day
    ---
    depth       : float         # Depth of the recording in um
    plane_step  : float         # The step in distance between planes - may extract it directly from the file header later
    wavelength  : float         # Wavelength of the recording in nm
    laser_power : float         # Power in mW
    recording_name : varchar(250) # Later it can be automatic, based on other fields
    """
# Do we need a path to the recording or can we construct it automatically from the name?


# TODO Add other useful behavioural metrics for getting a session overview
@schema
class Behaviour(dj.Imported):
    definition = """
    # Behavioural data associated with an imaging session
    -> Recording
    ---
    behave_session_path    : varchar(250)  # Path to the raw behavioural data - mostly for bugchecking
    n_trials               : int           # The number of trials in the session
    """
    #base_sess_name         : varchar(100)  # Should be moved earlier? partially redundant - os.path.basename gets the same
  
    # Excluded for now because datajoint does not currently support the saving of polars
    #processed_behaviour    : longblob      # Output from the behaviour pipeline
    
    def make(self, key):
        rec_name = (Recording & key).fetch1('recording_name')

        # Name now starts before preprocessing
        #base_recording_name = rec_name[:rec_name.find('_s2p')]

        # behave_session_dir = (DataPaths & 'data_type = "behave_data"').fetch1('path_to_data')
        
        #TODO: seems like we don't need the full path anymore - did Kaan change something? 
        # Can update or remove from the pipeline.
        #behave_session_dir = r'\\10.86.3.20\data_on_50TB\presentation'
        #key['behave_session_path'] = os.path.join(behave_session_dir, rec_name)
        key['behave_session_path'] = rec_name

        #key['base_sess_name'] = base_recording_name
        # Process the session so that it can be loaded in later
        print(key['behave_session_path'])
        w = WheelDetectionSession(sessiondir=key['behave_session_path'], load_flag=True, skip_google=True)
        behave_df = w.data.data
        try:
            stim_details = behave_df.drop_nulls('stim_start').select(['stim_start', 'contrast', 'spatial_freq', 'temporal_freq', 'stim_pos'])
        except:
            stim_details = behave_df.drop_nulls('t_stimstart_rig').select(['t_stimstart_rig', 'contrast', 'spatial_freq', 'temporal_freq', 'stim_pos'])
        

        key['n_trials'] = behave_df.shape[0]
        self.insert1(key)

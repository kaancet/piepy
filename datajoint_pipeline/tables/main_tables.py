import datajoint as dj
import PySimpleGUI as sg
import numpy as np
# from suite2p.gui import gui2p
import os
import subprocess
from intermediaries.run_fissa_correction import run_fissa_correction
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
    path_to_data : varchar(300) # The path the largest common folder for the data
    """
    contents = [
        ['server_local', r"\\nerffs17\boninlabwip2023\data"],
        ['server_nerfcluster', "/mnt/boninlab/boninlabwip2023/data"],
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
        behave_session_dir = r'\\10.86.3.20\data_on_50TB\presentation'
        key['behave_session_path'] = os.path.join(behave_session_dir, rec_name)

        #key['base_sess_name'] = base_recording_name
        # Process the session so that it can be loaded in later
        print(key['behave_session_path'])
        w = WheelDetectionSession(sessiondir=key['behave_session_path'], load_flag=True)
        behave_df = w.data.data
        stim_details = behave_df.drop_nulls('stim_start').select(['stim_start', 'contrast', 'spatial_freq', 'temporal_freq', 'stim_pos'])

        key['n_trials'] = behave_df.shape[0]
        self.insert1(key)


# # TODO restructure to pull table from Preprocessing - Remove once outdated
# @schema
# class RawTwoData(dj.Imported):
#     definition = """
#     # The output from suite2p registration and ROI extraction
#     -> Recording
#     ---
#     reg_data_path         : varchar(500)  # Path to the registered data - probably important in case it's changed. E.g. cell selection    
#     """

#     class Plane(dj.Part):
#         definition = """
#         # The neuropil corrected dF/F for each plane of the recording
#         -> RawTwoData
#         plane_n                 : int           # The number of the plane from the recording
#         ---
#         cell_select_done        : int           # 0 if not yet complete, 1 otherwise
#         plane_path              : varchar(500)  # The path to the registered data for that specific plane
#         data_df                 : longblob      # The dF/F trace that is output from FISSA
#         """
#         # TODO Maybe remove the cell select done field and find a way to make cell selection optional

#     # TODO add sanity check that the number of planes in the suite2p folder matches the number of planes marked for the recording
#     def make(self, key):
#         core_reg_path = r'\\nerffs17\boninlabwip2023\data\2photon\reg'
#         reg_data_path = os.path.join(core_reg_path, (Recording & key).fetch1('recording_name'), 'suite2p')
#         self.insert1({**key, 'reg_data_path': reg_data_path})

#         # Run through each plane for multi-plane recordings
#         plane_dirs = [f.name for f in os.scandir(reg_data_path) if f.is_dir() and 'plane' in f.name]
#         ordered_planes = natsorted(plane_dirs)
#         for p in ordered_planes:
#             p_num = int(p[5:])

#             sg.theme('DarkAmber')
#             layout = [[sg.Text('Run cell selection?')],
#                       [sg.Button('Ok'), sg.Button('No')] ]
#             window = sg.Window((Recording & key).fetch1('recording_name'), layout)

#             while True:
#                 # TODO make this window more informative - session, plane num, etc
#                 event, values = window.read()
#                 if event == sg.WIN_CLOSED or event == 'No':
#                     break
#                 elif event == 'Ok':
#                     window.close()
#                     plane_path = os.path.join(reg_data_path, 'plane0')

#                     stat_path = os.path.join(plane_path, 'stat.npy')
#                     result = subprocess.run(["python", "run_suite2p_gui.py", "-i", stat_path], capture_output=True)
#                     print(result)
#                     print('Cell selection done')

#                     sampling_freq = 30
#                     """fissa_out = run_fissa_correction(plane_path, sampling_freq)
#                      #print(fissa_out)
#                     print('FISSA finished')

#                     df_data = fissa_out.deltaf_result
#                     # Pull the dF/F trace out of the cell
#                     num_frames = np.sum([np.shape(df_data[0, frames])[1] for frames in range(df_data.shape[1])])
#                     num_neurons = df_data.shape[0]
#                     df_cat = np.empty((num_neurons, num_frames))
#                     for nn in num_neurons:
#                         nn_trace = np.hstack([df_data[0, frames][0,:] for frames in range(df_data.shape(1))])
#                         df_cat[nn, :] = nn_trace

#                     print('df successfully extracted')

#                     # Quick test plot
#                     plt.figure()
#                     plt.plot(fissa_out.raw[200, 20][0,:], label='Raw signal')
#                     plt.plot(fissa_out.result[200, 20][0,:], label='result')
#                     plt.plot(fissa_out.deltaf_raw[200, 20][0,:], label='dr raw')
#                     plt.plot(fissa_out.deltaf_result[200, 20][0,:], label='df result')
#                     plt.legend()
#                     plt.show() """

#                     print(p_num)
#                     print(plane_path)
#                     RawTwoData.Plane.insert1({**key, 'cell_select_done': 1, 'plane_n' : p_num, 'plane_path' : plane_path, 'data_df' : 'tmep'})

#                     # TODO Add a confirmation about having selected cells
#                     break

#                     # Irrelevant now that we have FISSA
#                     # f_data = np.load(os.path.join(plane_path, 'F.npy'))
#                     # f_neu_data = np.load(os.path.join(plane_path, 'F.npy'))
#                     # is_cell = np.load(os.path.join(plane_path, 'iscell.npy'))
#                     #
#                     # #filter for the selected cells
#                     # cell_filt = is_cell[:, 0] == 0
#                     # good_f = np.delete(f_data, cell_filt, axis=0)
#                     # good_f_neu = np.delete(f_neu_data, cell_filt, axis=0)

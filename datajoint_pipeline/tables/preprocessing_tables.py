import os
import numpy as np
from pathlib import Path
from datajoint_pipeline.tables.main_tables import Mouse, Session, Recording, DataPaths
import datajoint as dj
import datajoint_pipeline.helpers.nerfcluster_jobs as cluster_jobs
from datetime import datetime
from natsort import natsorted
import PySimpleGUI as sg
# from suite2p.gui import gui2p
import matplotlib.pyplot as plt
from datajoint_pipeline.intermediaries.run_fissa_correction import run_fissa_correction
from behavior_python.utils import parseStimpyLog
import subprocess

# TODO Ask Bram - what is the locals() doing here?
# schema = dj.schema('DJEphys', locals())
schema = dj.schema('two_p_experiments')


# TODO Add function to set up suite2p settings for preprocessing
# TODO could add a gui or some other way of more easily inputting/modifying suite2p settings
@schema
class Preprocessing(dj.Computed):
    definition = """
            -> Recording
            ---
            reg_file_name       : varchar(250)      # The foldername that will be output from the nerfcluster   
    """
    #, server_nerfcluster, server_local):
    def write_suite2p_config(self, key):

        purge_previous_results = True

        # TODO move several of these fields to a lookup table
        server_local = (DataPaths & 'data_type = "server_local"').fetch1('path_to_data')#r"\\nerffs17\boninlabwip2023\data"
        server_nerfcluster = (DataPaths & 'data_type = "server_nerfcluster"').fetch1('path_to_data')#r"/mnt/boninlab/boninlabwip2023/data"

        rec_name = (Recording & key).fetch1('recording_name')
        bash_filename = f'preprocessing_{rec_name}.sh'
        bash_path = os.path.join('2photon', "raw", rec_name, bash_filename)

        bash_path_local = os.path.join(server_local, bash_path)
        bash_path_nerfcluster = os.path.join(server_nerfcluster, bash_path).replace("\\","/")    
        suite2p_json_path = os.path.join(server_nerfcluster, r"user/Dylan/2photon/test_suite2p_settings.json").replace("\\","/") 
        scratch_dir = (DataPaths & 'data_type = "scratch_dir"').fetch1('path_to_data')
        mnt_dir = os.path.join(server_nerfcluster, '2photon/reg').replace("\\","/") 


        recording_root = os.path.join(server_nerfcluster, '2photon/raw', rec_name).replace("\\","/") 
        rec_root_local = os.path.join(server_local, '2photon/raw', rec_name)
        sess_dir = [f.name for f in os.scandir(rec_root_local) if f.is_dir() and 'run' in f.name]
        sess_file = [f.name for f in os.scandir(os.path.join(rec_root_local, sess_dir[0])) if '.sbx' in f.name]
        recording_path = os.path.join(recording_root, sess_dir[0]).replace("\\","/")

        user_email = "myersj94@imec.be"

        # TODO check if results already exist


        # Make and save the bash file that will run suite2p on the cluster 
        with open(bash_path_local, 'w', newline='\n') as bash_file:
            bash_file.write("#!/bin/bash\n")
            bash_file.write("set -e\n")
            if purge_previous_results:
                bash_file.write(f"rm -rf {scratch_dir}/{rec_name}_*\n\n")

            bash_file.write("nerf_suite2p_launch_slurm.py")
            bash_file.write(f"   --input    '{recording_path}'")
            bash_file.write(f"   --out_directory    '{scratch_dir}'")
            bash_file.write(f"   --out_mounted_directory    '{mnt_dir}'")
            bash_file.write(f"   --job_name    '{rec_name}'")
            bash_file.write(f"   --input_json    '{suite2p_json_path}'")
            bash_file.write(f"   --email    '{user_email}'")
            # bash_file.write("   --use_long_partition=False")
            bash_file.close()


        # TODO print some paths for the user to see that things connect up

        return bash_path_nerfcluster

    def preprocess_suite2p(self, key):
        
        bash_path = self.write_suite2p_config(key)
        client = cluster_jobs.initialize_client("nerfnode01", "dylanm-boninlab", "Tatt3r3dT3akTr335!")

        commands = [
            f"bash {bash_path}"
        ]
        cluster_jobs.execute_commands(client, commands)

        # For now I'm not sure I need to wait on the cluster to be finished - this way we can run in batch and check later
        """ while not os.path.exists(os.path.join(server_local, raw_path, 'done.txt')):
            # Check that nerfcluster job is still running
            commands = [
                f"squeue"
            ]
            nerfcluster_jobs.execute_commands(client, commands)
            print("Waiting for file to be created")
            time.sleep(60*5) """
        
        current_datetime = datetime.now()
        key['reg_file_name'] = (Recording & key).fetch1('recording_name') + '_s2p_' + current_datetime.strftime("%d-%m-%Y")
        self.insert1(key)
        print("Populated", key)

    def make(self, key):
        print("Preprocessing for key:", (Recording & key).fetch1('recording_name'))
        # TODO add some separation for the type of session here if needed
        self.preprocess_suite2p(key)


# TODO Find out it this can be made robust so that you don't need to add all the planes at once. - Maybe with a "cell_selection_done" field after all
@schema
class BasicRecordingData(dj.Computed):
    definition = """
    # The output from suite2p registration and ROI extraction
    -> Preprocessing
    ---
    n_planes                : int               # Number of planes recorded    
    frame_times             : longblob          # The calculated times of the frames from riglog
    frame_rate              : float             # The overall frame rate
    frame_interval          : float             # The average time between frames
    plane_frame_rate        : float             # The frame rate for each plane
    plane_frame_interval    : float             # The average time between frames for each plane
    flyback_frames          : int               # The number of flyback frames for the recording
    """

    def make(self, key):
        # Get the major paths from the lookup table for the data we will need
        core_reg_path = os.path.join((DataPaths & 'data_type = "server_local"').fetch1('path_to_data'), r'2photon\reg')
        reg_data_path = os.path.join(core_reg_path, (Preprocessing & key).fetch1('reg_file_name'), 'suite2p')
        
        # Get the paths to the suite2p output for each plane for this recording and find the number of planes
        # TODO add sanity check that the number of planes in the suite2p folder matches the number of planes marked for the recording
        plane_dirs = [f.name for f in os.scandir(reg_data_path) if f.is_dir() and 'plane' in f.name]
        ordered_planes = natsorted(plane_dirs)
        num_of_planes = len(ordered_planes)

        # Get the times of the 2p frames from the riglog
        behave_session_dir = os.path.join((DataPaths & 'data_type = "behave_data"').fetch1('path_to_data'), (Recording & key).fetch1('recording_name'))
        riglog = [f.name for f in os.scandir(behave_session_dir) if 'riglog' in f.name]
        rig_data = parseStimpyLog(os.path.join(behave_session_dir, riglog[0]))
        frame_times = rig_data[0]['imaging'][:,'duinotime'].to_numpy()

        # Get the average time between frame, the frame rate, and the time between frames for each plane
        frame_interval = np.mean(np.diff(frame_times, axis=0))
        frame_rate = 1000 / frame_interval
        plane_rate = frame_rate / num_of_planes
        plane_frame_interval = 1000 / plane_rate

        # Store the data about the recording for now.
        self.insert1({**key, 
            'n_planes' : num_of_planes,
            'frame_times': frame_times, 
            'frame_interval' : frame_interval,
            'frame_rate' : frame_rate,
            'plane_frame_rate' : plane_rate,
            'plane_frame_interval' : plane_frame_interval,
            'flyback_frames' : 0
            })

@schema
class Plane(dj.Imported):
    definition = """
    # The neuropil corrected dF/F for each plane of the recording
    -> BasicRecordingData
    plane_n                 : int           # The number of the plane from the recording
    ---
    plane_path              : varchar(500)  # The path to the registered data for that specific plane
    plane_depth             : float         # the recording depth of the plane on the 2p microscope
    """

    def make(self, key):
        
        # Get the major paths from the lookup table for the data we will need
        core_reg_path = os.path.join((DataPaths & 'data_type = "server_local"').fetch1('path_to_data'), r'2photon\reg')
        reg_data_path = os.path.join(core_reg_path, (Preprocessing & key).fetch1('reg_file_name'), 'suite2p')
        
        # Get the paths to the suite2p output for each plane for this recording and find the number of planes
        # TODO add sanity check that the number of planes in the suite2p folder matches the number of planes marked for the recording
        plane_dirs = [f.name for f in os.scandir(reg_data_path) if f.is_dir() and 'plane' in f.name]
        ordered_planes = natsorted(plane_dirs)

        base_depth = (Recording & key).fetch1('depth')
        #plane_depth_step =  (Recording & key).fetch1('plane_step')
        # Run through each plane for multi-plane recordings
        # TODO add a skip for flybak frames in recordings
        for p in ordered_planes:
            p_num = int(p[5:])

            # TODO consider making this path relative
            plane_path = os.path.join(reg_data_path, 'plane' + str(p_num))
            plane_depth = 200
            #Later, once we've updated the table
            # plane_depth = base_depth + (plane_depth_step * p_num) # Should work because it's 0 indexed
            self.insert1({**key, 'plane_n' : p_num, 'plane_path' : plane_path, 'plane_depth': plane_depth})
        print(f'{p_num + 1} planes detected and logged for {(Recording & key).fetch1("recording_name")}')
    


@schema
class RoiSelection(dj.Computed):
    definition = """
    # Table to run cell selection for all frames 
    -> Plane
    ---
    num_rois            : int               # The number of ROIs that were selected as cells during cell selection
    """

    def make(self, key):

        stat_path = os.path.join((Plane & key).fetch1('plane_path'), 'stat.npy')
        iscell_path = os.path.join((Plane & key).fetch1('plane_path'), 'iscell.npy')
        # TODO make sure that it stops if it lands on an error
        result = subprocess.run(["python", r"C:\Users\dylan\Documents\Repositories\two_photon_pipeline_boninlab\intermediaries\run_suite2p_gui.py", "-i", stat_path], capture_output=True)
        print(result)

        sg.theme('DarkAmber')
        layout = [[sg.Text('Cell selection completed?')],
                  [sg.Button('Yes'), sg.Button('No')] ]
        window = sg.Window((Recording & key).fetch1('recording_name'), layout)

        while True:
            # TODO make this window more informative - session, plane num, etc
            # TODO Add an option for if the window is closed and cell selection is incomplete
            event, values = window.read()
            if event == sg.WIN_CLOSED or event == 'No':
                break
            elif event == 'Yes':
                window.close()
                #'if event == 'Ok':
                iscell = np.load(iscell_path)
                n_rois = np.sum(iscell[:,0]==1)
                self.insert1({**key, 'num_rois': n_rois})
                break
        

@schema
class NeuropilCorrection(dj.Computed):
    definition = """
        # The neuropil corrected dF/F for each plane of the recording
        -> RoiSelection
        ---
        data_df                 : longblob     # The dF/F trace that is output from FISSA
        """
    # TODO get @nerffs to work

    def make(self, key):  
        
        print(f"running neuropil correction for plane {(Plane & key).fetch1('plane_n')} of recording {(Preprocessing & key).fetch1('reg_file_name')}")
        sampling_freq = (BasicRecordingData & key).fetch1('plane_frame_interval')
        print(f"Sampling frequency: {sampling_freq}")
        
        plane_path = (Plane & key).fetch1('plane_path')

        fissa_out, rois = run_fissa_correction(plane_path, sampling_freq)
            #print(fissa_out)
        print('FISSA finished')

        df_data = fissa_out.deltaf_result
        # Pull the dF/F trace out of the cell
        num_frames = np.sum([np.shape(df_data[0, frames])[1] for frames in range(df_data.shape[1])])
        num_neurons = df_data.shape[0]
        df_cat = np.empty((num_neurons, num_frames))
        #print(df_cat.shape[:])
        for nn in range(num_neurons):
            nn_trace = np.hstack([df_data[nn, frames][0,:] for frames in range(df_data.shape[1])])
            df_cat[nn, :] = nn_trace

        # TODO add spike extraction
        self.insert1({**key, 'data_df' : df_cat})
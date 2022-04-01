from .stacks import *
from .myio import *
from .retinoutils import *
from ..core.session import Session
from ..utils import *


class OnePSession(Session):
    def __init__(self,sessiondir,run_name):
        super().__init__(sessiondir)
        self.name = run_name
        self.get_runno()
        self.run_path = pjoin(self.data_paths.camPath,self.name)
        
        self.get_camlog()
        self.get_stimpy_logs()
        
        self._compare_logs()
        self.get_reference_image()
        
    def __repr__(self):
        r = f'''OneP Session Run {self.run_no} for {self.sessiondir}:\n'''
        for s in self.__dict__:
            if 'path' in s:
                r += f'''- {s} : {getattr(self,s,None)}\n'''
        return r
    
    def get_trial_average(self,fixed_frame_count:int=0):
        """ This is the main method that stim_mat, binary file and tiff stack is created 
        and the average is calculated for the run """
        
        # make the stim matrix
        screen_times = self.get_screen_epochs()
        self.correct_stimlog_times(screen_times)
        self.make_stim_matrix()
        min_frame_count = self.get_min_frame_per_repeat(fixed_count=fixed_frame_count)
        frame_starts = self.stim_mat[:,4]
        
        # get tiff stack
        tf_stack = load_stack(self.run_path, nchannels=1)

        # pool all the required frames
        all_frames = []
        for i in frame_starts:
            all_frames.extend(np.arange(i,i+min_frame_count))
        all_frames

        # group frames to respective tiff files indices
        tif_dict = {}
        for f_idx in range(len(tf_stack.frames_offset)-1):
            low = tf_stack.frames_offset[f_idx]
            high = tf_stack.frames_offset[f_idx+1]

            tif_dict[f_idx] = [i for i in all_frames if low <= i < high]

        temp_mat = np.zeros((len(all_frames),1,tf_stack.shape[2],tf_stack.shape[3]))
        # temp matrix
        repeat_mat = np.zeros((len(tif_dict),       # repeats/trial count 
                            min_frame_count,     # min frame count per repeat/trial
                            1,                   # channel(gray)
                            tf_stack.shape[2],   # height
                            tf_stack.shape[3]),  # width
                            dtype='uint32'
                            )
        pbar = tqdm(range(len(tif_dict)))

        bookmark = 0
        for i in pbar:
            pbar.set_description(f'Reading frames from tiff files {bookmark} / {temp_mat.shape[0]}')
            
            # read all the required frames from a single tiff, 
            read_frames = tf_stack[tif_dict[i]]
            temp_mat[bookmark:bookmark+read_frames.shape[0],:,:] = read_frames
            bookmark += read_frames.shape[0]

        npbar = tqdm(range(len(tif_dict)))
        for mult,j in enumerate(npbar):
            
            idx1 = mult*min_frame_count
            idx2 = mult*min_frame_count +min_frame_count
            npbar.set_description(f'Putting frames to correct order {idx1} - {idx2}')
            repeat_mat[j,:,0,:,:] = temp_mat[idx1:idx2,0,:,:]
            
        mean_repeat = np.mean(repeat_mat,axis=0)
        mean_repeat = mean_repeat - np.min(mean_repeat,axis=0)
        
        self.trial_average = mean_repeat.astype('uint32')
        
        #save average
        self.save_avg(frame_count=fixed_frame_count)

    
    def save_avg(self,**kwargs):
        run_analysis_dir = pjoin(self.data_paths.analysisPath,self.name)
        mov_dir = pjoin(run_analysis_dir,"movies")
        if not os.path.exists(mov_dir):
            os.makedirs(mov_dir)
        save_name = pjoin(mov_dir,f'avg_{self.run_no}')
        for k,v in kwargs.items():
            save_name += f'_{k}{v}'
        save_name += '.tif'
        tf.imwrite(save_name,self.trial_average)
        print(f'Saved {save_name}')
        
    def get_reference_image(self):
        """ Reads the reference vasculature image"""
        ref_dir = self.data_paths.camPath
        
        ref_img = [i for i in os.listdir(ref_dir) if i.endswith('.tif')]
        
        if len(ref_img):
            if len(ref_img) > 1:
                print(f' >>>WARNING<<< Multiple reference images found at {self.data_paths.camPath}, using: {ref_img[0]}')
            ref_img_path = pjoin(ref_dir,ref_img[0])
            print(f'Found reference image at {ref_img_path}')
            self.reference_img = tf.imread(ref_img_path)
            try:
                self.reference_img = self.reference_img[0,:,:]
            except IndexError:
                pass
        else:
            print(f' >>>WARNING<<< No reference image found at {self.data_paths.camPath}')
        
    def get_runno(self):
        r_list = self.name.split('_')
        self.run_no = int(r_list[0].strip('run'))
        
    def get_camlog(self):
        dir_content = os.listdir(self.run_path)

        # check/get camlog
        cam_log = [f for f in dir_content if f.endswith('log')]
        
        if len(cam_log)==1:
            self.path_camlog = pjoin(self.run_path,cam_log[0])
            self.camlog,_ = parseCamLog(self.path_camlog)
        elif len(cam_log) > 1:
            raise IOError(f'Multiple camlogs present in run directory')
        elif len(cam_log) == 0:
            raise IOError(f'!!No camlogs present in run directory!!')
            
    def get_stimpy_logs(self):
        super().read_data()
        self.filter_vstim()
        
    def _compare_logs(self):
        """ Compares camlog and stimlog['cam']"""
        camlog_times = self.camlog['timestamp'].to_numpy() / 10 #in 10ths of microseconds
        camlog_frames = len(camlog_times)
        
        stimlog_cam_times = self.rawdata['cam3']['duinotime'].to_numpy()
        stimlog_cam_frames = len(stimlog_cam_times)
        
        if camlog_frames < stimlog_cam_frames:
            print(f'{stimlog_cam_frames - camlog_frames} logged frame(s) not recorded by camlog!!')
        elif camlog_frames > stimlog_cam_frames:
            print(f'{camlog_frames - stimlog_cam_frames} logged frame(s) not logged in riglog!')
            
            self.camlog = self.camlog.iloc[:stimlog_cam_frames]      # removing last recorded frame
            camlog_times = self.camlog['timestamp'].to_numpy() / 10 # in 10ths of microseconds
            camlog_frames = len(camlog_times)                       # update length
            if camlog_frames==stimlog_cam_frames:
                print('camlogs are equal now!')

    def get_screen_epochs(self):
        """ Returns all screen times and a matrix that has 4 columns:
            0-trial_no
            1-screen start
            2-screen end
            3-screen duration
        """
        screen_df = self.rawdata['screen']
        screen_times = screen_df['duinotime'].to_numpy()
        
        idx = 1 if screen_df['value'].iloc[0] == 0 else 0 # sometimes screen events have a 0 value entry at the beginning
        
        starts = screen_df['duinotime'].iloc[idx::2].to_numpy().reshape(-1,1)
        ends = screen_df['duinotime'].iloc[idx+1::2].to_numpy().reshape(-1,1)
        
        trials = np.arange(1,len(starts)+1).reshape(-1,1)
        
        self.screen_epochs = np.hstack((trials,
                                        starts,
                                        ends,
                                        ends-starts)).astype('int')
        return screen_times
    
    def add_iStim(self):
        """ Patch function to add iStim in logs that don't have it, may cause problems..."""
        istim = np.zeros((len(self.rawdata['vstim']),1),dtype='int')
        self.rawdata['vstim'].loc[:,'iStim'] = istim
        
    def filter_vstim(self):
        """ Removes the blank time in the beginning and converts str to numeric values"""
        vstim = self.rawdata['vstim']
        vstim.loc[:,'photo'] = pd.to_numeric(vstim.photo,errors='coerce')
        vstim = vstim[~np.isnan(vstim['photo'])]
        self.rawdata['vstim'] = vstim
        try:
            vstim.loc[:,'iStim'] = pd.to_numeric(vstim.iStim,errors='coerce')
        except AttributeError:
            self.add_iStim()
    
    def correct_stimlog_times(self,screen_times):
        """ Corrects the vstim and stateMachine times to match the arduino timeframe"""
        stim_times = self.rawdata['vstim']['presentTime'].to_numpy() * 1000
        photodiodes =  self.rawdata['vstim']['photo'].to_numpy()
        
        state_times = self.rawdata['stateMachine']['elapsed'].to_numpy()
        
        first_stim_onset = np.round(stim_times[np.where(photodiodes==0)][0])
        photodiode_onset = screen_times[0]
        offset = first_stim_onset - photodiode_onset
        corrected_times = stim_times - offset
        self.rawdata['vstim']['corrected_times'] = corrected_times
        state_corrected = state_times - offset
        self.rawdata['stateMachine']['corrected_times'] = state_corrected
        
    def make_stim_matrix(self):
        """Iterates the stimlog """
        vstim = self.rawdata['vstim']
        stimlog_cam = self.rawdata['cam3']
        state_machine = self.rawdata['stateMachine']
        self.stim_mat = np.zeros((self.screen_epochs.shape[0],7),dtype='int')
        self.stim_mat[:,2] = -1
        for i,row in enumerate(self.screen_epochs):
            
            epoch_times = row[1:3] - [0, 60]
            # mask[np.where(stim_times>=epoch_times[0]) & np.where(stim_times<=epoch_times[1])] = 1
            stim_data = vstim[(vstim['corrected_times'] >= epoch_times[0]) & (vstim['corrected_times'] <= epoch_times[1])]
            
            # this is weird
            # stim_data = stim_data[stim_data['photo']!=1]
            state_data = state_machine[(state_machine['corrected_times'] >= epoch_times[0]) & (state_machine['corrected_times'] <= epoch_times[1])]
            loop_nr = 0
            
            if len(state_data)==0:
                continue
            else:
                uniq_trial = np.unique(state_data['cycle'])
        
            uniq_iStim = np.unique(stim_data['iStim'])
            
            assert len(uniq_iStim) == 1, f'More than 1 unique value in iStim {uniq_iStim}'
            assert len(uniq_trial) == 1, f'More than 1 unique value in iTrial {uniq_trial}'
            u_iStim = uniq_iStim[0]
            u_trial = uniq_trial[0]
            
            #[iStim,]
            epoch_props = [u_iStim,u_trial,0]
            
            stimlog_cam_times = stimlog_cam[(stimlog_cam['duinotime'] >= epoch_times[0]) & (stimlog_cam['duinotime'] <= epoch_times[1])]
            stimlog_cam_times = stimlog_cam_times['value'].to_numpy()
            frame_times = [stimlog_cam_times[0],stimlog_cam_times[-1]]

            # weird way but ok
            self.stim_mat[i,:] = [i+1] + epoch_props + frame_times + [np.diff(frame_times)[0]+1]
            
        # drop -1 rows
        mask = np.ones((self.stim_mat.shape[0]),dtype='bool')
        mask[np.where(self.stim_mat[:,2]==-1)] = False
        self.stim_mat = self.stim_mat[mask,:]
        print('Generated stimulus matrix')
            
    def get_min_frame_per_repeat(self,fixed_count:int=0):
        """ Gets the minimum frame count for averages """
        if fixed_count < 0:
            raise ValueError(f'Fixed frame count can"t be negative. got {fixed_count}')
        
        if fixed_count == 0:
            # return the minimum frame count
            return np.min(self.stim_mat[:,6])
        else:
            if fixed_count > np.min(self.stim_mat[:,6]):
                print(f'>>> WARNING <<< Fixed frame count is too low; there are trials with less then {fixed_count} frames, using {np.min(self.stim_mat[:,6])} frames instead.')
            else:
                return fixed_count
            
    def get_memmap_from_binary(self):
        return mmap_dat(self.binary_fname)
    
    def delete_binary(self):
        os.remove(pjoin(self.run_path,self.binary_fname))
        print('Deleting the temporary binary file')
    
    def make_binary(self):
        # This should normally return a TiffStack because of deleting of binary files after analysis
        tf_stack = load_stack(self.run_path,nchannels=1)
        try:
            self.binary_fname = tf_stack.export_binary(self.run_path)
            print('Creating the the temporary binary file')
            return 1
        except AttributeError:
            self.binary_fname = natsorted(glob(pjoin(self.run_path,'*.bin')))[0]
            print('Found binary')
            return 0

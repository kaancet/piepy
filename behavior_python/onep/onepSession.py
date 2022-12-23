import cv2
from .stacks import *
from .myio import *
from .retinoutils import *
from ..core.session import Session
from ..utils import *
from pystackreg import StackReg

class OnePSession(Session):
    def __init__(self,sessiondir,run_name):
        self.name = run_name
        self.get_runno()
        super().__init__(sessiondir,runno=self.run_no_str)
        
        self.isRegistered = False
        self.run_path = pjoin(self.data_paths.camPath,self.name)
        self.get_stack()
        
        self.get_camlog()
        self.get_stimpy_logs()
        
        self._compare_logs()
        self.get_reference_image()
        self.get_frame_t()
        
    def __repr__(self):
        r = f'''OneP Session Run {self.run_no} for {self.sessiondir}:\n'''
        for s in self.__dict__:
            if 'path' in s:
                r += f'''- {s} : {getattr(self,s,None)}\n'''
        return r
    
    def get_frame_t(self):
        """ Gets the avg frame time from experiment duration and frame count """
        
        avg_ftime = np.mean(np.diff(self.camlog.timestamp))
        if avg_ftime == 0:
            tmp = [i for i in self.camlog_comments if '# [' in i]
            exp_start = dt.strptime(tmp[0].split(']')[0][-8:],'%H:%M:%S')
            exp_end = dt.strptime(tmp[-1].split(']')[0][-8:],'%H:%M:%S')
            
            exp_dur = exp_end - exp_start
            exp_dur = exp_dur.seconds
            
            total_frame_count = len(self.camlog)
            
            self.frame_t = (exp_dur/total_frame_count) # in seconds
        else:
            # potential failsafe for pycams rig differences?
            if avg_ftime<1:
                self.frame_t = avg_ftime 
            else:
                self.frame_t = avg_ftime / 10_000 # pycams measures in 10 microsecond intervals
            
        display(f'Avg. frame time: {self.frame_t*1000} ms, does this make sense?')
    
    def get_stack(self):
        """ Gets the stack as a class attr """
        self.tif_stack = load_stack(self.run_path, nchannels=1)
    
    def _get_avg(self):
        pass
    
    def get_trial_average(self,batch_count:int=None,downsample:int=1,frame_count:int=0,trial_count:int=None,pre_stim_t:float=0,post_stim_t:float=0,overF:bool=False):
        """ This is the main method that stim_mat, binary file and tiff stack is created 
        and the average is calculated for the run """
        
        # make the stim matrix
        screen_times = self.get_screen_epochs()
        self.correct_stimlog_times(screen_times)
        
        extra_start_frames = int(np.round(pre_stim_t / self.frame_t))
        extra_end_frames = int(np.round(post_stim_t / self.frame_t))
        self.make_stim_matrix(extra_start_frames,extra_end_frames)
        min_frame_count = self.get_min_frame_per_repeat(fixed_count=frame_count)
        # TODO: There might be issues here if both pre,post_stim_t and fixed frame are used together
        
        # fix min frames
        for i in range(self.stim_mat.shape[0]):
            if self.stim_mat[i,-1] > min_frame_count:
                self.stim_mat[i,-1] = min_frame_count
                self.stim_mat[i,5] -= 1
        frame_starts = self.stim_mat[:,4] 
        
        # get tiff stack
        self.get_stack() # running this here again because if registering is done this will automatically get the registered stacks

        frame_offsets = self.tif_stack.frames_offset

        # pool all the required frames
        all_frames = []
        if trial_count is not None:
            frame_starts = frame_starts[:trial_count]
            frame_offsets = frame_offsets[:trial_count]
        else:
            trial_count = self.stim_mat.shape[0]

        for i in frame_starts:
            all_frames.extend(np.arange(i,i+min_frame_count))
        end_frames = self.stim_mat[:,5]

        if batch_count > 1:
            
            # make batches of frames
            batch_size = int(len(all_frames) / batch_count)
            print(f'Analyzing in {batch_count} batches [{batch_size}]')
            batches = []
            for i in range(0, len(all_frames), batch_size):
                batches.append(all_frames[i:i + batch_size])
            
            # add the last "remaining" frames to the previous batch
            if len(all_frames) % batch_count:
                batches[-2].extend(batches[-1])
                batches = batches[:-1]

            # adjust batches to align batch length with end of trials
            missing_frames = []
            for b in batches:
                last_frame = b[-1]
                larger_end_frame = end_frames[np.where(end_frames >= last_frame)][0]
                missing_frames.append(larger_end_frame - last_frame)
                
            for i,mf in enumerate(missing_frames,start=1):
                try:
                    batches[i-1].extend(batches[i][:mf])
                    batches[i] = batches[i][mf:]
                except IndexError:
                    pass
        else:
            batches = [all_frames]
            print(f'Analyzing in a single batch [{len(all_frames)}]')
        
        
        mean_mat = np.zeros((min_frame_count,            # min frame count per repeat/trial
                                 1,                          # channel(gray)
                                 int(self.tif_stack.shape[2]/downsample),    # height
                                 int(self.tif_stack.shape[3]/downsample))   # width
                                 )
        prev_last_frame = 0
        for curr_batch in batches:
            # group frames to respective tiff files indices
            tif_dict = {}
            for f_idx in range(len(frame_offsets)-1):
                low = self.tif_stack.frames_offset[f_idx]
                high = self.tif_stack.frames_offset[f_idx+1]

                temp = [i for i in curr_batch if low <= i < high]
                if temp:
                    tif_dict[f_idx] = temp

            # temp matrix
            temp_mat = np.zeros((len(curr_batch),1,int(self.tif_stack.shape[2]/downsample),int(self.tif_stack.shape[3]/downsample)))
        
            pbar = tqdm(list(tif_dict.keys()))
            bookmark = 0 
            for i in pbar:
                pbar.set_description(f'Reading frames from tiff files {curr_batch[bookmark]} / {curr_batch[-1]}')
                
                # read all the required frames from a single tiff, 
                read_frames = self.tif_stack[tif_dict[i]]
                # downsample
                if downsample>1:
                    downed_frames = downsample_movie(read_frames[:,0,:,:])
                else:
                    downed_frames = read_frames[:,0,:,:]
                temp_mat[bookmark:bookmark+read_frames.shape[0],0,:,:] = downed_frames
                bookmark += read_frames.shape[0]
                
            batch_last_frame_div = np.where(end_frames==curr_batch[-1])[0][0]
            npbar = tqdm(range(batch_last_frame_div - prev_last_frame))
            for mult,j in enumerate(npbar):
                
                idx1 = mult*min_frame_count
                idx2 = mult*min_frame_count + min_frame_count 
                npbar.set_description(f'Getting the mean {idx1} - {idx2}')
                if temp_mat[idx1:idx2,:,:,:].shape[0] != mean_mat.shape[0]:
                    print('akjsbdfkajsbkajdf')
                mean_mat[:,0,:,:] += temp_mat[idx1:idx2,0,:,:] / trial_count
                
            prev_last_frame = batch_last_frame_div
            
            del temp_mat
        
        # do df
        if extra_start_frames:
            mean_mat[:,0,:,:] = mean_mat[:,0,:,:] - np.mean(mean_mat[:extra_start_frames+1,0,:,:],axis=0)
        else:
            mean_mat[:,0,:,:] = mean_mat[:,0,:,:] - np.mean(mean_mat[:,0,:,:],axis=0)
            
        if overF: # do df/F
            mean_mat[:,0,:,:] = mean_mat[:,0,:,:] / (np.mean(mean_mat[:extra_start_frames+1,0,:,:],axis=0)+333)
            
        # self.trial_average = mean_mat.astype(np.float16)
        # normalize to 16bit uint
        self.trial_average = cv2.normalize(mean_mat, None, 0, 2**16, cv2.NORM_MINMAX, cv2.CV_16U)
        
        #save average
        self.save_avg(frame_count=frame_count,trial_count=trial_count,start_frames=extra_start_frames,extra_end_frames=extra_end_frames,downsample=downsample,batch_count=batch_count,overF=overF)
   
    def save_avg(self,**kwargs):
        run_analysis_dir = pjoin(self.data_paths.analysisPath,self.name)
        mov_dir = pjoin(run_analysis_dir,"movies")
        if not os.path.exists(mov_dir):
            os.makedirs(mov_dir)
        save_name = pjoin(mov_dir,f'avg_{self.run_no}')
        for k,v in kwargs.items():
            save_name += f'_{k}{int(v)}'
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
        else:
            print(f' >>>WARNING<<< No reference image found at {self.data_paths.camPath}')
        
    def get_runno(self):
        r_list = self.name.split('_')
        self.run_no_str = r_list[0]
        self.run_no = int(r_list[0].strip('run'))
        
    def get_camlog(self):
        dir_content = os.listdir(self.run_path)

        # check/get camlog
        cam_log = [f for f in dir_content if f.endswith('log')]
        
        if len(cam_log)==1:
            self.path_camlog = pjoin(self.run_path,cam_log[0])
            self.camlog,self.camlog_comments,_ = parseCamLog(self.path_camlog)
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
        
        first_stim_onset = np.round(stim_times[np.where(photodiodes==1)][0])
        photodiode_onset = screen_times[0]
        offset = first_stim_onset - photodiode_onset
        corrected_times = stim_times - offset
        self.rawdata['vstim']['corrected_times'] = corrected_times
        state_corrected = state_times - offset
        self.rawdata['stateMachine']['corrected_times'] = state_corrected
        
    def make_stim_matrix(self,extra_start_frames:int=0,extra_end_frames:float=0):
        """Iterates the stimlog """
        vstim = self.rawdata['vstim']
        stimlog_cam = self.rawdata['cam3']
        state_machine = self.rawdata['stateMachine']
        self.stim_mat = np.zeros((self.screen_epochs.shape[0],7),dtype='int')
        self.stim_mat[:,2] = -1
        prev_epoch_end = 0
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
            
            
            # TODO add a failsafe here to not overflow to next stim epoch
            frame_times = [stimlog_cam_times[0] - extra_start_frames, stimlog_cam_times[-1] + extra_end_frames]
            
            if (stimlog_cam_times[0] - extra_start_frames) < prev_epoch_end:
                raise OverflowError(f'The amount of  pre stim frames overflows into the previous stimulus epoch!!! {stimlog_cam_times[0]} - {extra_start_frames}')
            prev_epoch_end = stimlog_cam_times[-1]

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
            min_count = np.min(self.stim_mat[:,6])
        else:
            if fixed_count < np.min(self.stim_mat[:,6]):
                print(f'>>> WARNING <<< Fixed frame count is too low; there are trials with less then {fixed_count} frames, using {np.min(self.stim_mat[:,6])} frames instead.')
                min_count = np.min(self.stim_mat[:,6])
            else:
                min_count = fixed_count
                
        mask = np.ones((len(self.stim_mat)),dtype='bool')
        mask[np.where(self.stim_mat[:,-1]<min_count)] = False
        self.stim_mat = self.stim_mat[mask,:]
        
        return min_count
    
    def register_run(self,overwrite:bool=False):
        """ Registers all the tif stacks """
        # check if a registered run already exists
        reg_path = pjoin("J:\\data\\1photon\\reg",self.sessiondir,self.name)
        if os.path.exists(reg_path):
            if len(os.listdir(reg_path))+1 == len(os.listdir(self.run_path)): #+1 is to account for the camlog
                print(f'A registered 1P directory already exists: {reg_path}')
                if overwrite:
                    print('Overwriting...')
                else:
                    self.isRegistered = True
                    self.run_path = reg_path
                    return None
        else:
            os.makedirs(reg_path)

        pbar = trange(len(self.tif_stack.frames_offset)-1)
        
        sr = StackReg(StackReg.RIGID_BODY)
        for i in pbar: 
            # loop through tif stacks
            stack_idx = [self.tif_stack.frames_offset[i],self.tif_stack.frames_offset[i+1]]
            stack_name = self.tif_stack.filenames[i].split(os.sep)[-1]
            stack = self.tif_stack[stack_idx[0]:stack_idx[1]]
            
            if i==0:
                # get the first frame for registering
                frame0 = stack[0][0,:,:]
            
            # allocate empty matrix size of tif_stacks
            transformed = np.zeros_like(stack,dtype='uint16')
            
            for j in range(stack.shape[0]): 
                # loop through stack frames
                frame = stack[j][0,:,:]
                transformed[j,0,:,:] = sr.register_transform(frame0,frame)
                pbar.set_description(f'Registering {stack_name}, frames {j}/{stack.shape[0]}')
            
            
            # save the registered stack in reg directory
            save_name = pjoin(reg_path,stack_name)
            tf.imwrite(save_name,transformed)
            
        # change the run path to registered directory
        self.isRegistered = True
        self.reg_run_path = reg_path
        
        # move the camlog and update related paths
        self.data_paths.camRegPath = pjoin("J:\\data\\1photon\\reg",self.sessiondir)
        
        

import cv2
from .stacks import *
from .myio import *
from .retinoutils import *
from ..core.session import Session
from ..utils import *
from pystackreg import StackReg

class OnePSession(Session):
    def __init__(self,sessiondir,runno:int=None):
        super().__init__(sessiondir)

        self.isRegistered = False
        self.run_paths = [pjoin(self.data_paths.camPath,p['runname']) for p in self.data_paths.runPaths]
        
        # stuff selected for analysis
        self.selected_run = None
        self.selected_run_path = None
        self.selected_stacks = None
        self.selected_rawdata = None
        self.selected_camlog = None
        self.selected_comments = None
        self.selected_camlog_comments = None
        
        # gets the stacks, camlogs and stimpy logs for all the runs of a session
        self.get_stacks()
        self.get_camlogs()
        self.get_stimpy_logs()
        self.get_reference_image()
        
    def __repr__(self):
        r = f'''OneP Session Run {self.run_no} for {self.sessiondir}:\n'''
        for s in self.__dict__:
            if 'path' in s:
                r += f'''- {s} : {getattr(self,s,None)}\n'''
        return r
    
    def set_run(self,run_no:int) -> None:
        try:
            self.selected_run_path = self.run_paths[run_no]
            self.selected_run = self.selected_run_path.split(os.sep)[-1]
            self.selected_stacks = self.tif_stacks[run_no]
            self.selected_rawdata = self.rawdata[run_no]
            self.selected_camlog = self.camlogs[run_no]
            self.selected_comments = self.comments[run_no]
            self.selected_camlog_comments = self.camlog_comments[run_no]
        except:
            raise IndexError(f"Tried to select the {run_no}th run, seems it doesn't exist, you have {len(self.tif_stacks)} runs in session {self.sessiondir}") 
    
    def get_frame_time(self) -> None:
        """ Gets the avg frame time from experiment duration and frame count """
        avg_ftime = np.mean(np.diff(self.selected_camlog['timestamp']))
        if avg_ftime == 0:
            tmp = [i for i in self.selected_camlog_comments if '# [' in i]
            exp_start = dt.strptime(tmp[0].split(']')[0][-8:],'%H:%M:%S')
            exp_end = dt.strptime(tmp[-1].split(']')[0][-8:],'%H:%M:%S')
            
            exp_dur = exp_end - exp_start
            exp_dur = exp_dur.seconds
            
            total_frame_count = len(self.selected_camlog)
            
            frame_t = (exp_dur/total_frame_count) # in seconds
        else:
            # potential failsafe for pycams rig differences?
            if avg_ftime<1:
                frame_t = avg_ftime 
            else:
                frame_t = avg_ftime / 10_000 # pycams measures in 10 microsecond intervals
        self.frame_t = frame_t   
        display(f'Run:{self.selected_run} Avg. frame time: {self.frame_t*1000} ms')
    
    def get_stacks(self) -> None:
        """ Gets the stack as a np.memmap """
        self.tif_stacks = []
        for r in self.run_paths:
            self.tif_stacks.append(load_stack(r, nchannels=1))
    
    def get_camlogs(self) -> None:
        """ Gets the camlogs from the """
        self.camlogs = []
        self.camlog_comments = []
        for r in self.run_paths:
            dir_content = os.listdir(r)

            # only get the camlog and not the tiffs
            cam_log = [f for f in dir_content if f.endswith('log')]
        
            if len(cam_log)==1:
                p_camlog = pjoin(r,cam_log[0])
                camlog,cmnt,_ = parseCamLog(p_camlog)
                self.camlog_comments.append(cmnt)
                self.camlogs.append(camlog)
            elif len(cam_log) > 1:
                raise IOError(f'Multiple camlogs present in run directory {r}')
            elif len(cam_log) == 0:
                raise IOError(f'!!No camlogs present in run directory!! {r}')
    
    def get_stimpy_logs(self) -> None:
        super().read_data()
        self.filter_vstim()
    
    def _generate_frames_for_analysis(self, 
                                      pre_stim_t:float, 
                                      post_stim_t:float,
                                      fixed_dur:int = None) -> None:

        extra_start_frames = int(np.round(pre_stim_t / self.frame_t))
        extra_end_frames = int(np.round(post_stim_t / self.frame_t))
    
        #!!
        self.make_stim_matrix(extra_start_frames,extra_end_frames)
        
        if fixed_dur is None:
            # return the minimum frame count
            min_count = np.min(self.stim_mat[:,6])
        else:
            fixed_count_frames = int(np.round(fixed_dur / self.frame_t))
            if fixed_count_frames < 0:
                raise ValueError(f'Fixed frame count can"t be negative. got {fixed_count_frames}')
            min_count = fixed_count_frames
        
        # getting rid of trials with less then minimum frame (for recordings during the task) 
        mask = np.ones((len(self.stim_mat)),dtype='bool')
        mask[np.where(self.stim_mat[:,-1]<min_count)] = False
        self.stim_mat = self.stim_mat[mask,:]
        
        # fix min frames
        for i in range(self.stim_mat.shape[0]):
            if self.stim_mat[i,-1] > min_count:
                frame_diff = self.stim_mat[i,-1] - min_count
                self.stim_mat[i,-1] = min_count
                # remove frames from the end
                self.stim_mat[i,5] -= frame_diff
        self.min_frame_count = min_count
        self.pre_stim_frames = extra_start_frames
          
    def get_all_run_averages(self,**kwargs):
        for run_i,run_path in enumerate(self.run_paths):
            # set the run
            self.set_run(run_i)
            # compare logs and create the matrix that will be used for clustering the frames in trials
            self._compare_logs()
            self.get_frame_time()
            
            # make the stim matrix and set the frames for analysis
            self._get_screen_epochs()
            self._correct_stimlog_times()
            
            self._generate_frames_for_analysis(pre_stim_t=kwargs.get('pre_stim_t',0),
                                               post_stim_t=kwargs.get('post_stim_t',0),
                                               fixed_dur=kwargs.get('fixed_dur',None))
            
            trial_avg = self.get_run_trial_average(batch_count=kwargs.get('batch_count',1),
                                                   downsample=kwargs.get('downsample',1),
                                                   trial_count=kwargs.get('trial_count',None))
            
            # do df
            if self.pre_stim_frames:
                trial_avg[:,0,:,:] = trial_avg[:,0,:,:] - np.mean(trial_avg[:self.pre_stim_frames+1,0,:,:],axis=0)
            else:
                trial_avg[:,0,:,:] = trial_avg[:,0,:,:] - np.mean(trial_avg[:,0,:,:],axis=0)
                
            if kwargs.get('dff',True): # do df/F
                trial_avg[:,0,:,:] = trial_avg[:,0,:,:] / (np.mean(trial_avg[:self.pre_stim_frames+1,0,:,:],axis=0)+3)
                
            # saving
            # self.trial_average = mean_mat.astype(np.float16)
            # normalize to 16bit uint
            trial_avg = cv2.normalize(trial_avg, None, 0, 2**16, cv2.NORM_MINMAX, cv2.CV_16U)
            
            #save average
            self.save_avg(trial_avg,run_no=run_i,**kwargs)
            
    def get_run_trial_average(self,
                              batch_count:int=1,
                              downsample:int=1,
                              trial_count:int=None) -> np.ndarray:
        """ This is the main method that stim_mat, binary file and tiff stack is created 
        and the average is calculated for the run """
        # get tiff stack
        #self.get_stack() # running this here again because if registering is done this will automatically get the registered stacks

        frame_offsets = self.selected_stacks.frames_offset
        frame_starts = self.stim_mat[:,4] 
        # pool all the required frames
        all_frames = []
        if trial_count is not None:
            frame_starts = frame_starts[:trial_count]
            frame_offsets = frame_offsets[:trial_count]
        else:
            trial_count = self.stim_mat.shape[0]

        for i in frame_starts:
            all_frames.extend(np.arange(i,i+self.min_frame_count))
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
        
        mean_mat = np.zeros((self.min_frame_count,            # min frame count per repeat/trial
                            1,                          # channel(gray)
                            int(self.selected_stacks.shape[2]/downsample),    # height
                            int(self.selected_stacks.shape[3]/downsample))   # width
                            )
        prev_last_frame = 0
        for curr_batch in batches:
            # group frames to respective tiff files indices
            tif_dict = {}
            for f_idx in range(len(frame_offsets)-1):
                low = self.selected_stacks.frames_offset[f_idx]
                high = self.selected_stacks.frames_offset[f_idx+1]

                temp = [i for i in curr_batch if low <= i < high]
                if temp:
                    tif_dict[f_idx] = temp

            # temp matrix
            temp_mat = np.zeros((len(curr_batch),1,int(self.selected_stacks.shape[2]/downsample),int(self.selected_stacks.shape[3]/downsample)))
        
            pbar = tqdm(list(tif_dict.keys()))
            bookmark = 0 
            for i in pbar:
                pbar.set_description(f'Reading frames from tiff files {curr_batch[bookmark]} / {curr_batch[-1]}')
                
                # read all the required frames from a single tiff, 
                read_frames = self.selected_stacks[tif_dict[i]]
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
                
                idx1 = mult*self.min_frame_count
                idx2 = mult*self.min_frame_count + self.min_frame_count 
                npbar.set_description(f'Getting the mean {idx1} - {idx2}')
                if temp_mat[idx1:idx2,:,:,:].shape[0] != mean_mat.shape[0]:
                    print('\n\n !! !!\n\n')
                mean_mat[:,0,:,:] += temp_mat[idx1:idx2,0,:,:] / trial_count
                
            prev_last_frame = batch_last_frame_div
            
            del temp_mat
        return mean_mat   
   
    def save_avg(self,mov:np.ndarray,run_no:int,**kwargs):
        run_analysis_dir = pjoin(self.data_paths.savePath,self.selected_run)
        mov_dir = pjoin(run_analysis_dir,"movies")
        if not os.path.exists(mov_dir):
            os.makedirs(mov_dir)
        save_name = pjoin(mov_dir,f'avg_{run_no}')
        for k,v in kwargs.items():
            save_name += f'_{k}{int(v)}'
        save_name += '.tif'
        tf.imwrite(save_name,data=mov)
        print(f'Saved {save_name}')
        
    def get_reference_image(self) -> None:
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
        
    def _compare_logs(self) -> None:
        """ Compares camlog and stimlog['cam']"""
        camlog_times = self.selected_camlog['timestamp'].to_numpy() / 10 #in 10ths of microseconds
        camlog_frames = len(camlog_times)
        
        stimlog_cam_times = self.selected_rawdata['cam3']['duinotime'].to_numpy()
        stimlog_cam_frames = len(stimlog_cam_times)
        
        if camlog_frames < stimlog_cam_frames:
            print(f'{stimlog_cam_frames - camlog_frames} logged frame(s) not recorded by camlog!!')
        elif camlog_frames > stimlog_cam_frames:
            print(f'{camlog_frames - stimlog_cam_frames} logged frame(s) not logged in riglog!')
            
            self.selected_camlog = self.selected_camlog.slice(0,stimlog_cam_frames)     # removing last recorded frame
            camlog_times = self.selected_camlog['timestamp'].to_numpy() / 10 # in 10ths of microseconds
            camlog_frames = len(camlog_times)                       # update length
            if camlog_frames==stimlog_cam_frames:
                print('camlogs are equal now!')

    def _get_screen_epochs(self) -> None:
        """ Creates screen times array and a screen epochs matrix that has 4 columns:
            0-trial_no
            1-screen start
            2-screen end
            3-screen duration
        """
        screen_df = self.selected_rawdata['screen']
        idx = 1 if screen_df[0,'value'] == 0 else 0 # sometimes screen events have a 0 value entry at the beginning
        
        starts = screen_df.slice(offset=idx).gather_every(2)
        ends = screen_df.slice(offset=idx+1).gather_every(2)
        ends = ends.drop('code')
        ends = ends.rename({'timereceived':'timereceived_end',
                            'duinotime':'duinotime_end'})
        
        epoch_df = starts.join(ends,on='value',how='left')
        epoch_df = epoch_df.with_columns((pl.col('duinotime_end')-pl.col('duinotime')).alias('screen_duration'))
        
        if epoch_df[1,'screen_duration'] - epoch_df[0,'screen_duration']>1000:
            # sometimes for some reason the first trial is an extra
            epoch_df = epoch_df[1:]
            
        trials = pl.Series('trial_no',np.arange(1,len(epoch_df)+1))
        epoch_df = epoch_df.with_columns(trials)
        
        self.screen_epochs = epoch_df.select(['trial_no','duinotime','duinotime_end','screen_duration']).with_columns(pl.col('*').cast(pl.Int64)).to_numpy()
        
    def filter_vstim(self) -> None:
        """ Removes the blank time in the beginning and converts str to numeric values"""
        for i,data in enumerate(self.rawdata):
            vstim = data['vstim']
            vstim = vstim.drop_nulls(subset=['photo'])
            # makes teh iStim column
            if 'iStim' in vstim.columns:
                vstim = vstim.with_columns(pl.col('iStim').cast(pl.Int32))
            else:
                istim = pl.Series('iStim',[0]*len(vstim))
                vstim = vstim.with_columns(istim)
            self.rawdata[i]['vstim'] = vstim
    
    def _correct_stimlog_times(self) -> None:
        """ Corrects the vstim and stateMachine times to match the arduino timeframe"""
        stim_times = self.selected_rawdata['vstim']['presentTime'].to_numpy() * 1000
        photodiodes =  self.selected_rawdata['vstim']['photo'].to_numpy()
        state_times = self.selected_rawdata['statemachine']['elapsed'].to_numpy()
        
        # vstim time
        vstim_onset = np.round(stim_times[np.where(photodiodes==1)][0])
        photodiode_onset = self.screen_epochs[0,1] #first 
        
        offset = vstim_onset - photodiode_onset
        corrected_times = pl.Series('corrected_times',stim_times - offset)
        state_corrected = pl.Series('corrected_times',state_times - offset)
        
        self.selected_rawdata['vstim'] = self.selected_rawdata['vstim'].with_columns(corrected_times)
        self.selected_rawdata['statemachine'] = self.selected_rawdata['statemachine'].with_columns(state_corrected)
        display(f"Corrected the stimlog and statemachine times by shifting {offset} seconds")
           
    def make_stim_matrix(self,extra_start_frames:int=0,extra_end_frames:float=0) -> None:
        """Iterates the screen epoch matrix """
        vstim = self.selected_rawdata['vstim']
        stimlog_cam = self.selected_rawdata['cam3']
        state_machine = self.selected_rawdata['statemachine']
        self.stim_mat = np.zeros((self.screen_epochs.shape[0],7),dtype='int')
        
        self.stim_mat[:,2] = -1
        prev_epoch_end = 0
        for i,row in enumerate(self.screen_epochs):
            
            epoch_times = row[1:3] #- [0, 60] # is this the blank at the end??
            stim_data = vstim.filter((pl.col('corrected_times') >= epoch_times[0]) &
                                     (pl.col('corrected_times') <= epoch_times[1]))
            
            # this is weird
            # stim_data = stim_data[stim_data['photo']!=1]
            state_data = state_machine.filter((pl.col('corrected_times') >= epoch_times[0]) &
                                              (pl.col('corrected_times') <= epoch_times[1]))

            if len(state_data)==0:
                continue
            else:
                uniq_trial = nonan_unique(state_data['cycle'].to_numpy())
        
            uniq_iStim = nonan_unique(stim_data['iStim'].to_numpy())
            
            assert len(uniq_iStim) == 1, f'More than 1 unique value in iStim {uniq_iStim}'
            assert len(uniq_trial) == 1, f'More than 1 unique value in iTrial {uniq_trial}'
            u_iStim = uniq_iStim[0]
            u_trial = uniq_trial[0]
            
            #[iStim,]
            epoch_props = [u_iStim,u_trial,0]
            
            stimlog_cam_data = stimlog_cam.filter((pl.col('duinotime') >= epoch_times[0]) &
                                                  (pl.col('duinotime') <= epoch_times[1]))
            stimlog_cam_times = stimlog_cam_data['value'].to_numpy()
            
            #TODO: add a failsafe here to not overflow to next stim epoch
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
    
    
    def register_run(self,overwrite:bool=False) -> None:
        """ Registers all the tif stacks """
        # check if a registered run already exists
        reg_path = pjoin("J:\\data\\1photon\\reg",self.sessiondir,)
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

        pbar = trange(len(self.selected_stacks.frames_offset)-1)
        
        sr = StackReg(StackReg.RIGID_BODY)
        for i in pbar: 
            # loop through tif stacks
            stack_idx = [self.selected_stacks.frames_offset[i],self.selected_stacks.frames_offset[i+1]]
            stack_name = self.selected_stacks.filenames[i].split(os.sep)[-1]
            stack = self.selected_stacks[stack_idx[0]:stack_idx[1]]
            
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
        
        

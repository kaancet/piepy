import cv2
from polars import DataFrame
from .stacks import *
from .myio import *
from .retinoutils import *
from ..utils import *


class CamDataAnalysis:
    def __init__(self, data:pl.DataFrame,runpath:str) -> None:
        self.data = data
        self.runpath = runpath
    
    def get_tif_stack(self) -> None:
        """ Gets the stack as a np.memmap """
        self.tif_stack = load_stack(self.runpath, nchannels=1)
    
    def get_camlogs(self) -> None:
        """ Parses the camlog """
        dir_content = os.listdir(self.runpath)

        # only get the camlog and not the tiffs
        cam_log = [f for f in dir_content if f.endswith('log')]
    
        if len(cam_log)==1:
            p_camlog = pjoin(self.runpath,cam_log[0])
            self.camlog,self.camlog_comment,_ = parseCamLog(p_camlog)
        elif len(cam_log) > 1:
            raise IOError(f'Multiple camlogs present in run directory {self.runpath}')
        elif len(cam_log) == 0:
            raise IOError(f'!!No camlogs present in run directory!! {self.runpath}')
    
    def get_frame_time(self) -> None:
        """ Gets the avg frame time from experiment duration and frame count """
        avg_ftime = np.mean(np.diff(self.camlog['timestamp']))
        if avg_ftime == 0:
            tmp = [i for i in self.camlog_comment if '# [' in i]
            exp_start = dt.strptime(tmp[0].split(']')[0][-8:],'%H:%M:%S')
            exp_end = dt.strptime(tmp[-1].split(']')[0][-8:],'%H:%M:%S')
            
            exp_dur = exp_end - exp_start
            exp_dur = exp_dur.seconds
            
            total_frame_count = len(self.camlog)
            
            frame_t = (exp_dur/total_frame_count) # in seconds
        else:
            # potential failsafe for pycams rig differences?
            if avg_ftime<1:
                frame_t = avg_ftime 
            else:
                frame_t = avg_ftime / 10_000 # pycams measures in 10 microsecond intervals
        self.frame_t = frame_t * 1000
        display(f'Avg. frame time: {self.frame_t} ms')
        
    def set_minimum_dur(self,duration:float=None) -> None:
        """ Sets all the rows in the frame mat to have the same amount of frames
            If no value provided, uses the trial with minimum number of frames """
        if duration is None:
            frame_count = np.min(self.frame_mat[:,-1])
        else:
            frame_count = int(np.round(duration / self.frame_t))
            if frame_count < 0:
                raise ValueError(f'Fixed frame count can"t be negative. got {frame_count}')
        
        # getting rid of trials with less then minimum frame (for recordings during the task) 
        mask = np.ones((len(self.frame_mat)),dtype='bool')
        mask[np.where(self.frame_mat[:,-1]<frame_count)] = False
        self.frame_mat = self.frame_mat[mask,:]
        
        # fix min frames
        for i in range(self.frame_mat.shape[0]):
            if self.frame_mat[i,-1] > frame_count:
                frame_diff = self.frame_mat[i,-1] - frame_count
                self.frame_mat[i,-1] = frame_count
                # remove frames from the end
                self.frame_mat[i,-2] -= frame_diff
        
    def scale_camlog_timeunit(self,scale_by:float) -> None:
        """ Scales the timestamp by desired value """
        self.camlog = self.camlog.with_columns((pl.col('timestamp')*scale_by).alias('timestamp'))
    

class OnePAnalysis(CamDataAnalysis):
    def __init__(self,data:pl.DataFrame,runpath:str) -> None:
        super().__init__(data,runpath)
        
        self.get_tif_stack()
        self.get_camlogs()
        self.get_frame_time()

    def trial_avg(self, 
                  pre_t:int=100, 
                  post_t:int=0,
                  duration:float=None, 
                  dFoverF:bool = True,
                  **kwargs) -> None:
        """ """
        # make the stim matrix
        self.make_frame_matrix()
        
        # add pre and post frames
        pre = int(np.round(pre_t / self.frame_t))
        post = int(np.round(post_t / self.frame_t))
        
        # column (-3) is start_frame, column (-2) is end_frame
        self.frame_mat[:,-3] = self.frame_mat[:,-3] - pre
        self.frame_mat[:,-2] = self.frame_mat[:,-2] + post
        self.frame_mat[:,-1] = self.frame_mat[:,-1] + (pre + post)
        
        self.set_minimum_dur(duration)
        
        trial_avg = self._get_average(**kwargs)
        
        # dF = F-F0
        if pre:
            trial_avg[:,0,:,:] = trial_avg[:,0,:,:] - np.mean(trial_avg[:pre+1,0,:,:],axis=0)
        else:
            trial_avg[:,0,:,:] = trial_avg[:,0,:,:] - np.mean(trial_avg[:,0,:,:],axis=0)
        
        # dF/F    
        if dFoverF:
            trial_avg[:,0,:,:] = trial_avg[:,0,:,:] / (np.mean(trial_avg[:pre+1,0,:,:],axis=0)+3)
        
        # saving
        # self.trial_average = mean_mat.astype(np.float16)
        # normalize to 16bit uint
        trial_avg = cv2.normalize(trial_avg, None, 0, 2**16, cv2.NORM_MINMAX, cv2.CV_16U)
        
        return trial_avg
    
    def make_frame_matrix(self) -> None:
        """ Makes a frame matrix to be used in trial averaging """
        stimmed_frames = self.data.filter(pl.col('onepcam_frame_ids').list.len()!=0)
        stimmed_frames = stimmed_frames.select(['trial_no','onepcam_frame_ids'])
        
        self.frame_mat = np.zeros((len(stimmed_frames),len(stimmed_frames.columns)+3),dtype=int)
        for i,row in enumerate(stimmed_frames.iter_rows()):
            frm = [row[1][0], row[1][1], row[1][1]-row[1][0]]
            tmp = [i+1, row[0], *frm]
            # [i, trial_no, state_outcome, frame_start, frame_end, frame_diff]
            self.frame_mat[i,:] = tmp
    
    def _get_average(self,
                     batch_count:int=1,
                     downsample:int=1,
                     trial_count:int=None) -> np.ndarray:
        """ This is the main method that frame_mat, binary file and tiff stack is created 
        and the average is calculated for the run """
        
        frame_offsets = self.tif_stack.frames_offset
        frame_starts = self.frame_mat[:,-3]
        frame_count = self.frame_mat[:,-1][0]
        # pool all the required frames
        all_frames = []
        if trial_count is not None:
            frame_starts = frame_starts[:trial_count]
            frame_offsets = frame_offsets[:trial_count]
        else:
            trial_count = self.frame_mat.shape[0]

        for i in frame_starts:
            all_frames.extend(np.arange(i,i+frame_count))
        end_frames = self.frame_mat[:,-2]

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
        
        mean_mat = np.zeros((frame_count,            # min frame count per repeat/trial
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
            
            # TODO: Test this with batches
            try:
                batch_last_frame_div = np.where(end_frames==curr_batch[-1])[0][0]
            except:
                batch_last_frame_div = np.where(end_frames==end_frames[-1])[0][0]
            
            npbar = tqdm(range(batch_last_frame_div - prev_last_frame))
            for mult,j in enumerate(npbar):
                
                idx1 = mult*frame_count
                idx2 = mult*frame_count + frame_count 
                npbar.set_description(f'Getting the mean {idx1} - {idx2}')
                if temp_mat[idx1:idx2,:,:,:].shape[0] != mean_mat.shape[0]:
                    print('\n\n !! !!\n\n')
                mean_mat[:,0,:,:] += temp_mat[idx1:idx2,0,:,:] / trial_count
                
            prev_last_frame = batch_last_frame_div
            
            del temp_mat
        return mean_mat
    
    def save_avg(self,mov:np.ndarray,save_path:str,**kwargs) -> None:
        mov_dir = pjoin(save_path,"movies")
        if not os.path.exists(mov_dir):
            os.makedirs(mov_dir)
        save_name = pjoin(mov_dir,f'avg')
        for k,v in kwargs.items():
            save_name += f'_{k}{int(v)}'
        save_name += '.tif'
        tf.imwrite(save_name,data=mov)
        print(f'Saved {save_name}')
        
        
class EyecamAnalysis(CamDataAnalysis):
    def __init__(self, runpath: str, data: DataFrame) -> None:
        super().__init__(runpath, data)
        

class FaceCamAnalysis(CamDataAnalysis):
    def __init__(self, runpath: str, data: DataFrame) -> None:
        super().__init__(runpath, data)
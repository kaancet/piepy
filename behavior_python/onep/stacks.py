import os
import numpy as np
from tqdm import tqdm
import tifffile as tf
from glob import glob
from natsort import natsorted
from os.path import join as pjoin
from .myio import chunk_indices,mmap_dat

def load_stack(foldername, nchannels=None, imager_preview = False):
    ''' 
    Searches the correct format to load from a folder.
    '''
    #    TODO: follow a specific order
    # order = ['binary','tiff','imager','video']
    # First check whats in the folder
    if os.path.isfile(foldername):
        if foldername.endswith('.bin') or foldername.endswith('.dat'): 
            return mmap_dat(foldername)
    # Check binary sequence.
    files = natsorted(glob(pjoin(foldername,'*.bin')))
    if len(files):
        # these don't need channel number because it is written with the filename
        if len(files) == 1:
            return mmap_dat(files[0])
        print('Loading binary stack.')
        return BinaryStack(files) 
    # check tiff sequence
    for ext in ['.TIFF','.TIF','.tif','.tiff']:
        files = natsorted(glob(pjoin(foldername,'*'+ext)))
        if len(files):
            return TiffStack(files, nchannels = nchannels)
    # check imager
    files = natsorted(glob(pjoin(foldername,'Analog*.dat')))
    if len(files):
        return ImagerStack(foldername, imager_preview = imager_preview)
    # check for avi and mov
    for ext in ['.avi','.mov','.mj2']:
        files = natsorted(glob(pjoin(foldername,'*'+ext)))
        if len(files):
            return VideoStack(files, extension = ext, nchannels = nchannels)
    # check for dat
    files = natsorted(glob(pjoin(foldername,'*.dat')))
    if len(files):
        if len(files) == 1:
            return mmap_dat(files[0])
        return BinaryStack(foldername)


class GenericStack():
    def __init__(self,filenames,extension):
        self.filenames = filenames
        self.fileextension = extension
        self.dims = None
        self.dtype = None
        self.frames_offset = []
        self.files = []
        self.current_fileidx = None
        self.current_stack = None

    def _get_frame_index(self,frame):
        '''
        Finds out in which file some frames are.
        '''
        fileidx = np.where(self.frames_offset <= frame)[0][-1]
        return fileidx,frame - self.frames_offset[fileidx]
    
    def _load_substack(fileidx):
        pass
    
    def _get_frame(self,frame):
        ''' 
        Returns a single frame from the stack.
        '''
        fileidx,frameidx = self._get_frame_index(frame)
        if not fileidx == self.current_fileidx:
            self._load_substack(fileidx)
        return self.current_stack[frameidx]

    def __len__(self):
        return self.shape[0]
    
    def __getitem__(self, *args, squeeze = False):
        index  = args[0]
        idx1 = None
        idx2 = None
        if type(index) is tuple: # then look for 2 channels
            if type(index[1]) is slice:
                idx2 = range(index[1].start, index[1].stop, index[1].step)
            else:
                idx2 = index[1]
            index = index[0]
        if type(index) is slice:
            idx1 = range(*index.indices(self.nframes))#start, index.stop, index.step)
        elif type(index) in [int,np.int32, np.int64]: # just a frame
            idx1 = [index]
        else: # np.array?
            idx1 = index
        img = np.empty((len(idx1),*self.dims),dtype = self.dtype)
        for i,ind in enumerate(idx1):
            img[i] = self._get_frame(ind)
        if not idx2 is None:
            if squeeze:
                return img[:,idx2].squeeze()
            else:
                return img[:,idx2]
        if squeeze:
            return img.squeeze()
        else:
            return img

    def export_binary(self, foldername,
                      basename = 'frames',
                      chunksize = 512,
                      start_frame = 0,
                      end_frame = None,
                      channel = None):
        '''
        Exports a binary file.
        '''
        nframes,nchan,h,w = self.shape
        if end_frame is None :
            end_frame = nframes
        nframes = end_frame - start_frame
        chunks = chunk_indices(nframes,chunksize)    
        chunks = [[c[0]+start_frame,c[1]+start_frame] for c in chunks]
        shape = [nframes,*self.shape[1:]]
        if not channel is None:
            shape[1] = 1
        fname = pjoin('{0}'.format(foldername),'{4}_{0}_{1}_{2}_{3}.bin'.format(
            *shape[1:],self.dtype,basename))
        if not os.path.isdir(os.path.dirname(fname)):
            os.makedirs(os.path.dirname(fname))
        out = np.memmap(fname,
                        dtype = self.dtype,
                        mode = 'w+',
                        shape=tuple(shape))
        for c in tqdm(chunks, desc='Exporting binary'):
            if channel is None:
                out[c[0] - start_frame:c[1] - start_frame] = self[c[0]:c[1]]
            else:
                out[c[0] - start_frame:c[1] - start_frame,0] = self[c[0]:c[1],channel]
        out.flush()
        del out
        return fname

    def export_tiffs(self, foldername,
                     basename = 'frames',
                     chunksize = 512,
                     start_frame = 0,
                     end_frame = None,
                     channel = None):
        '''
        Exports tifffiles.
        '''
        nframes,nchan,h,w = self.shape
        if end_frame is None :
            end_frame = nframes
        nframes = end_frame - start_frame
        chunks = chunk_indices(nframes,chunksize)    
        chunks = [[c[0]+start_frame,c[1]+start_frame] for c in chunks]
        shape = [nframes,*self.shape[1:]]
        if not channel is None:
            shape[1] = 1

        file_no = 0
        fname = pjoin('{0}'.format(foldername),'{0}_{1:05d}.tif')
        if not os.path.isdir(os.path.dirname(fname)):
            os.makedirs(os.path.dirname(fname))
        
        for c in tqdm(chunks, desc='Exporting tiffs'):
            if channel is None:
                tf.imsave(fname.format(basename,file_no),self[c[0]:c[1]].reshape((-1,*self.dims[1:])))
            else:
                tf.imsave(fname.format(basename,file_no),self[c[0]:c[1],channel].squeeze())
            file_no += 1


class TiffStack(GenericStack):
    def __init__(self,filenames,
                 extension = '.tiff', # this will try this extension first and then .tif, .TIFF and .TIF
                 nchannels = 2): 
        '''
        Select a stack from a sequence of TIFF stack files

        '''
        self.extension = extension
        if type(filenames) is str:
            # check if it is a folder
            if os.path.isdir(filenames):
                dirname = filenames
                filenames = []
                for extension in [self.extension,'.tif','.TIFF','.TIF']:
                    if not len(filenames): # try other
                        self.extension = extension
                        filenames = natsorted(glob(pjoin(dirname,'*'+self.extension)))
        if not len(filenames):
            raise(OSError('Could not find files.'))
        super(TiffStack,self).__init__(filenames,extension)

        self.imread = tf.imread
        self.TiffFile = tf.TiffFile
        offsets = [0]
        for f in tqdm(self.filenames, desc='Parsing tiffs'):
            # Parse all files in the stack
            tmp =  tf.TiffFile(f)
            # get the size from the pages (works with OEM files)
            dims = [len(tmp.pages),*tmp.pages[0].shape]
            #dims = [*tmp.series[0].shape]
            if len(dims) == 2: # then these are single page tiffs
                dims = [1,*dims]
            dtype = tmp.pages[0].dtype
            offsets.append(dims[0])
            del tmp
        # offset for each file
        self.frames_offset = np.cumsum(offsets)
        if nchannels is None:
            nchannels = 1
        self.frames_offset = (self.frames_offset/nchannels).astype(int)
        self.dims = dims[1:]
        if len(self.dims) == 2:
            self.dims = [nchannels,*self.dims]
        self.dims[0] = nchannels
        self.dtype = dtype
        self.nframes = self.frames_offset[-1]
        self.shape = tuple([self.nframes,*self.dims])
        
    def _imread(self, filename):
        arr = None
        with self.TiffFile(filename) as tf:
            arr = np.stack([p.asarray() for p in tf.pages])
        return arr
    
    def _load_substack(self,fileidx,channel = None):
        self.current_stack = self._imread(self.filenames[fileidx]).reshape([-1,*self.dims])
        self.current_fileidx = fileidx
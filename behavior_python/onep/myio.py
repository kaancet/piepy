import os
import math
import numpy as np
import pandas as pd
from tqdm import tqdm


def load_dat(filename,
             nframes = None,
             offset = 0,
             shape = None,
             dtype='uint16'): 
    '''
    Loads frames from a binary file.
    
    Inputs:
        filename (str)       : fileformat convention, file ends in _NCHANNELS_H_W_DTYPE.dat
        nframes (int)        : number of frames to read (default is None: the entire file)
        offset (int)         : offset frame number (default 0)
        shape (list|tuple)   : dimensions (NCHANNELS, HEIGHT, WIDTH) default is None
        dtype (str)          : datatype (default uint16) 
    Returns:
        An array with size (NFRAMES,NCHANNELS, HEIGHT, WIDTH).

    Example:
        dat = load_dat(filename)
        
    ''' 
    if not os.path.isfile(filename):
        raise OSError('File {0} not found.'.format(filename))
    if shape is None or dtype is None: # try to get it from the filename
        dtype,shape,_ = _parse_binary_fname(filename,
                                            shape = shape,
                                            dtype = dtype)
    if type(dtype) is str:
        dt = np.dtype(dtype)
    else:
        dt = dtype

    if nframes is None:
        # Get the number of samples from the file size
        nframes = int(os.path.getsize(filename)/(np.prod(shape)*dt.itemsize))
    framesize = int(np.prod(shape))

    offset = int(offset)
    with open(filename,'rb') as fd:
        fd.seek(offset*framesize*int(dt.itemsize))
        buf = np.fromfile(fd,dtype = dt, count=framesize*nframes)
    
    buf = buf.reshape((-1,*shape), order='C')
    return buf

def _parse_binary_fname(fname,lastidx=None, dtype = 'uint16', shape = None, sep = '_'):
    '''
    Gets the data type and the shape from the filename 
    This is a helper function to use in load_dat.
    
    out = _parse_binary_fname(fname)
    
    With out default to: 
        out = dict(dtype=dtype, shape = shape, fnum = None)
    '''
    fn = os.path.splitext(os.path.basename(fname))[0]
    fnsplit = fn.split(sep)
    fnum = None
    if lastidx is None:
        # find the datatype first (that is the first dtype string from last)
        lastidx = -1
        idx = np.where([not f.isnumeric() for f in fnsplit])[0]
        for i in idx[::-1]:
            try:
                dtype = np.dtype(fnsplit[i])
                lastidx = i
            except TypeError:
                pass
    if dtype is None:
        dtype = np.dtype(fnsplit[lastidx])
    # further split in those before and after lastidx
    before = [f for f in fnsplit[:lastidx] if f.isdigit()]
    after = [f for f in fnsplit[lastidx:] if f.isdigit()]
    if shape is None:
        # then the shape are the last 3
        shape = [int(t) for t in before[-3:]]
    if len(after)>0:
        fnum = [int(t) for t in after]
    return dtype,shape,fnum

def mmap_dat(filename,
             mode = 'r',
             nframes = None,
             shape = None,
             dtype='uint16'):
    '''
    Loads frames from a binary file as a memory map.
    This is useful when the data does not fit to memory.
    
    Inputs:
        filename (str)       : fileformat convention, file ends in _NCHANNELS_H_W_DTYPE.dat
        mode (str)           : memory map access mode (default 'r')
                'r'   | Open existing file for reading only.
                'r+'  | Open existing file for reading and writing.                 
        nframes (int)        : number of frames to read (default is None: the entire file)
        offset (int)         : offset frame number (default 0)
        shape (list|tuple)   : dimensions (NCHANNELS, HEIGHT, WIDTH) default is None
        dtype (str)          : datatype (default uint16) 
    Returns:
        A memory mapped  array with size (NFRAMES,NCHANNELS, HEIGHT, WIDTH).

    Example:
        dat = mmap_dat(filename)
    '''
    
    if not os.path.isfile(filename):
        raise OSError('File {0} not found.'.format(filename))
    if shape is None or dtype is None: # try to get it from the filename
        dtype,shape,_ = _parse_binary_fname(filename,
                                            shape = shape,
                                            dtype = dtype)
    if type(dtype) is str:
        dt = np.dtype(dtype)
    else:
        dt = dtype
    if nframes is None:
        # Get the number of samples from the file size
        nframes = int(os.path.getsize(filename)/(np.prod(shape)*dt.itemsize))
    dt = np.dtype(dtype)
    return np.memmap(filename,
                     mode=mode,
                     dtype=dt,
                     shape = (int(nframes),*shape))
    
def chunk_indices(nframes, chunksize = 512, min_chunk_size = 16):
    '''
    Gets chunk indices for iterating over an array in evenly sized chunks
    '''
    chunks = np.arange(0,nframes,chunksize,dtype = int)
    if (nframes - chunks[-1]) < min_chunk_size:
        chunks[-1] = nframes
    if not chunks[-1] == nframes:
        chunks = np.hstack([chunks,nframes])
    return [[chunks[i],chunks[i+1]] for i in range(len(chunks)-1)]
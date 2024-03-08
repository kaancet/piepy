import numpy as np
from scipy.interpolate import interp1d
import pandas as pd
import polars as pl
import os
import json
import sys
import time
from ast import literal_eval

from tqdm import tqdm
from datetime import datetime as dt
try:
    from cStringIO import StringIO
except:
    try:
        from StringIO import StringIO
    except ImportError:
        from io import StringIO

def display(msg):
    sys.stdout.write('['+dt.today().strftime('%y-%m-%d %H:%M:%S')+'] - ' + msg + '\n')
    sys.stdout.flush()
    
def unique_except(x,exceptions:list):
    """ Returns the unique values in an array except the given list"""
    uniq = np.unique(x)
    ret = [i for i in uniq if i not in exceptions]
    return np.asarray(ret)

def nonan_unique(x,sort:bool=False) -> np.ndarray:
    """ Returns the unqie list without nan values """
    u = pd.unique(x[~np.isnan(x)])
    if sort:
        return np.sort(u)
    else:
        return u

def get_fraction(data_in:np.ndarray,fraction_of,window_size:int=10,min_period:int=None) -> np.ndarray:
    """ Returns the fraction of values in data_in """
    if min_period is None:
        min_period = window_size
    
    fraction = []
    for i in range(len(data_in)):
        window_start = int(i-window_size/2)
        if window_start < 0:
            window_start = 0
        window_end = int(i+window_size/2)
        window = data_in[window_start:window_end]
   
        if len(window) < min_period:
            to_append = np.nan
        else:
            tmp = []
            for i in window:
                tmp.append(1 if i==fraction_of else 0)
                to_append = float(np.mean(tmp))
        fraction.append(to_append)
                
    return np.array(fraction)
    
def timeit(msg):
    def decorator(func):
        def wrapper(*args,**kwargs):
            ts = time.time()
            result = func(*args,**kwargs)
            te = time.time()
            display(f'{msg} : {te-ts:.3}s')
            return result
        return wrapper
    return decorator
    
def JSONConverter(obj):
    if isinstance(obj,dt):
        return obj.__str__()
    if isinstance(obj,np.ndarray):
        return obj.tolist()

def jsonify(data):
    """ Jsonifies the numpy arrays inside the analysis dictionary, mostly for saving and pretty printing"""
    jsonified = {}
    
    for key,value in data.items():
        if isinstance(value,list):
            value = [jsonify(item) if isinstance(item,dict) else item for item in value]
        if isinstance(value,dict):
            value = jsonify(value)
        if type(value).__module__=='numpy':
            value = value.tolist()
        jsonified[key] = value

    return jsonified

def find_nearest(array, value):
    if len(array):
        if isinstance(array,list):
            array = np.array(array)
        try:
            idx = np.nanargmin(np.abs(array - value))
        except:
            idx = 0
        return idx, array[idx]
    else:
        return None

def save_dict_json(path: str, dict_in: dict) -> None:
    """ Saves a dictionary as a .json file """
    with open(path,'w') as fp:
        jsonstr = json.dumps(dict_in, indent=4, default=JSONConverter)
        fp.write(jsonstr)

def load_json_dict(path: str) -> dict:
    """ Loads .json file as a dict 
        :param path : path of the .json file
        :type path  : path string
    """
    with open(path) as f_in:
        return json.load(f_in)

#TODO: there is definetly a better way to do this?
def parseConfig():
    # set the directory paths and animal ids
    config_dir = os.path.dirname(os.path.abspath(__file__))
    while True:
        try:
            config = parsePref(os.path.join(config_dir,'config.json'))
            break
        except:
            config_dir = os.path.dirname(config_dir)

    return config

def parsePref(preffile):
    with open(preffile, 'r') as infile:
        pref = json.load(infile)
        return pref

def parseVStimLog(fname):
    comments = []
    faulty = True
    with open(fname,'r') as fd:
        for line in fd:
            if line.startswith('#'):
                comments.append(line.strip('\n').strip('\r'))
                if '# CODES: stateMachine=20' in line:
                    faulty = False

    # # if state machine init not present
    # if faulty:
    #     display('LOGGING INITIALIZATION FAULTY, FIXING COMMENT HEADERS')
    #     comments += ['# Started state machine v1.2 - timing sync to rig',
    #     '# CODES: stateMachine=20',
    #     '# STATE HEADER: code,elapsed,cycle,newState,oldState,stateElapsed,trialType',
    #     '# CODES: vstim=10',
    #     '# VLOG HEADER:code,presentTime,iStim,iTrial,iFrame,blank,contrast,posx,posy,indicatorFlag']

    codes = {}
    vlogheader = []
    righeader = []
    for c in comments:
        if c.startswith('# CODES:'):
            cod = c.strip('# CODES:').strip(' ').split(',')
            for cd in cod:
                k,v = cd.split('=')
                codes[int(v)] = k
        elif c.startswith('# VLOG HEADER:'):
            cod = c.strip('# VLOG HEADER:').strip(' ').split(',')
            vlogheader = [c.replace(' ','') for c in cod]
        elif c.startswith('# RIG CSV:'):
            cod = c.strip('# RIG CSV:').strip(' ').split(',')
            righeader = [c.replace(' ','') for c in cod]

    logdata = pd.read_csv(fname,
                          names = [i for i in range(len(vlogheader))],
                          delimiter=',',
                          header=None,comment='#',engine='c')
    
    data = dict()
    for v in codes.keys():
        k = codes[v]
        data[k] = logdata[logdata[0]==v]
        if len(data[k]):
            
            #get the columns from most filled row
            tmp_nona = data[k].dropna()
            if len(tmp_nona):
                tmp = tmp_nona.iloc[0].copy()
            else:
                tmp = data[k].iloc[0].copy()
            ii = np.where([type(t) is str for t in tmp])
            for i in ii:
                tmp[i] = 0
                
            idx = np.where([~np.isnan(d) for d in tmp])[0]
            data[k] = data[k][idx]
            if len(idx) <= len(righeader):
                cols = righeader
            else:
                cols = vlogheader[:len(idx)]
            data[k] = pd.DataFrame(data = data[k])
            data[k].columns = cols
            
    if 'vstim' in data.keys() and 'screen' in data.keys():
        # extrapolate duinotime from screen indicator
        indkey = 'not found'
        fliploc = []
        if 'indicatorFlag' in data['vstim'].keys():
            indkey = 'indicatorFlag'
            fliploc = np.where(np.diff(np.hstack([0,
                                                  data['vstim']['indicatorFlag'],
                                                  0]))!=0)[0]
        elif 'blank' in data['vstim'].keys():
            indkey = 'blank'
            fliploc = np.where(np.diff(np.hstack([0,
                                                  data['vstim']['blank']==0,
                                                  0]))!=0)[0]
        if len(data['screen'])==len(fliploc):
            data['vstim']['duinotime'] = interp1d(
                fliploc,
                data['screen']['duinotime'],
                fill_value="extrapolate")(
                    np.arange(len(data['vstim'])))
        else:
            
            print(
                'The number of screen pulses {0} does not match the visual stimulation {1}:{2} log.'
                  .format(len(data['screen']),indkey,len(fliploc)))
    return data,comments

def parseCamLog(fname):
    """
    Parses the camlog
    """
    comments = []
    with open(fname,'r') as fd:
        for i,line in enumerate(fd):
            if line.startswith('#'):
                comments.append(line.strip('\n').strip('\r'))
    
    commit = None
    for c in comments:
        if c.startswith('# Log header:'):
            cod = c.strip('# Log header:').strip(' ').split(',')
            camlogheader = [c.replace(' ','') for c in cod]
        elif c.startswith('# Commit hash:'):
            commit = c.strip('# Commit hash:').strip(' ')
            
    camdata = pl.read_csv(fname,has_header=False,comment_prefix='#',new_columns=camlogheader)
    return camdata,comments,commit

def parseStimpyLog(fname):
    """ Parses the log file (riglog or stimlog) and returns data and comments

        :return: data and comments 
        :rtype: DataFrame and list
    """
    comments = []
    faulty = False
    with open(fname,'r') as fd:
        for i,line in enumerate(fd):
            if line.startswith('#'):
                comments.append(line.strip('\n').strip('\r'))
                if '# CODES: stateMachine=20' in line:
                    faulty = False
                    
    # if state machine initialization not present directly add the state machine comment lines to comments list
    if faulty:
        display('LOGGING INITIALIZATION FAULTY, FIXING COMMENT HEADERS')
        toAdd = ['# Started state machine v1.2 - timing sync to rig',
        '# CODES: stateMachine=20','# STATE HEADER: code,elapsed,cycle,newState,oldState,stateElapsed,trialType',
        '# CODES: vstim=10',
        '# VLOG HEADER:code,presentTime,iStim,iTrial,iFrame,blank,contrast,posx,posy,indicatorFlag']
        comments += toAdd
    
    codes = {}
    for c in comments:
        if c.startswith('# CODES:'):
            code_list = c.strip('# CODES').strip(': ').split(',')
            for code_str in code_list:
                code_name,code_nr = code_str.split('=')
                codes[int(code_nr)] = code_name.lower()
        elif c.startswith('# VLOG HEADER:'):
            header_list = c.strip('# VLOG HEADER:').strip(' ').split(',')
            vlogheader = [header_str.replace(' ','') for header_str in header_list]
                # ExperimentController now logs with LOGHEADER, because why not
        elif c.startswith('# LOG HEADER:'):
            cod = c.strip('# LOG HEADER:').strip(' ').split(',')
            vlogheader = [c.replace(' ','') for c in cod]
        elif c.startswith('# STATE HEADER:'):
            header_list = c.strip('# STATE HEADER:').strip(' ').split(',')
            stateheader = [header_str.replace(' ','') for header_str in header_list]
        elif c.startswith('# RIG CSV:'):
            header_list = c.strip('# RIG CSV:').strip(' ').split(',')
            righeader = [header_str.replace(' ','') for header_str in header_list]
              
    if fname.endswith('.riglog'):
        display('Parsing riglog...')
        
        q = pl.scan_csv(fname,
                has_header=False,
               comment_char='#')
        col_names = {k:righeader[i] for i,k in enumerate(q.columns)}

        q.with_columns(pl.col('code'))
        q = q.rename(col_names)
        
        logdata = q.select([
                            pl.col('code').apply(remove_brackets),
                            pl.col('timereceived'),
                            pl.col('duinotime'),
                            pl.col('value').apply(remove_brackets)
                            ]).collect()
        
    elif fname.endswith('.stimlog'):
        display('Parsing stimlog...')
    
        q = pl.scan_csv(fname,
                    has_header=False)
        q = q.with_columns(pl.col('*').cast(pl.Float32,strict=False))

        col_names = {k:vlogheader[i] for i,k in enumerate(q.columns)}
        logdata = q.rename(col_names).collect()
        
    data = {}
    not_found = []
    for code_nr in tqdm(codes.keys(),desc='Reading logs '):
        # TODO: cam1=6,cam2=7,cam3=8 changing the code_keys for later, semi hardcoded and depends on rig!!
        if code_nr == 6:
            code_key = 'facecam'
        elif code_nr == 7:
            code_key = 'eyecam'
        elif code_nr == 8:
            code_key = 'onepcam'
        else:
            code_key = codes[code_nr]
        data[code_key] = logdata.filter(pl.col('code') == code_nr)
        if len(data[code_key]):
            """
            TODO: This is semi hard coded, 
            maybe find a better way to automate this 
            so it is easier to add different loggers and their dedicated headers
            """
            if code_nr==20:
                state_data = data[code_key][:,0:len(stateheader)]
                col_names = {data[code_key].columns[i]:k for i,k in enumerate(stateheader)}
                data[code_key] = state_data.rename(col_names)
        else:
            not_found.append(code_key)
    if len(not_found):
        display(f'No data found for log key(s) : {not_found}')        
    return data,comments

def parseStimpyGithubLog(fname) -> dict:
    """ """
    headers = []
    markers = []
    sources = []        
    comments = []
    with open(fname, 'r') as file:
        lines = file.readlines()
    
    for l in lines:
        line = l.strip('\n').strip('\r')
        if l.startswith('####'):
            headers.append(line)
        elif line.startswith('###'):
            markers.append(line)
        elif line.startswith('##'):
            sources.append(line)
        elif line.startswith('#'):
            comments.append(line)
      
    # make a source parser dict
    source_key_cols = {}
    source_key_name = {}
    for s in sources:
        """e.g.:'## 2:StateMachine ['state', 'prev_state']'"""
        temp, args = s.split(' ', 2)[-2:]
        code_str, name = temp.split(':', 1)
        code = int(code_str)
        cols = literal_eval(args)
        
        source_key_cols[code] = ['code', 'time'] + cols
        
        #make names similar to old stimpy
        if name == 'StateMachine': name = 'stateMachine'
        if name == 'FunctionBased': name = 'vstim'
        if name == 'PhotoIndicator': name = 'photo'
        
        source_key_name[code] = name
    
    """
    # Below part for columns types is hard-coded, 
    # this can easily be pulled from a log.config file in the future,
    # which is planned for future implementation for flexible logging
    """
    # first two is always int(code) and float(time)                                
    source_key_type = {0:[pl.Int64, pl.Float64, pl.Float64, pl.Float64, pl.Float64, pl.Float64, pl.List, pl.List, pl.Int64, pl.Boolean, pl.Utf8, pl.Float64, pl.Float64, pl.Boolean, pl.Utf8], # 0:FunctionBased ['duration', 'contrast', 'ori', 'phase', 'pos', 'size', 'flick', 'interpolate', 'mask', 'sf', 'tf', 'opto', 'pattern']
                       1:[pl.Int64, pl.Float64, pl.Boolean, pl.Float64, pl.List, pl.Utf8, pl.Int64, pl.Int64, pl.Boolean],  # 1:PhotoIndicator ['state', 'size', 'pos', 'units', 'mode', 'frames', 'enabled']
                       2:[pl.Int64, pl.Float64, pl.Int64, pl.Int64], ## 2:StateMachine ['state', 'prev_state']
                       3:[pl.Int64, pl.Float64, pl.Int64, pl.Int64, pl.Int64, pl.Int64]} ## 3:LogDict ['block_nr', 'trial_nr', 'condition_nr', 'trial_type']
    
    for k in source_key_cols.keys():
        col_count = len(source_key_cols[k])
        type_count = len(source_key_type[k])
        assert col_count == type_count, f"The number of column names({col_count}) =/= column types({type_count})"
        
        
    logdata = pl.read_csv(fname,
                          comment_prefix='#',
                          separator=',',
                          has_header=False)
    
    data = {}
    for k,v in source_key_name.items():
        col_names = source_key_cols[k]
        col_types = source_key_type[k]
        code_filt = logdata.filter(pl.col('column_1')==k)
        
        # reattach the columns where the list was seperated
        _list_starts = []
        _list_ends = []
        for c in code_filt.columns:
            _val = code_filt[0,c]
            if isinstance(_val,str):
                if '[' in _val:
                    _list_starts.append(c)
                elif ']' in _val:
                    _list_ends.append(c)
        
        assert len(_list_starts) == len(_list_ends), f"PROBLEMATIC LOGGING OF LISTS !!"
        for i,c_l in enumerate(_list_starts):
            code_filt = code_filt.with_columns((pl.col(c_l)+','+pl.col(_list_ends[i])).alias(c_l))
            code_filt = code_filt.with_columns((pl.col(c_l).str.json_decode()))
            code_filt = code_filt.drop(_list_ends[i])
        
        # rename the columns
        code_filt = code_filt.rename({code_filt.columns[i]:c for i,c in enumerate(col_names)})
        
        # drops the columns that are all null
        keeping = []
        for i,col in enumerate(code_filt):
            if not col.null_count() == code_filt.height:
                keeping.append(i)
        col_names = [col_names[i] for i in keeping]
        col_types = [col_types[i] for i in keeping]
        code_filt = code_filt.select(col_names)
        
        # try to change dtypes
 
        for i,c in enumerate(code_filt.columns):
            t = col_types[i]
            if t!=pl.List:
                try:
                    code_filt = code_filt.with_columns(pl.col(c).cast(t))
                except pl.InvalidOperationError:
                    # for converting string boolean to boolean
                    code_filt = code_filt.with_columns(pl.col(c).str.to_lowercase().map_dict({"true":True, "false":False}))
        # drop the code column from all of the data
        code_filt =  code_filt.drop('code')
        data[v] = code_filt

    return data,comments
        
def stitchLogs(data_list:list,isStimlog:bool) -> dict:
    """ Stitches the seperate log files """

    final_data = data_list[0]
    for i in range(len(data_list)-1):
        to_append = data_list[i+1] #skipping the first
        for k,v in final_data.items():
            if isStimlog:
                #stimlog
                if k == 'vstim':
                    to_append[k] = to_append[k].with_columns([(pl.col('presentTime')+v[-1,'presentTime']),
                                                              (pl.col('iTrial')+v[-1,'iTrial']+1)])
                elif k == 'statemachine':
                    to_append[k] = to_append[k].with_columns([(pl.col('elapsed')+v[-1,'elapsed']),
                                                              (pl.col('cycle')+v[-1,'cycle'])])
            else:
                #riglog
                if not to_append[k].is_empty():
                    if k not in ['reward','position','screen']:
                        # adjust the values 
                        to_append[k] = to_append[k].with_columns(pl.col('value')+v[-1,'value']+1)
                    elif k == 'screen':
                        to_append[k] = to_append[k].slice(1)
                        to_append[k] = to_append[k].with_columns(pl.col('value')+v[-1,'value'])
                        
                    to_append[k] = to_append[k].with_columns([(pl.col('timereceived')+v[-1,'timereceived']),
                                                            (pl.col('duinotime')+v[-1,'duinotime'])])
            final_data[k] = pl.concat([v, to_append[k]])
    return final_data        
    
def extrapolate_time(data):
    """ Extrapolates duinotime from screen indicator

        :param data: 
        :type data: dict
    """
    if 'vstim' in data.keys() and 'screen' in data.keys():
        
        indkey = 'not found'
        fliploc = []
        if 'indicatorFlag' in data['vstim'].columns:
            indkey = 'indicatorFlag'
            fliploc = np.where(np.diff(np.hstack([0,
                                                  data['vstim']['indicatorFlag'],
                                                  0]))!=0)[0]
        elif 'photo' in data['vstim'].columns:
            indkey = 'photo'
            vstim_data = data['vstim'].to_pandas()
            fliploc = np.where(np.diff(np.hstack([0,
                                                  vstim_data['photo']==0,
                                                  0]))!=0)[0]
        
        if len(data['screen'])==len(fliploc):
            temp = interp1d(fliploc,data['screen']['duinotime'],fill_value="extrapolate")(np.arange(len(data['vstim']))).tolist()
            temp_df = pl.Series('duinotime',temp)
            data['vstim'] = data['vstim'].hstack([temp_df])
        else:
            
            print(
                'The number of screen pulses {0} does not match the visual stimulation {1}:{2} log.'
                  .format(len(data['screen']),indkey,len(fliploc)))
    return data

def remove_brackets(x,convert_flag=True):
    """ Removes brackets form a read csv file, best applied in a dataframe with df.applymap(remove_brackets)

        :param x: value to have the brackets removed(if exists)
        :type x: str, int, float
        :param convert_flag: Flag to determine whether to convert to float or not(this should be True 99% of time)
        :type convert_flag: bool
        :return: returns the bracket removed value
        :rtype: float
    """
    if isinstance(x,str):
        if '[' in x:
            x = x.strip('[')
        elif ']' in x:
            x = x.strip(']')
        if convert_flag:
            try:
                return float(x)
            except:
                return '' 
        else:
            return x
    else:
        return x

def parseProtocolFile(protfile):
    options = {}
    with open(protfile,'r') as fid:
        string = fid.read().split('\n')
        for i,s in enumerate(string):
            tmp = s.split('=')
            tmp = [t.strip(' ') for t in tmp]
            # Because the first lines are always like this...
            if len(tmp)>1:
                if "#" in tmp[1]:
                    tmp[1] = tmp[1].split('#')[0]
                options[tmp[0]] = tmp[1].replace('\r','')
                try:
                    prot[tmp[0]] = int(prot[tmp[0]])
                except:
                    try:
                        prot[tmp[0]] = float(prot[tmp[0]])
                    except:
                        pass
            else:
                break
        tmp = string[i::]
        tmp = [t.replace('\r','').replace('\t',' ').strip().split() for t in tmp]
        tmp = [';'.join(t) for t in tmp ] # the delimiter is ";" becuase "," causes issue with evolveParams comma
        try:
            params = pd.read_csv(StringIO(u"\n".join(tmp)),
                                    index_col=False,
                                    delimiter=';')
        except pd.io.common.EmptyDataError:
            params = None
    return options,params

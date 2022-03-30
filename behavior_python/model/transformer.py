import numpy as np
import pandas as pd
from scipy.stats import bernoulli
from ..utils import display


class DataTransformer(object):
    """ Transforms the pandas DataFrame data to a series of numpy ndarrays"""
    __slots__ = ['_input_data','output_data']
    def __init__(self) -> None:
        self._input_data = None
        self.output_data = {}

    @property
    def input_data(self):
        return self._input_data
    
    @input_data.setter
    def input_data(self,data:pd.DataFrame):
        if not isinstance(data,pd.DataFrame):
            raise TypeError(f'Data needs to be a DataFrame, got {type(data)} instead')
        self._input_data = data
        
    def transform_columns(self,col_names:list=None) -> None:
        """ Automatically transforms all the columns without any pre processing """ 
        # check input integrity
        if col_names is None:
            columns = self._input_data.columns
        else:
            if not isinstance(col_names,(list,np.ndarray)):
                raise TypeError(f'col_names argument must be a list, got {type(col_names)} instead')
            columns = []
            for c in col_names:
                if c not in self.input_data.columns:
                    raise ValueError(f'Column name {c} is not a valid column name')
                columns.append(c)
        
        # put columns into numpy arrays
        for col in columns:
            dat_col = self._input_data[col].to_numpy()
            self.output_data[col] = dat_col
            
             
class GLMHMMTransfromer(DataTransformer):
    __slots__ = ['isTransformed']            
    def __init__(self) -> None:
        super().__init__()
        self.isTransformed = False
        
    def transform_data(self,input_data:pd.DataFrame=None,stim_property:str='contrast') -> None:
        if input_data is None:
            if self.input_data is None:
                display('No input data provided, do that before attemting transformation')
                return 0
        else:
            self.input_data = input_data
        
        # the order of this is important 
        # as functions make use of vectors set in previous function calls
        self.set_signed_stim_vector(stim_property)
        self.remap_choice_vector()
        self.set_prev_choice_vector()
        self.set_rewarded_vector()
        self.create_wsls_covariate()
        self.isTransformed = True
        
    def get_session_unnormalized_data(self):
        if self.isTransformed:
            input_matrix = np.hstack((self.output_data['signed_stim'],
                                      self.output_data['previous_choice'],
                                      self.output_data['wsls']))
            y = self.output_data['remapped_choice']
            rewarded = self.output_data['rewarded']
            
            return input_matrix, y, rewarded
        else:
            display('Transform the data first before getting session_data')
            return 0
        
    def remap_choice_vector(self) -> None:
        """raw choice vector has CW = -1 (correct response for stim on left),
        CCW = 1 (correct response for stim on right) and viol = 0.  Let's
        remap so that CW = 0, CCw = 1, and viol = -1
        """
        
        choice_mapping = {-1: 0, 0: -1, 1:1}
        new_choice_vector = [choice_mapping[old_choice] for old_choice in self.input_data['choice'].to_numpy()]
        self.output_data['choice'] = self.input_data['choice'].to_numpy().reshape(-1,1)
        self.output_data['remapped_choice'] = np.array(new_choice_vector).reshape(-1,1)
            
    def set_signed_stim_vector(self,stim_property:str='contrast') -> None:
        """ Sets signed stim property, where negative(-) means left"""
        valid_props = ['sf','tf','contrast']
        if stim_property not in valid_props:
            raise ValueError(f'{stim_property} is not a valid property, expected one of {valid_props}')
        
        stim_left = np.nan_to_num(self.input_data[stim_property + '_l'].to_numpy(), nan=0)
        stim_right = np.nan_to_num(self.input_data[stim_property + '_r'].to_numpy(), nan=0)
        
        signed_contrast = stim_right - stim_left
        self.output_data['signed_stim'] = signed_contrast.reshape(-1,1)
        
    def set_rewarded_vector(self) -> None:
        """ Sets the rewarded vector as 1(rewarded) and -1(unrewarded)"""
        rewarded = [1 if len(r) else -1 for r in self.input_data['reward'].to_numpy()]
        self.output_data['rewarded'] = np.array(rewarded).reshape(-1,1)
        
    def set_prev_choice_vector(self) -> None:
        """Sets the previous choice vector
        Here the nogos are discarded and the most recent 
        non-nogo previous choice is put instead of nogos.
        
        locs_mapping: array of size (~num_viols)x2, where the entry in
        column 1 is the location in the previous choice vector that was a
        remapping due to a violation and the
        entry in column 2 is the location in the previous choice vector that
        this location was remapped to
        """
        choice = self.output_data['choice']
        previous_choice = np.vstack([np.array(choice[0]).reshape(-1,1), choice])[:-1]
        locs_to_update = np.where(previous_choice == -1)[0]
        locs_with_choice = np.where(previous_choice != -1)[0]
        loc_first_choice = locs_with_choice[0]
        locs_mapping = np.zeros((len(locs_to_update) - loc_first_choice, 2),
                            dtype='int')
        
        for i, loc in enumerate(locs_to_update):
            if loc < loc_first_choice:
                # since no previous choice, bernoulli sample: (not output of
                # bernoulli rvs is in {1, 2})
                previous_choice[loc] = bernoulli.rvs(0.5, 1) - 1
            else:
                # find nearest loc that has a previous choice value that is not
                # -1, and that is earlier than current trial
                potential_matches = locs_with_choice[
                    np.where(locs_with_choice < loc)]
                absolute_val_diffs = np.abs(loc - potential_matches)
                absolute_val_diffs_ind = absolute_val_diffs.argmin()
                nearest_loc = potential_matches[absolute_val_diffs_ind]
                locs_mapping[i - loc_first_choice, 0] = int(loc)
                locs_mapping[i - loc_first_choice, 1] = int(nearest_loc)
                previous_choice[loc] = previous_choice[nearest_loc]
        assert len(np.unique(
            previous_choice)) <= 2, "previous choice should be in {0, 1}; " + str(
            np.unique(previous_choice))
        
        remapped_prev_choice = 2 * previous_choice - 1 # remapped to {-1,1}
        self.output_data['previous_choice'] = remapped_prev_choice.reshape(-1,1)
        self.output_data['locs_mapping'] = locs_mapping.reshape(-1,2)
        
    def create_wsls_covariate(self) -> None:
        """
        output:
        wsls: vector of size T, entries are in {-1, 1}.  1 corresponds to
        previous choice = right and success OR previous choice = left and
        failure; -1 corresponds to
        previous choice = left and success OR previous choice = right and failure
        """
        previous_reward = np.vstack([np.array(self.output_data['rewarded'][0]).reshape(-1,1), self.output_data['rewarded']])[:-1]
        # Now need to go through and update previous reward to correspond to
        # same trial as previous choice:
        for i, loc in enumerate(self.output_data['locs_mapping'][:, 0]):
            nearest_loc = self.output_data['locs_mapping'][i, 1]
            previous_reward[loc] = previous_reward[nearest_loc]
        wsls = previous_reward * self.output_data['previous_choice']
        assert len(np.unique(wsls)) == 2, "wsls should be in {-1, 1}"
        self.output_data['wsls'] = wsls.reshape(-1,1)
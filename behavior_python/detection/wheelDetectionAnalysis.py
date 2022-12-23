import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, wilcoxon,fisher_exact,barnard_exact,boschloo_exact
from ..utils import nonan_unique
from collections import defaultdict


class DetectionAnalysis:
    def __init__(self,data:dict) -> None:
        self.data = data
        self.hit_rate_dict = self.get_hitrates()
        
    def get_hitrates(self) -> dict:
        """ Gets the hit rates, counts and confidence intervals for each contrast for each side """
        hit_rate_dict = {}
        
        for i,k in enumerate(self.data.keys()):
            hit_rate_dict[k] = {}
            # get contrast data
            v = self.data[k]
            
            contrast_list = nonan_unique(v['contrast'],sort=True)
            
            correct_ratios = []
            confs = []
            counts = {}
            for c in contrast_list:
                c_data = v[v['contrast']==c]
                counts[c] = len(c_data)
                ratio = len(c_data[c_data['answer']==1]) / len(c_data[c_data['answer']!=-1])
                confs.append(1.96 * np.sqrt((ratio * (1 - ratio)) / len(c_data)))
                correct_ratios.append(ratio)
            
            hit_rate_dict[k]['nonsided'] = {'contrasts': contrast_list,
                                            'counts':counts,
                                            'hit_rate':correct_ratios,
                                            'confs':confs
                                            }
            
            sides = nonan_unique(v['stim_side'],sort=True)
            
            for j,side in enumerate(sides):
                side_data = v[v['stim_side']==side]
                contrast_list = nonan_unique(side_data['contrast'],sort=True)
                if side<0:
                    contrast_list = contrast_list[::-1]
                side_correct_ratios = []
                confs = []
                counts = {}
                for c in contrast_list:
                    c_data = side_data[side_data['contrast']==c]
                    counts[c] = len(c_data)
                    ratio = len(c_data[c_data['answer']==1]) / len(c_data[c_data['answer']!=-1])
                    confs.append(1.96 * np.sqrt((ratio * (1 - ratio)) / len(c_data)))
                    side_correct_ratios.append(ratio)
                    
                hit_rate_dict[k][side] = {'contrasts':contrast_list,
                                            'counts':counts,
                                            'hit_rate':side_correct_ratios,
                                            'confs':confs
                                            }
                
        return hit_rate_dict

    def get_deltahit(self,contrast:float,side:str='contra'):
        """Return the delta between the hitrates of a given contrast """
        
        non_opto_key = [k for k in self.hit_rate_dict.keys() if 'opto' not in k][0]
        opto_key = [k for k in self.hit_rate_dict.keys() if 'opto' in k][0]
        
        if side=='catch':
            side_key = 0.0
        elif side == 'contra':
            temp_keys = [k for k in self.hit_rate_dict[non_opto_key].keys() if isinstance(k,float)]
            side_key = [k for k in temp_keys if k>0][0]
        elif side == 'ipsi':
            temp_keys = [k for k in self.hit_rate_dict[non_opto_key].keys() if isinstance(k,float)]
            side_key = [k for k in temp_keys if k<0][0]
        
        nonopto_dict = self.hit_rate_dict[non_opto_key][side_key]
        idx_c = np.where(nonopto_dict['contrasts']==contrast)[0][0]
        nonopto_hit = nonopto_dict['hit_rate'][idx_c]
        
        opto_dict = self.hit_rate_dict[opto_key][side_key]
        idx_c = np.where(opto_dict['contrasts']==contrast)[0][0]
        opto_hit = opto_dict['hit_rate'][idx_c]
        
        delta_hit = (nonopto_hit - opto_hit) / (nonopto_hit + opto_hit)
        
        return delta_hit

    def get_hitrate_pvalues_exact(self, stim_data_keys:list=None, stim_side:str='contra', method:str='barnard') -> dict:
        """ Calculates the p-values for each contrast in different stim datasets"""
        if stim_data_keys is None:
            stim_data_keys = self.data.keys()
            
        if len(stim_data_keys) != 2:
            raise ValueError(f'There should be 2 different sets of stimuli data to get the p value in hit rate differences')
        
        def def_value():
            contingency_table = np.zeros((2,2))
            contingency_table[:] = np.nan
            return contingency_table
        
        table_dict = defaultdict(def_value) # this guy gets contrast values as keys
        for i,k in enumerate(stim_data_keys):
            v = self.data[k]
            if stim_side == 'contra':
                side_data = v[v['stim_side'] > 0]
            elif stim_side == 'ipsi':
                side_data = v[v['stim_side'] < 0]
            elif stim_side == 'catch':
                side_data = v[v['stim_side'] == 0]
            else:
                raise ValueError(f'stim_side argument only takes [contra,ipsi,catch] values')
            
            contrast_list = nonan_unique(side_data['contrast'],sort=True)
            for c in contrast_list:
                data_correct = side_data[(side_data['contrast']==c) & (side_data['answer']==1)]
                data_incorrect = side_data[(side_data['contrast']==c) & (side_data['answer']==0)]

                table_dict[c][i,:] = [len(data_correct), len(data_incorrect)]
            
        p_values = {}
        for k,table in table_dict.items():
            if np.all(np.isnan(table)==False): # all elements are filled
                if method == 'barnard':
                    res = barnard_exact(table, alternative='two-sided')
                elif method == 'boschloo':
                    res = boschloo_exact(table,alternative='two-sided')
                elif method == 'fischer':
                    res = fisher_exact(table,alternative='two-sided')
                p_values[k] = res.pvalue
            else:
                p_values[k] = np.nan
        
        return p_values

    @staticmethod
    def get_df_subset(data_in:pd.DataFrame,subset_dict:dict) -> pd.DataFrame:
        """ Gets the subset of a DataFrame that satisfies the conditions in the subset_dict. Makes a copy to return """
        df = data_in.copy(deep=True)
        for k,v in subset_dict.items():
            if k in data_in.columns:
                df = df[df[k]==v]
            else:
                raise ValueError(f'There is no column named {k} in session data')
        return df

    @staticmethod
    def get_subset(data:dict,subset_dict:dict) -> dict:
        """ Gets the subset of all data that satisfies the conditions in the subset_dict. Makes a copy to return """
        ret_dict = {}
        for k,v in data.items():
            df = DetectionAnalysis.get_df_subset(v,subset_dict=subset_dict)
            ret_dict[k] = df
        return ret_dict

    @staticmethod
    def get_pvalues_nonparametric(x1,x2,method:str='mannu') -> dict:
        """ Returns the significance value of two distributions """
        if method not in ['mannu','wilcoxon']:
            raise ValueError(f'{method} not a valid statistical test yet, try mannu or wilcoxon')
        if method == 'mannu':
            _,p = mannwhitneyu(x1,x2)
        elif method == 'wilcoxon':
            res = wilcoxon(x1,x2)
            p = res.p
        return p
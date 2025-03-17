import os
import hashlib
import polars as pl
from scipy.optimize import curve_fit
from multiprocessing import Pool, set_start_method


from .wheelDetectionSession import WheelDetectionSession, get_run_stats
from ....core.config import config as cfg
from ....core.data_functions import make_subsets


def generate_unique_session_id(baredate:str,animalid:str,*args,digit_len:int=7) -> int:
    """Creates a unique session no, 
    NOTE: this assumes baredate and animalid combinations are unique 

    Args:
        baredate (str): date in string form (e.g. 240801)
        animalid (str): ID of the animal in string form (e.g. KC133)
        digit_len (int, optional): length of digits in the unique ID. Defaults to 6.

    Returns:
        int: unique id
    """
    all_strings = [baredate,animalid,*args]
    combined_string = "|".join(sorted(all_strings))
    # Create a hash from the combined string
    hash_object = hashlib.sha256(combined_string.encode('utf-8'))
    # Convert the hash to an integer
    unique_id = int(hash_object.hexdigest(), 16) % 10**digit_len
    return unique_id


class WheelDetectionExperimentHub:
    def __init__(self):
        self.summary_data = None
        
    def set_data(self, 
                 data:pl.DataFrame|list, 
                 make_summary:bool=True, 
                 load_sessions:bool=False) -> None:
        """Sets the data, 

        Args:
            data (pl.DataFrame | list): can be a previously saved data frame or a list of experiment paths
            make_summary (bool, optional): Makes a summary dataframe from the main dataframe. Defaults to True.
            load_sessions (bool, optional): Loads the sessions instead of reanalyzing them. Defaults to False.
        """
        if isinstance(data, pl.DataFrame):
            self.data = data
            # try to extract experiment list from the data
            self.exp_list = self.data["session_path"].unique(maintain_order=True).to_list()
            #  = [pjoin(cfg.paths["presentation"],e) for e in exp_names]
            
        if isinstance(data,list):
            self.exp_list = data
            self.gather_sessions(load_sessions=load_sessions)
            
        if make_summary:
            self.summary_data = self.make_summary_data()
            return self.summary_data
       
    @staticmethod
    def parse_session_name(session_path:str) -> dict:
        """Parses the exp names from the dir path of experiment
        Expects the session directory to be of the form:
        date_animalid_paradigm_opto_area__imaging_user

        Args:
            session_path (str): full path of the experiment

        Returns:
            dict: Parsed values
        """
        sessiondir = session_path.split(os.sep)[-1]
        parts_of_session = sessiondir.split("_")
        area = parts_of_session[4]
        
        ret_dict = {"session_path":session_path,
                    "sessiondir":sessiondir,
                    "date":parts_of_session[0],
                    "animalid":parts_of_session[1],
                    "user":parts_of_session[-1]
                    }
        
        for part in parts_of_session:
            if "opto" in part:
                ret_dict["opto_power"] = int(part[-3:]) / 100
            else:
                ret_dict["opto_power"] = 'na'
                
            if part in ["1P","2P"]:
                ret_dict["imaging"] = part
            else:
                ret_dict["imaging"] = "na"
                
            if part.lower() in ["detection","detect"]:
                ret_dict["paradigm"] = part.lower()
            else:
                ret_dict["paradigm"] = 'na'

        isCNO = False
        if "CNO" in area:
            area = area.strip("CNO")
            isCNO = True
        _temp = {"area":area,
                 "isCNO":isCNO}
            
        return {**ret_dict,**_temp}
        
    def _get_session(self,session_path:str) -> None:
        """Gets a single session from the sesison path
        Appends the resulting DataFrame to the multiprocess list

        Args:
            session_path (str): Path of the session
        """
        temp = self.parse_session_name(session_path)
        cfg.set_verbosity(False)
        print(temp['sessiondir'])
        sesh = WheelDetectionSession(temp.pop("sessiondir"),self.load_flag)
        _data = sesh.runs[0].data.data
        _meta = sesh.runs[0].meta
        
        stim_combination = (
                _data["stim_type"].unique().sort().drop_nulls().to_list()
            )
        
        _stats = get_run_stats(_data)
        # prepend stats dict keys to then easily group them in summary
        _stats = {f"stat_{k}":v for k,v in _stats.items()}
        
        # contrast titration boolean
        uniq_stims = _data["contrast"].drop_nulls().unique().to_numpy()
        isTitrated = False
        if len(uniq_stims) > len(_meta["opts"]["contrastVector"]):
            isTitrated = True
            
        meta_lit = {
            "opto_ratio": _meta["opts"]["optoRatio"],
            "opto_targets": len(_data["opto_pattern"].unique().to_numpy())-1,
            "stimulus_count": len(_data["stim_type"].unique().drop_nulls()),
            "stim_combination": "+".join(stim_combination),
            "isTitrated": isTitrated,  # this should only give 0 or 1
            "rig": _meta["rig"]["name"],
            "session_id": generate_unique_session_id(temp.pop("date"),
                                                     temp.pop("animalid"),
                                                     _meta["run_start_time"]),
        }
        non_lit = {
                "contrast_vector": [_meta["opts"]["contrastVector"]] * len(_data),
                "sf_values": [_data["sf"].unique().drop_nulls().to_list()] * len(_data),
                "tf_values": [_data["tf"].unique().drop_nulls().to_list()] * len(_data),
            }
        # print(session_path,_data.shape, len(_stats), len(meta_lit), len(temp), flush=True)
        lit_dict = {**_stats, **meta_lit, **temp}
        # create the polars frame
        df = _data.with_columns([pl.lit(v).alias(k) for k, v in lit_dict.items()])
        list_df = pl.DataFrame(non_lit)
        df = pl.concat([df, list_df], how="horizontal")
        # self._list_data.append(df)
        # queue.put(df)
        return df
    
    def gather_sessions(self,load_sessions:bool=False) -> pl.DataFrame:
        """_summary_

        Args:
            load_session (bool, optional): Whether to load sessions or reanalyze them. Defaults to False.

        Returns:
            pl.DataFrame: All the trials of all the experiments in hub.exp_list
        """        
        self.load_flag = load_sessions
        try:
            set_start_method("spawn")
        except RuntimeError:
            pass
        
        with Pool(processes=cfg.multiprocess["cores"]) as pool:
            _list_data = pool.map(self._get_session, self.exp_list)
            
        # concat everything on the list
        data = _list_data[0]
        for df2 in _list_data[1:]:
            df2 = df2.select(data.columns)
            if data.dtypes != df2.dtypes:
                data = data.with_columns(
                    [
                        pl.col(n).cast(t)
                        for n, t in zip(data.columns, df2.dtypes)
                        if t != pl.Null
                    ]
                )
            
            data =  pl.concat([data,df2])
            
        data = data.sort(["date","animalid","session_id"],
                         descending=[False,False,False])
        
        # make reorder column list
        reorder = ["total_trial_no"] + data.columns
        # in the end add a final all trial count column
        data = data.with_columns(
            pl.Series(name="total_trial_no", values=list(range(1, len(data) + 1)))
        )
        # reorder
        self.data = data.select(reorder)
    
    @staticmethod
    def filter_sessions(data:pl.DataFrame, **kwargs) -> pl.DataFrame:
        """Filters the instance cumulative data according to filter_dict

        Raises:
            KeyError: Invalid key in keyword arguments

        Returns:
            pl.DataFrame: Filtered DataFrame
        """
        if kwargs is None:
            return data
        filt_df = data.clone()   
        for k, v in kwargs.items():
            if k not in filt_df.columns:
                raise KeyError(
                    f"The filter key {k} is not present in the data columns, make sure you have the correct data column names"
                )

            if isinstance(filt_df[k],pl.List):
                for el in v:
                    filt_df = filt_df.filter(pl.col(k).list.contains(el))
            
            else:
                if isinstance(v,list):
                    filt_df = filt_df.filter(pl.col(k).is_in(v))
                else:
                    filt_df = filt_df.filter(pl.col(k)==v)
        
        return filt_df
        
    def make_summary_data(self,
                          data:pl.DataFrame|None = None
                          ) -> pl.DataFrame:
        """ Generate a summary table from the data

        Args:
            data (pl.DataFrame | None, optional): Data that will be used to create the summary. Defaults to None, in which case hub.data is used

        Returns:
            pl.DataFrame: Summary data
        """
        if data is None:
            data = self.data
        
        df = (data
            .group_by(
                [
                    "animalid",
                    "area",
                    "stimulus_count",
                    "stim_combination",
                    "opto_targets",
                    "isTitrated",
                    "isCNO",
                ]
            )
            .agg(
                [
                    (pl.count().alias("total_trials")),
                    (pl.col("stim_type").unique().drop_nulls()),
                    (pl.col("date").unique().count().alias("experiment_count")),
                    (pl.col("date").unique(maintain_order=True)),
                    (
                        pl.col("session_id")
                        .unique(maintain_order=True)
                        .alias("session_ids")
                    )
                ] +
                [
                    pl.col(k).unique(maintain_order=True) for k in data.columns if k.startswith("stat_")
                ]
            )
            .drop_nulls()
            .sort(
                [
                    "animalid",
                    "area",
                    "stimulus_count",
                    "stim_combination",
                    "opto_targets",
                    "isTitrated",
                    "isCNO",
                ]
            )
        )
        
        #Some rows in the summary data has multiple sessions, leading to list type entries for some columns (i.e session_ids),
        #to bee able to filter data properly later on, this function creates new "listed" columns, where the entries are put into lists of
        #length n, where n corresponds to the length of session count for that entry
        # l_cols = [
        #     c
        #     for c, t in zip(df.columns, df.dtypes)
        #     if not isinstance(t, pl.List)
        # ]
        # session_ids = df["session_ids"].to_list()
        # for c in l_cols:
        #     _vals = df[c].to_list()
        #     _listed = [[a] * len(s) for a, s in zip(_vals, session_ids)]
        #     df = df.with_columns(pl.Series(f"listed_{c}", _listed))
        
        return df
    
    def trial_count_more_than(self,
                              trial_count:float,
                              data:pl.DataFrame|None=None) -> tuple[pl.DataFrame, pl.DataFrame]:
        """ Ease of access function to filter by the stimulus trial count

        Args:
            trial_count (float): Threshold trial count
            data (pl.DataFrame | None, optional): Data to be filtered. Defaults to None. Uses the instance data if None

        Returns:
            pl.DataFrame: Filtered cumulative data and summary data
        """
        if data is None:
            data = self.data
        
        filt_df = data.filter(pl.col("stat_stim_trial_count") >= trial_count)

        return filt_df
        
    def ez_hit_rate_more_than(self,
                              hit_rate_thresh: float,
                              data:pl.DataFrame|None=None) -> pl.DataFrame:
        """ Ease of access function to filter by the easy (>=50% contrast) trial hit rate
        
        Args:
            hit_rate_thresh (float): Threshold hit rate
            data (pl.DataFrame | None, optional): Data to be filtered. Defaults to None. Uses the instance data if None

        Returns:
            pl.DataFrame: Filtered cumulative data and summary data
        """
        if data is None:
            data = self.data
        
        filt_df = data.filter(pl.col("stat_easy_hit_rate") >= hit_rate_thresh)

        return filt_df
    
    def fa_rate_less_than(self,
                          fa_rate_thresh:float,
                          data:pl.DataFrame|None=None) -> pl.DataFrame:
        """ Ease of access function to filter by the easy (>=50% contrast) trial hit rate
        
        Args:
            hit_rate_thresh (float): Threshold hit rate
            data (pl.DataFrame | None, optional): Data to be filtered. Defaults to None. Uses the instance data if None

        Returns:
            pl.DataFrame: Filtered cumulative data and summary data
        """
        if data is None:
            data = self.data
            
        filt_df = self.data.filter(pl.col("stat_session_false_alarm") <= fa_rate_thresh)

        return filt_df
        
    def filter_by_areas(
        self,
        areas: list[str]|str,
        data:pl.DataFrame|None=None,
        strict_performance:bool=True,
        **kwargs,
    ) -> pl.DataFrame:
        """ An auxiliary method that specifically filters the area and after that the kwargs

        Args:
            areas (list[str] | str): name of the area
            data (pl.DataFrame | None, optional): Data to be filtered. Defaults to None. Uses the instance data if None
            strict_performance (bool, optional): 

        Returns:
            pl.DataFrame: Filtered summary data and cumulative data
        """
        if data is None:
            data = self.data
            
        if isinstance(areas,str):
            areas = [areas]
            
        if len(kwargs):
            data = self.filter_sessions(data,**kwargs)
        filt_df = data.filter(pl.col("area").is_in(areas))
        
        # here is a step to get only the "best session" if there are multiple sessions
        # the best session is defined as hr>75 and trial_count>600
        # if multiple sessions, take the last one
        _temp = filt_df.group_by(["animalid","area"]).agg(pl.col("*").unique(maintain_order=True)).sort("animalid")
        single_session_ids = _temp.filter(pl.col("session_id").list.len()==1)["session_id"].explode().to_list()
        multi_sessions_ids = _temp.filter(pl.col("session_id").list.len()>1)["session_id"].explode().to_list()
        multi_session_df = filt_df.filter(pl.col("session_id").is_in(multi_sessions_ids))
        
        # TODO: change this to AND, if not strict do OR
        for filt_tup in make_subsets(multi_session_df,["animalid","area"]):
            _df = filt_tup[-1]
            hr_df = self.ez_hit_rate_more_than(75,_df)
            if len(hr_df)>=1:
                trial_df = self.trial_count_more_than(300,hr_df)
                if len(trial_df):
                    # successfully filtered and got some sessions, sort and pick the best
                    selected_session = trial_df.sort(["stat_stim_trial_count","stat_easy_hit_rate"])[-1,"session_id"]
                else:
                    # couldn't successfuly filter for trial count, sort and pick the best from hit rate filtered
                    selected_session = hr_df.sort(["stat_stim_trial_count","stat_easy_hit_rate"])[-1,"session_id"]
            else:
                if not strict_performance:
                    # couldn't successfuly filter for hit rate, try trial count
                    trial_df = self.trial_count_more_than(300,_df)
                    if len(trial_df):
                        # successfully filtered only with trial count, sort and pick the best
                        selected_session = trial_df.sort(["stat_stim_trial_count","stat_easy_hit_rate"])[-1,"session_id"]
                    else:
                        print("No session for hit rate >=75 and stim_trial_count>=300")
                else:
                    selected_session = None
                    print(f"{filt_tup[0]} doesn't have a session with hit rate >=75 and stim_trial_count>=300, removing...")
            
            if selected_session is not None:    
                single_session_ids.append(selected_session) 
            
        filt_df = filt_df.filter(pl.col("session_id").is_in(single_session_ids))
        return filt_df
    
    def filter_by_animals(
        self,
        animalids:list[str]|str,
        data:pl.DataFrame|None=None,
        strict_performance:bool=True,
        **kwargs,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """ An auxiliary method that specifically filters the animal and after that the kwargs

        Args:
            animalids (list[str] | str): ID of the animal
            data (pl.DataFrame | None, optional): Data to be filtered. Defaults to None. Uses the instance data if None
            strict_performance (bool, optional): 
            
        Returns:
            pl.DataFrame: Filtered summary data and cumulative data
        """
        if data is None:
            data = self.data
            
        if isinstance(animalids,str):
            animalids = [animalids]
            
        if len(kwargs):
            data = self.filter_sessions(data,**kwargs)
        filt_df = data.filter(pl.col("animalid").is_in(animalids))
            
        # here is a step to get only the "best session" if there are multiple sessions
        # the best session is defined as hr>75 and trial_count>600
        # if multiple sessions, take the first one
        _temp = filt_df.group_by(["animalid","area"]).agg(pl.col("*").unique(maintain_order=True)).sort("animalid")
        single_session_ids = _temp.filter(pl.col("session_id").list.len()==1)["session_id"].explode().to_list()
        multi_sessions_ids = _temp.filter(pl.col("session_id").list.len()>1)["session_id"].explode().to_list()
        multi_session_df = filt_df.filter(pl.col("session_id").is_in(multi_sessions_ids))
        
        for filt_tup in make_subsets(multi_session_df,["animalid"]):
            _df = filt_tup[-1]
            hr_df = self.ez_hit_rate_more_than(75,_df)
            if len(hr_df)>=1:
                trial_df = self.trial_count_more_than(300,hr_df)
                if len(trial_df):
                    # successfully filtered and got some sessions, sort and pick the best
                    selected_session = trial_df.sort(["stat_stim_trial_count","stat_easy_hit_rate"])[-1,"session_id"]
                else:
                    # couldn't successfuly filter for trial count, sort and pick the best from hit rate filtered
                    selected_session = hr_df.sort(["stat_stim_trial_count","stat_easy_hit_rate"])[-1,"session_id"]
            else:
                if not strict_performance:
                    # couldn't successfuly filter for hit rate, try trial count
                    trial_df = self.trial_count_more_than(300,_df)
                    if len(trial_df):
                        # successfully filtered only with trial count, sort and pick the best
                        selected_session = trial_df.sort(["stat_stim_trial_count","stat_easy_hit_rate"])[-1,"session_id"]
                    else:
                        print("No session for hit rate >=75 and stim_trial_count>=300")
                else:
                    print(f"{filt_tup[0]} doesn't have a session with hit rate >=75 and stim_trial_count>=300, removing...")
                
            single_session_ids.append(selected_session) 
            
        filt_df = filt_df.filter(pl.col("session_id").is_in(single_session_ids))
        return filt_df

    @staticmethod
    def transform_to_rig_time(data: pl.DataFrame) -> pl.DataFrame:
        """Transforms the reaction time of trials that dont't have rig_reaction_time to that time frame"""
        
        with_rig_time = data.drop_nulls("rig_response_time")
        resp_time = with_rig_time["state_response_time"]
        rig_time = with_rig_time["rig_response_time"]

        def m1_func(x, a):
            m = 1
            return m * x + a

        if len(rig_time):
            popt, pcov = curve_fit(
                m1_func, resp_time, rig_time
            )  # popt[0] is the time diff intercept

            all_resp_time = data["response_latency"]
            extrp_rig_times = m1_func(all_resp_time, *popt)

            tmp = pl.Series("temp_response_times", extrp_rig_times)
            data = data.with_columns(tmp)

            data = data.with_columns(
                pl.when(pl.col("outcome") == "hit")
                .then(pl.col("temp_response_times"))
                .otherwise(pl.col("response_latency"))
                .alias("reaction_time_2")
            )
            # drop the temp column
            data = data.drop("temp_response_times")

        else:
            print("NO RIG TIME TO INTERPOLATE, COPYING STATE TIME")
            data = data.with_columns(pl.col("response_latency").alias("reaction_time"))

        return data
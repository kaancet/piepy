import polars as pl
import numpy as np
from ...core.statistics import bootstrap_confidence_interval


class WheelGroupedAggregator:
    def __init__(self) -> None:
        self.data = None
        self.outcomes = None
    
    def set_outcomes(self,outcomes:list[str]) -> None:
        """Sets the outcomes that the aggregator will group data into. 
        If the data has been set before, checks if the outcomes are valid
        
        Args:
            outcomes (list[str]): A list of outcome values

        Raises:
            ValueError: If outcome has elements that are non existent in instance data
        """
        if self.data is not None:
            for o in outcomes:
                if o not in self.data["outcome"]:
                    raise ValueError(f"{o} is not a valid outcome type that is present in the outcome column of the data")
            
            if len(outcomes) < self.data["outcome"].drop_nulls().n_unique():
                print(f" >WARNING< There are unused outcome values {outcomes} vs {self.data['outcome'].unique().to_list()}")
        self.outcomes = outcomes
        
    def set_data(self, data:pl.DataFrame) -> None:
        """Sets the data to be grouped/aggregated and statsed on
        If the outcomes has been set before, checks if the data has valid values in it's outcome column

        Args:
            data (pl.DataFrame): Run/session data

        Raises:
            ValueError: If data has values that are not present in instance outcomes
        """
        if self.outcomes is not None:
            for o in self.outcomes:
                if o not in data["outcome"]:
                    raise ValueError("Provided dataframe has not all the values in corresponding to previously set outcomes!")
                
            if len(self.outcomes) < data["outcome"].drop_nulls().n_unique():
                print(f" >WARNING< There are unused outcome values {self.outcomes} vs {data['outcome'].unique().to_list()}")
        self.data = data
        
    def group_data(self,group_by:list[str],do_sort:bool=True) -> None:
        """Groups the data by given group_by column names

        Args:
            group_by (list[str]): List of column names to group the data by
            do_sort (bool, optional): Sort the data in the order of group_by. Defaults to True.

        Raises:
            ValueError: A column name in group_by does not exist in the instance data
        """
        for c_name in group_by:
            if c_name not in self.data.columns:
                raise ValueError(f"{c_name} not in data columns!!")
            
        self.group_by = group_by

        q = self.data.group_by(group_by).agg(
            [
                # (pl.col("stim_pos").first()),
                pl.count().alias("count"),
            ]+ 
            [
                (pl.col("outcome")==o).sum().alias(f"{o}_count") for o in self.outcomes
            ]+
            [
                (pl.col("response_time").alias("response_times")),
                (pl.col("reaction_time").alias("reaction_times")),
            ]+
            [
                
                (pl.col("response_time").median().alias("median_response_times")),
                (pl.col("reaction_time").median().alias("median_reaction_times")),
            ]+
            [
                
            ]+
            [
                (
                    pl.col("response_time")
                    .filter(pl.col("outcome") == o)
                    .alias(f"{o}_response_times")
                ) for o in self.outcomes
            ]+
            [
                (
                    pl.col("response_time")
                    .filter(pl.col("outcome") == o)
                    .median()
                    .alias(f"median_{o}_response_times")
                ) for o in self.outcomes
            ] +
            [
                (
                    pl.col("reaction_time")
                    .filter(pl.col("outcome") == o)
                    .alias(f"{o}_reaction_times")
                ) for o in self.outcomes
            ] +
            [
                (
                    pl.col("reaction_time")
                    .filter(pl.col("outcome") == o)
                    .median()
                    .alias(f"median_{o}_reaction_times")
                ) for o in self.outcomes
            ]+
            [
                (pl.col("stimkey").first()),
                (pl.col("stim_label").first()),
            ]
        )
        
        # calculate confidence intervals of each columns that has "time" in it
        time_cols = [c for c in q.columns if "time" in c and "median" not in c and "confs" not in c]
        for t_c in time_cols:
            _temp_ci = []
            for v in q[t_c].to_list():
                v = [i for i in v if i is not None] # drop the nulls 
                if len(v) > 1:
                    med,ci_p, ci_n = bootstrap_confidence_interval(v,statistic=np.median)
                    _temp_ci.append([ci_p,ci_n])
                else:
                    _temp_ci.append([])
                
            q = q.with_columns(pl.Series(f"median_{t_c}_confs",_temp_ci))

        if do_sort:
            q = q.sort(group_by)
        self.grouped_data = q
    
    @staticmethod
    @np.vectorize
    def confidence95(ups:float, downs:float):
        """ Wald's 95% confidence interval

        Args:
            ups (float): Number of positive results, e.g. hit, correct, etc
            downs (float): Number of negative results, e.g. miss, nogo, etc

        Returns:
            _type_: _description_
        """
        n = ups + downs
        if n == 0:
            return 0.0, 0.0, 0.0

        hr = float(ups) / n
        
        z = 1.96
        bound_lower = ((hr + (z**2)/(2*n) - z * np.sqrt((hr*(1-hr)+(z**2)/(4*n))/n))/(1+(z**2)/n))
        bound_upper = ((hr + (z**2)/(2*n) + z * np.sqrt((hr*(1-hr)+(z**2)/(4*n))/n))/(1+(z**2)/n))
        bound_upper = min(1,bound_upper)
        bound_lower = max(0,bound_lower)
        return (bound_upper - hr), hr, hr - bound_lower
    
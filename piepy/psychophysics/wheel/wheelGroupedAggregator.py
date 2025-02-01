import polars as pl


class WheelGroupedAggregator:
    def __init__(self) -> None:
        self.data = None
        self.outcomes = None
    
    def set_outcomes(self,outcomes:list[str]) -> None:
        """ Sets the outcomes that the aggregator will group data into. 
        If the data has been set before, checks if the outcomes are valid
        
        Args:
            outcomes: A list of outcome values
        """
        if self.data is not None:
            for o in outcomes:
                if o not in self.data["outcome"]:
                    raise ValueError(f"{o} is not a valid outcome type that is present in the outcome column of the data")
            
            if len(outcomes) < self.data["outcome"].drop_nulls().n_unique():
                print(f" >WARNING< There are unused outcome values {outcomes} vs {self.data['outcome'].unique().to_list()}")
        self.outcomes = outcomes
        
    def set_data(self, data:pl.DataFrame) -> None:
        """ Sets the data to be grouped/aggregated and statsed on
        If the outcomes has been set before, checks if the data has valid values in it's outcome column
        
        Args:
            data: DataFrame of the run/session
        """
        if self.outcomes is not None:
            for o in self.outcomes:
                if o not in data["outcome"]:
                    raise ValueError("Provided dataframe has not all the values in corresponding to previously set outcomes!")
                
            if len(self.outcomes) < data["outcome"].drop_nulls().n_unique():
                print(f" >WARNING< There are unused outcome values {self.outcomes} vs {data['outcome'].unique().to_list()}")
        self.data = data
        
    def group_data(self, group_by: list[str], do_sort: bool = True) -> None:
        """Groups the data by given group_by column names
        
        Args:
            outcomes: list of outcomes to aggregate under
            group_by: List of column names to group the data by
            do_sort: falg to indicate whether to sort the data in the order of group_by
        """

        for c_name in group_by:
            if c_name not in self.data.columns:
                raise ValueError(f"{c_name} not in data columns!!")
            
        self.group_by = group_by

        q = self.data.group_by(group_by).agg(
            [
                (pl.col("stim_pos").first()),
                pl.count().alias("count"),
            ]+ 
            [
                (pl.col("outcome")==o).sum().alias(f"{o}_count") for o in self.outcomes
            ]+
            [
                (pl.col("response_time").alias("response_times")),
                (pl.col("rig_response_time").alias("rig_response_times")),
                (pl.col("reaction_time").alias("reaction_times")),
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
                    .alias(f"{o}_median_response_time")
                ) for o in self.outcomes
            ] +
            [
                (
                    pl.col("rig_response_time")
                    .filter(pl.col("outcome") == o)
                    .alias(f"{o}_rig_response_times")
                ) for o in self.outcomes
            ]+
            [
                (
                    pl.col("rig_response_time")
                    .filter(pl.col("outcome") == o)
                    .median()
                    .alias(f"{o}_median_rig_response_time")
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
                    .alias(f"{o}_median_reaction_time")
                ) for o in self.outcomes
            ]+
            [
                (pl.col("wheel_t")),
                (pl.col("wheel_pos")),
                (pl.col("signed_contrast").first()),
                (pl.col("opto").first()),
                (pl.col("stimkey").first()),
                (pl.col("stim_label").first()),
            ]
        )

        if do_sort:
            q = q.sort(group_by)
        self.grouped_data = q

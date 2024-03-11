from behavior_python.plotters.bokeh_plot.bokeh_base import *
from behavior_python.plotters.plotter_utils import Color


from bokeh.models import FactorRange
import polars as pl



class TrialTypeBarGraph(Graph):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cds = None
        self.color = Color()
        
    @staticmethod
    def morph_data(data_in:pl.DataFrame) -> dict:
        q = (
            data_in.lazy()
            .groupby(["stim_label","signed_contrast"])
            .agg(
            [
                (pl.col("outcome")==1).sum().alias("correct"),
                (pl.col("outcome")==0).sum().alias("miss")
            ]
            ).sort(["stim_label","signed_contrast"])
            )
        df = q.drop_nulls().collect()
         
        
        df = df.with_columns(pl.concat_list(pl.col(['stim_label','signed_contrast'])).alias('xaxis'))
        data_dict = df.select('xaxis','correct','miss').to_dict(as_series=False)
        temp = [(x[0],x[1]) for x in data_dict['xaxis']]
        data_dict['xaxis'] = temp
        
        return data_dict
    
    def reset_cds(self) -> None:
        self.cds = None
        
    def set_cds(self,data:pl.DataFrame) -> None:
        data_dict = self.morph_data(data)
        
        if self.cds is None:
            self.cds = ColumnDataSource(data=data_dict)
        else:
            self.cds.data = data_dict
    
    def plot(self) -> None:
        
        
        f =figure(title="",x_range=FactorRange(*self.cds.data['xaxis']),width=600,height=400,
                  x_axis_label='Trial Type)', y_axis_label='Count')
        
        regions = ['correct','miss']
        
        f.vbar_stack(regions,x='xaxis',width=0.9, alpha=0.6,color=['#189e00','#d60606'],source=self.cds)
        
        hover = HoverTool(tooltips=[('Correct Count', '@correct'),
                                    ('Miss Count', '@miss')])
        f.add_tools(hover)
        
        f.xaxis.major_label_orientation = 45
        
        self.fig = f
        
        
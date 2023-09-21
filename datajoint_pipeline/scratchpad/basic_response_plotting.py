import two_p_experiments as twop
from behavior_python.detection.wheelDetectionSession import WheelDetectionSession
import os
import numpy as np

"""
Good code for starting to work out what all the parts that will be needed in the pipeline are
"""

def cut_data_windows(ca_data, ca_time, timings):
    start_indices = [np.argwhere(ca_time >= t)[0] for t in timings]
    ca_data[:, start_indices]


#os.environ['PYDEVD_WARN_SLOW_RESOLVE_TIMEOUT'] == "20"
twop.RawTwoData.populate()
#twop.Behaviour.populate(display_progress=True)

"""
session_data = twop.Behaviour * twop.RawTwoData.Plane & ['mouse_id = "KC143"', 'plane_n = 0']
print(session_data)


df = (session_data).fetch('data_df')
w = WheelDetectionSession(sessiondir=os.path.basename(session_data.fetch('behave_session_path')), load_flag=True)
behave = w.data.data


# Get the timings of all of the 2p frames
behave
"""
# Get the times of all of the trials




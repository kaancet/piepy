import datajoint as dj
import main_tables as mt
import preprocessing_tables as pt
from datetime import datetime
from behavior_python.utils import parseStimpyLog
import matplotlib.pyplot as plt
import numpy as np


# pt.CellSelection.populate('mouse_id="KC143"')

# pt.NeuropilCorrection.populate()

mt.Behaviour.populate(mt.Recording - 'mouse_id="KC146"')

# testMouse = {'mouse_id': 'KC143', 'dob': '2023-01-01', 'sex': 'M'}
# twop.Mouse.insert1(testMouse, skip_duplicates=True)

# sessionData = {
#     'mouse_id': 'KC143',
#     'session_date': '2023-04-11',a
#     'experimental_rig': 2,
#     'experimenter': 'Kaan',
#     'img_data_path': r'\\nerffs17\boninlabwip2023\data\2photon\raw',
#     'behaviour_path': r'C:\Users\dylan\Documents\Repositories\temp_data\presentation'
# }

# twop.Session.insert1(sessionData,skip_duplicates=True)

# recData = {'mouse_id': 'KC143',
#            'session_date': '2023-04-11',
#            'rec_idx': 1,
#            'depth': 200,
#            'wavelength': 920,
#            'laser_power': 30,
#            'recording_name': r'230411_KC143_AL__2P_KC_s2p_04-09-2023'
#            }
# twop.Recording.insert1(recData, skip_duplicates=True)

# twop.Behaviour.populate()
# twop.RawTwoData.populate()



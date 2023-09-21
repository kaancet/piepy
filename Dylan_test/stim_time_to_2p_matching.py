import numpy as np
import behavior_python.core
from behavior_python.core.session import *
from behavior_python.detection.wheelDetectionSession import WheelDetectionData, WheelDetectionSession

print('functioning')
#session_data = Session(r'230411_KC143_AL__2P_KC', load_flag=True)
w = WheelDetectionSession(sessiondir=r'230411_KC143_AL__2P_KC', load_flag=True)
#df = WheelDetectionData(session_data)
df = w.data.data

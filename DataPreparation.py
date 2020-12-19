import pandas as pd
from pathlib import Path

ShotDataServe = pd.read_csv(Path.cwd() / 'data' / "ShotDataServe1000.csv")
ShotDataServeReturn = pd.read_csv(Path.cwd() / 'data' / "ShotDataServeReturn1000.csv")
ShotDataRally = pd.read_csv(Path.cwd() / 'data' / "ShotDataRally1000.csv")
ShotDataServe['Type'] = 'Serve'
ShotDataServeReturn['Type'] = 'ServeReturn'
ShotDataRally['Type'] = 'Rally'
ShotData = pd.concat([ShotDataServe, ShotDataServeReturn, ShotDataRally], ignore_index=True)

# Add ending locations of each player
xPend = []; yPend = []; xOend = []; yOend = [];

for i in range(ShotData.shape[0]):
    t1 = ShotData.loc[i, 't1']
    dur = ShotData.loc[i, 'duration']

    xPend.append(ShotData.loc[i, 'px1'] * dur + ShotData.loc[i, 'px0'])
    yPend.append(ShotData.loc[i, 'py1'] * dur + ShotData.loc[i, 'py0'])
    xOend.append(ShotData.loc[i, 'ox1'] * dur + ShotData.loc[i, 'ox0'])
    yOend.append(ShotData.loc[i, 'oy1'] * dur + ShotData.loc[i, 'oy0'])

ShotData['pxEnd'] = xPend; ShotData['pyEnd'] = yPend
ShotData['oxEnd'] = xOend; ShotData['oyEnd'] = yOend


# ShotData.to_pickle(Path.cwd() / 'pickle' / 'New' / "ShotData_1000.plk")

## This was assigning OLD buckets, based purely on 1 meter squared bins

# Now bucket the court and assign values to each shot, one for bucket it started in based on player locations one for
# bucket it ended in based on player locations
# Therefore, in simulating next shots, we simulate 1000 shots where the
# Pstart/Ostart buckets = Pend/Oend buckets, AND the type is correct, ie, servereturn or rally

# xBin = np.arange(-16,17,1)
# yBin = np.arange(-10,11,1)

# pxBinStart = (pd.cut(ShotData.px0, xBin, labels=False, retbins=True, right=False))[0]
# pyBinStart = (pd.cut(ShotData.py0, yBin, labels=False, retbins=True, right=False))[0]
# ShotData['pxBinStart'] = pxBinStart
# ShotData['pyBinStart'] = pyBinStart
# ShotData['PstartBucket'] = (pyBinStart*32 + pxBinStart)
#
# oxBinStart = (pd.cut(ShotData.ox0, xBin, labels=False, retbins=True, right=False))[0]
# oyBinStart = (pd.cut(ShotData.oy0, yBin, labels=False, retbins=True, right=False))[0]
# ShotData['oxBinStart'] = oxBinStart
# ShotData['oyBinStart'] = oyBinStart
# ShotData['OstartBucket'] = (oyBinStart*32 + oxBinStart)
#
# pxBinEnd = (pd.cut(ShotData.xPend, xBin, labels=False, retbins=True, right=False))[0]
# pyBinEnd = (pd.cut(ShotData.yPend, yBin, labels=False, retbins=True, right=False))[0]
# ShotData['pxBinEnd'] = pxBinEnd
# ShotData['pyBinEnd'] = pyBinEnd
# ShotData['PendBucket'] = (pyBinEnd*32 + pxBinEnd)
#
# oxBinEnd = (pd.cut(ShotData.xOend, xBin, labels=False, retbins=True, right=False))[0]
# oyBinEnd = (pd.cut(ShotData.yOend, yBin, labels=False, retbins=True, right=False))[0]
# ShotData['oxBinEnd'] = oxBinEnd
# ShotData['oyBinEnd'] = oyBinEnd
# ShotData['OendBucket'] = (oyBinEnd*32 + oxBinEnd)

# Finally, save data
# ShotData.to_pickle(Path.cwd() / 'pickle' /"ShotData1000.plk")
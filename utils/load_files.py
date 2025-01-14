
from cmath import nan
import numpy as np
import pandas as pd


# load_xyz_file:
#   Load a .xyz resistivity inversion file output from AarhusWorkbench
#   
#   Input:
#       FilePath: Full path to file, must include .xyz
# 
#   Output: 
#       XYZ_File: Dataframe containing all columns from .xyz file

def load_xyz_file(FilePath):
    BlankCol = 'Blank'
    file_getheader = pd.read_csv(FilePath, header=0)
    nH = len(file_getheader.loc[file_getheader[file_getheader.columns[0]].str.contains("/")])
    file_colnames = pd.read_csv(FilePath, header=nH, delim_whitespace=True)
    fhead = pd.Index.tolist(file_colnames.columns)[1:]
    fhead.append(BlankCol)
    XYZ_File = pd.read_csv(FilePath, header=nH, delim_whitespace=True)
    XYZ_File.columns = fhead
    XYZ_File.drop(BlankCol, axis=1, inplace=True)
    print('load_xyz_file')
    return XYZ_File


# get_res_model:
#   Extract resistivity model from loaded .xyz resistivity inversion
#       file from load_xyz_file
#   
#   Input:
#       XYZ_Data: Dataframe output from 'load_xyz_file'
# 
#   Output: 
#       Depth: ndarray of depth values with final depth of (inf) assigned to bottom layer
#       Rho: ndarray of rho values

def get_res_model(XYZ_Data):
    Rho_Col_Idx = [cl for cl in XYZ_Data.columns if 'RHO_' in cl and 'RHO_I_STD' not in cl]
    Dep_Col_Idx = [cl for cl in XYZ_Data.columns if 'DEP_BOT' in cl and 'DEP_BOT_STD' not in cl]

    Rho = XYZ_Data[Rho_Col_Idx].to_numpy()
    Depth = np.append(XYZ_Data[Dep_Col_Idx].to_numpy(), np.inf*np.ones([np.shape(Rho)[0], 1]), 1)
    return Depth, Rho


# load_xyz_file:
#   Apply a depth threshold to a resistivity model, removing all
#       model elements below a certain threshold
#   
#   Input:
#       Rho: ndarray of resistivity floats (Ohm m)
#       Depth: ndarray of depth floats (meters)
#       maxDepth: int, float or ndarray of depths (meters)
#           If maxDepth is a single value, all Depth and Rho values below 
#               last Depth index above maxDepth are removed
#           If maxDepth is an ndarray, it must be the same size as Depth
#               (e.g. the depth of investigation is applied). In this
#               case, all Depth and Rho values below last Depth index above
#               maximum value in maxDepth are removed; at other locations,
#               the Rho values below maxDepth are assigned (nan)  
# 
#   Output: 
#       DepthCut: ndarray of depth values above maxDepth threshold
#       RhoCut: ndarray of rho values above maxDepth threshold

def depth_thresh(Depth, Rho, maxDepth):
    import numpy as np
    import pandas as pd
    if type(maxDepth)==int or type(maxDepth)==float:
        Idx = np.where(Depth[0]<=maxDepth)
        DepthCut = Depth[:, 0 : Idx[0][-1] + 1]
        RhoCut = Rho[:, 0 : Idx[0][-1] + 1]

    elif type(maxDepth) == np.array: 
        Idx = np.where(Depth[0]<=max(maxDepth))
        DepthCut = Depth[:, 0 : Idx[0][-1] + 1]
        RhoCut = Rho[:, 0 : Idx[0][-1] + 1]

        nR, nC = np.shape(DepthCut)
        for r in np.arange(nR):
            Idx = np.where(DepthCut[r]>=maxDepth[r])
            DepthCut[r, Idx] = nan
            RhoCut[r, Idx] = nan
    
    elif type(maxDepth) == pd.Series:       # Convert Dataframe series to numpy array, do same as previous statement
        maxDepth = maxDepth.to_numpy()

        Idx = np.where(Depth[0]<=max(maxDepth))
        DepthCut = Depth[:, 0 : Idx[0][-1] + 1]
        RhoCut = Rho[:, 0 : Idx[0][-1] + 1]

        nR, nC = np.shape(DepthCut)
        for r in np.arange(nR):
            Idx = np.where(DepthCut[r]>=maxDepth[r])
            DepthCut[r, Idx] = nan
            RhoCut[r, Idx] = nan
            
    else:
        print('maxDepth must be single int or float, or must be numpy.ndarray')
    return DepthCut, RhoCut

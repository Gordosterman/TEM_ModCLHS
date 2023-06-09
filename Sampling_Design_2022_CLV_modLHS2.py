
# %% ##################################################################
# Open packages

# from locale import normalize
import os
import sys

import copy
# from cmath import inf
# from importlib import reload
# from tkinter import Label
import numpy as np


# import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import scipy as sc
import scipy.stats.mstats as st
import scipy.spatial.distance as sp
from sklearn.cluster import KMeans as kmeans

import matplotlib.pyplot as plt
# from matplotlib import cm
import time as tm
# import multiprocessing as mp


import utils as ut

print(sys.path)
fpath = os.path.join(os.path.dirname(__file__), 'utils')
figpath = os.path.join(os.path.dirname(__file__), 'Figures\\')
outputpath = os.path.join(os.path.dirname(__file__), 'Outputs\\')
sys.path.append(fpath)


np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

# %% ##################################################################
# Load data

# Data path is relative to script path
DataFilePath = os.path.dirname(__file__) + '\\Data\\tTEM Inversion XYZ Files\\'
DataFileName = 'Clarklind tTEM Inversion.xyz'

# Load xyz file
XYZ_Inv = ut.load_xyz_file(DataFilePath + DataFileName)

# CLARKLIND LOCATIONS:
TrueUTM = [[277952, 4011000],
           [277824, 4011080],
           [277833, 4011227],
           [278091, 4011156],
           [278196, 4011263]]


# OUTPUT FILE NAMES


# %% ##################################################################
# Set global variables

nDL = 5                             # Number of soundings to select for sampling (direct sampling) plus the origin

SoundingMaxDepth = 35               # Maximum depth to plot/analyze soundings
SoundingRhoRng = [10, 120]          # Range of resistivity values to plot in soundings
Souding1DLog = True                 # Plot soundings in log space (True) or linear space (False)

MaxX = 800                          # Set geographic boundaries of study area
MaxY = np.inf

Sz = 20                             # Set sounding point size

IndAngle = 30                       # Induction "smoke-ring" expansion angle

WCLV_Thresh = 0.25

SoundingPlotParams_Basic = {
    'RhoRng': [10, 120],
    'Log': True,
    'MaxDepth': 35,
    'LnClr': 'k',
    'LnWidth': [3, 2],
}

XSectionPlotParams = {              # Set basic cross-section parameters
    'FigSize': [15, 3],
    'MaxDepth': [],
    'MaxElev': [],
    'SoundWidth': 5,
    'LinLog': 'Log',
    'RhoRng': [20, 100],
    'SearchRadius': 10,
    'ShowPlanView': True,
    'PlanViewLayer': 9,
    'SensAng': IndAngle,
}

HistPlotParams = {
    'YLabel': '# Counts',
    'Edges': 41,
    'Weight': 'Counts',
    'FaceColor': (0, 0, 1, 0.75),
    'EdgeColor': 'gray',
    'XLims': [],
}


def SoundingColorFun(MainColor, Metric):
    import numpy as np
    import matplotlib.colors as cl
    nS = np.shape(Metric)[0]
    LnClr = np.ndarray((nS, 3))
    if isinstance(MainColor, list) | isinstance(MainColor, tuple):
        LnClr[0] = list(MainColor)
    elif isinstance(MainColor, str):
        LnClr[0] = list(cl.to_rgb(MainColor))

    MinGray = 0.2

    ScalMetric = (Metric - np.min(Metric))/(np.max(Metric - np.min(Metric)))*(1 - MinGray)
    LnClr[1:] = [np.asarray([1, 1, 1]) * ScalMetric[i] for i in range(1, nS)]

    return LnClr


def find_soundings(UTM_All, UTM_Sounding, Range):
    import numpy as np
    import scipy.spatial.distance as sp

    nS = np.shape(UTM_All)[0]
    Dist_All = np.reshape(sp.cdist(UTM_All, [UTM_Sounding], metric='euclidean'), nS)
    Idx = np.where(Dist_All <= Range)[0]
    Idx = Idx[np.argsort(Dist_All[Idx])]
    Dist = Dist_All[Idx]
    return Dist, Idx


def find_in_quantile(Pts, Qts):
    nPts = np.shape(Pts)[1]
    nQts = np.shape(Pts)[0]
    nQPts = np.zeros((nDL, nLHS)).astype(int)
    Qtl = np.zeros((nDL, nLHS)).astype(int)
    for i in range(nQts):
        for j in range(nPts):
            nQPts[i, j] = np.sum((Pts[:, j] >= Qts[i, j]) & (Pts[:, j] < Qts[i + 1, j]))

            Idx = np.where((Pts[:, j] >= Qts[i, j]) & (Pts[:, j] < Qts[i + 1, j]))
            Qtl[Idx, j] = i

    # Qtl: Tells which quantile in Qts that the given point Pts is found in
    # nQPs: Tells how many points are within given quantile
    return Qtl, nQPts


def calc_obj_func(nQPts, LHS_Select, LHS_Params, WCLV_Select, WW):
    CC = np.corrcoef(LHS_Params.T)
    TT = np.corrcoef(LHS_Select.T)
    ObjFuncA = np.sum(np.abs(nQPts - 1))
    ObjFuncB = np.sum(np.abs(CC - TT))
    # C_pwr = 2
    # ObjFuncC = np.power(np.sum(np.power(WCLV_Select, C_pwr)), 1/C_pwr)
    ObjFuncC = np.max(WCLV_Select)

    ObjFuncSum = WW[0]*ObjFuncA + WW[1]*ObjFuncB + WW[2]*ObjFuncC
    ObjFunAll = np.asarray([ObjFuncA, ObjFuncB, ObjFuncC, ObjFuncSum])
    return ObjFunAll


def ModCLHS(SA_Params, QDist, LHS_Params_All, WCLV_All, SubsetIdx, WW):
    StartIdx = SA_Params['StartIdx']
    nIter = SA_Params['nIter']
    nRuns = SA_Params['nRuns']
    nDL = SA_Params['nDL']
    P = SA_Params['P']
    StartTemp = SA_Params['Ts']
    MinTemp = SA_Params['Tm']
    CoolRate = SA_Params['CR']

    ObjFuncA = np.zeros((nIter, nRuns))
    ObjFuncB = np.zeros((nIter, nRuns))
    ObjFuncC = np.zeros((nIter, nRuns))
    ObjFuncFull = np.zeros((nIter, nRuns))

    AcceptIdx = np.zeros((nIter, nRuns, nDL)).astype(int)

    # LHS_Params = LHS_Params_All[SubsetIdx]
    # WCLV = WCLV_All[SubsetIdx]

    for r in range(nRuns):
        t = tm.time()
        nIdx = np.shape(SubsetIdx)[0]
        IdxPerm = copy.copy(SubsetIdx)
        np.random.shuffle(IdxPerm)

        if np.any(StartIdx):
            if (np.shape(StartIdx)[0] == nRuns) & (nRuns > 1):
                IdxS = StartIdx[r]
            else:
                IdxS = StartIdx[0]
        else:
            WCLV_25q = np.quantile(WCLV_All[SubsetIdx], 0.25)
            WCLV_25q_Idx = SubsetIdx[np.where(WCLV_All[SubsetIdx] < WCLV_25q)[0]]  # Get 25th quantile WCLV samples
            IdxS = np.random.choice(WCLV_25q_Idx, nDL, replace=False).astype(int)

        MtchIdx = [np.where(IdxPerm == [i])[0][0] for i in IdxS]
        IdxR = np.delete(IdxPerm, MtchIdx).astype(int)

        # IdxS = IdxPerm[:nDL].astype(int)          # Selected Indices to start iterations

        AcceptIdx[0, r] = IdxS
        LHS_Select = LHS_Params_All[AcceptIdx[0, r]]

        nQPts = np.zeros((nDL, nLHS))
        for i in range(nLHS):
            nQPts[:, i] = np.histogram(LHS_Select[:, i], bins=QDist[:, i])[0]

        AcceptIdx_Temp = np.zeros((nIter, nDL)).astype(int)
        ObjFuncA_Temp = np.zeros(nIter)
        ObjFuncB_Temp = np.zeros(nIter)
        ObjFuncC_Temp = np.zeros(nIter)
        ObjFuncFull_Temp = np.zeros(nIter)

        RandValA = np.random.rand(nIter)
        RandValB = np.random.rand(nIter)

        CoolTemp = StartTemp

        for k in range(nIter):

            if k == 0:
                Metro = 1
                ObjDiff = 0
                LHS_Select = LHS_Params_All[IdxS]
                WCLV_Select = WCLV_All[IdxS]
                ObjFuncAll_Select = calc_obj_func(nQPts, LHS_Select,
                                                  LHS_Params_All, WCLV_Select, WW)
            else:
                Dif = np.sum(np.abs(nQPts - 1), axis=1)
                TotalMax = np.max(Dif).astype(int)
                if RandValB[k] < P:                        # Pick random sample, swap with reserve indices
                    RandSample = np.random.randint(nDL)         # Sample to be swapped
                    TempIdx = IdxS[RandSample]                  # Temporarily store sample to be swapped 
                    RandIdx = np.random.randint(nIdx - nDL)     # Index of sample to be swapped from reserve pool
                    IdxS[RandSample] = IdxR[RandIdx]            # Swap random reserve sample into Selected Pool
                    IdxR[RandIdx] = TempIdx                     # Swap de-selected sample into reserve pool

                else:                                      # Swap worst samples (samples associated w/ too many/too few quantiles)     
                    MaxIdx = np.where(Dif == TotalMax)[0]                           # Samples to be swapped
                    TempIdx = IdxS[MaxIdx]                                          # Temporarily store sample to be swapped 
                    RandIdx = np.random.randint(nIdx - nDL, size=len(MaxIdx))     # Index of sample to be swapped from reserve pool
                    IdxS[MaxIdx] = IdxR[RandIdx]                                    # Swap random reserve sample into Selected Pool
                    IdxR[RandIdx] = TempIdx                                         # Swap de-selected sample into reserve pool

                LHS_Select = LHS_Params_All[IdxS]

                # Calcuate Objective Function

                nQPts = np.zeros((nDL, nLHS))
                for i in range(nLHS):
                    nQPts[:, i] = np.histogram(LHS_Select[:, i], bins=QDist[:, i])[0]

                WCLV_Select = WCLV_All[IdxS]

                ObjFuncAll_Select = calc_obj_func(nQPts, LHS_Select,
                                                  LHS_Params_All, WCLV_Select, WW)

                # Perform Annealing Schedule
                ObjDiff = ObjFuncAll_Select[3] - ObjFuncFull_Temp[k - 1]
                Metro = np.exp(-ObjDiff/CoolTemp) + np.random.rand()*CoolTemp

                CoolTemp = np.max([CoolTemp*CoolRate, MinTemp])
            # Generate uniform random number between 0 and 1, compare to Metro

            if (RandValA[k] < Metro or ObjDiff < 0):                 # Accept change
                AcceptIdx_Temp[k] = IdxS
                ObjFuncA_Temp[k] = ObjFuncAll_Select[0]
                ObjFuncB_Temp[k] = ObjFuncAll_Select[1]
                ObjFuncC_Temp[k] = ObjFuncAll_Select[2]
                ObjFuncFull_Temp[k] = ObjFuncAll_Select[3]
            else:                                   # Reject change
                AcceptIdx_Temp[k] = AcceptIdx_Temp[k - 1]
                ObjFuncA_Temp[k] = ObjFuncA_Temp[k - 1]
                ObjFuncB_Temp[k] = ObjFuncB_Temp[k - 1]
                ObjFuncC_Temp[k] = ObjFuncC_Temp[k - 1]
                ObjFuncFull_Temp[k] = ObjFuncFull_Temp[k - 1]

        AcceptIdx[:, r] = AcceptIdx_Temp
        ObjFuncA[:, r] = ObjFuncA_Temp
        ObjFuncB[:, r] = ObjFuncB_Temp
        ObjFuncC[:, r] = ObjFuncC_Temp
        ObjFuncFull[:, r] = ObjFuncFull_Temp
        print('Run #: ' + str(r) + '; Runtime = ' + '{:.2f}'.format(tm.time() - t))

    return ObjFuncA, ObjFuncB,  ObjFuncC, AcceptIdx


# def ModCLHS(SA_Params, QDist, LHS_Params_All, WCLV_All, SubsetIdx, WW): # ORIGINAL
    # StartIdx = SA_Params['StartIdx']
    # nIter = SA_Params['nIter']
    # nRuns = SA_Params['nRuns']
    # nDL = SA_Params['nDL']
    # P = SA_Params['P']
    # StartTemp = SA_Params['Ts']
    # MinTemp = SA_Params['Tm']
    # CoolRate = SA_Params['CR']

    # ObjFuncA = np.zeros((nIter, nRuns))
    # ObjFuncB = np.zeros((nIter, nRuns))
    # ObjFuncC = np.zeros((nIter, nRuns))
    # ObjFuncFull = np.zeros((nIter, nRuns))

    # AcceptIdx = np.zeros((nIter, nRuns, nDL)).astype(int)

    # LHS_Params = LHS_Params_All[SubsetIdx]
    # WCLV = WCLV_All[SubsetIdx]

    # for r in range(nRuns):
    #     t = tm.time()
    #     nIdx = np.shape(LHS_Params)[0]
    #     IdxPerm = np.random.permutation(nIdx)

    #     if np.any(StartIdx):
    #         if (np.shape(StartIdx)[0]) == nRuns & (nRuns > 1):
    #             IdxS = StartIdx[r]
    #         else:
    #             IdxS = StartIdx[0]
    #     else:
    #         WCLV_25q = np.where(WCLV < np.quantile(WCLV, 0.25))[0]  # Get 25th quantile WCLV samples
    #         IdxS = np.random.choice(WCLV_25q, nDL).astype(int)

    #     MtchIdx = [np.where(IdxPerm == i)[0][0] for i in IdxS]
    #     IdxR = np.delete(IdxPerm, MtchIdx).astype(int)

    #     # IdxS = IdxPerm[:nDL].astype(int)          # Selected Indices to start iterations

    #     AcceptIdx[0, r] = IdxS
    #     LHS_Select = LHS_Params[AcceptIdx[0, r].astype(int)]

    #     nQPts = np.zeros((nDL, nLHS))
    #     for i in range(nLHS):
    #         nQPts[:, i] = np.histogram(LHS_Select[:, i], bins=QDist[:, i])[0]

    #     AcceptIdx_Temp = np.zeros((nIter, nDL)).astype(int)
    #     ObjFuncA_Temp = np.zeros(nIter)
    #     ObjFuncB_Temp = np.zeros(nIter)
    #     ObjFuncC_Temp = np.zeros(nIter)
    #     ObjFuncFull_Temp = np.zeros(nIter)

    #     RandValA = np.random.rand(nIter)
    #     RandValB = np.random.rand(nIter)

    #     CoolTemp = StartTemp

    #     for k in range(nIter):
    #         Dif = np.sum(np.abs(nQPts - 1), axis=1)
    #         TotalMax = np.max(Dif).astype(int)
    #         if RandValB[k] < P:                        # Pick random sample, swap with reserve indices
    #             RandSample = np.random.randint(nDL)         # Sample to be swapped
    #             TempIdx = IdxS[RandSample]                  # Temporarily store sample to be swapped 
    #             RandIdx = np.random.randint(nIdx - nDL)     # Index of sample to be swapped from reserve pool
    #             IdxS[RandSample] = IdxR[RandIdx]            # Swap random reserve sample into Selected Pool
    #             IdxR[RandIdx] = TempIdx                     # Swap de-selected sample into reserve pool

    #         else:                                      # Swap worst samples (samples associated w/ too many/too few quantiles)     
    #             MaxIdx = np.where(Dif == TotalMax)[0]                           # Samples to be swapped
    #             TempIdx = IdxS[MaxIdx]                                          # Temporarily store sample to be swapped 
    #             RandIdx = np.random.randint(nIdx - nDL, size=len(MaxIdx))     # Index of sample to be swapped from reserve pool
    #             IdxS[MaxIdx] = IdxR[RandIdx]                                    # Swap random reserve sample into Selected Pool
    #             IdxR[RandIdx] = TempIdx                                         # Swap de-selected sample into reserve pool

    #         LHS_Select = LHS_Params[IdxS]

    #         # Calcuate Objective Function

    #         nQPts = np.zeros((nDL, nLHS))
    #         for i in range(nLHS):
    #             nQPts[:, i] = np.histogram(LHS_Select[:, i], bins=QDist[:, i])[0]

    #         WCLV_Select = WCLV[IdxS]

    #         ObjFuncAll_Select = calc_obj_func(nQPts, LHS_Select,
    #                                           LHS_Params_All, WCLV_Select, WW)

    #         # Perform Annealing Schedule
    #         CoolTemp = np.max([CoolTemp*CoolRate, MinTemp])

    #         if k == 0:
    #             Metro = 1
    #         else:
    #             ObjDiff = ObjFuncAll_Select[3] - ObjFuncFull_Temp[k - 1]
    #             Metro = np.exp(-ObjDiff/CoolTemp) + np.random.rand()*CoolTemp

    #         # Generate uniform random number between 0 and 1, compare to Metro

    #         if (RandValA[k] < Metro or ObjDiff < 0):                 # Accept change
    #             AcceptIdx_Temp[k] = IdxS
    #             ObjFuncA_Temp[k] = ObjFuncAll_Select[0]
    #             ObjFuncB_Temp[k] = ObjFuncAll_Select[1]
    #             ObjFuncC_Temp[k] = ObjFuncAll_Select[2]
    #             ObjFuncFull_Temp[k] = ObjFuncAll_Select[3]
    #         else:                                   # Reject change
    #             AcceptIdx_Temp[k] = AcceptIdx_Temp[k - 1]
    #             ObjFuncA_Temp[k] = ObjFuncA_Temp[k - 1]
    #             ObjFuncB_Temp[k] = ObjFuncB_Temp[k - 1]
    #             ObjFuncC_Temp[k] = ObjFuncC_Temp[k - 1]
    #             ObjFuncFull_Temp[k] = ObjFuncFull_Temp[k - 1]

    #     AcceptIdx[:, r] = SubsetIdx[AcceptIdx_Temp]
    #     ObjFuncA[:, r] = ObjFuncA_Temp
    #     ObjFuncB[:, r] = ObjFuncB_Temp
    #     ObjFuncC[:, r] = ObjFuncC_Temp
    #     ObjFuncFull[:, r] = ObjFuncFull_Temp
    #     print('Run #: ' + str(r) + '; Runtime = ' + '{:.2f}'.format(tm.time() - t))

    # return ObjFuncA, ObjFuncB,  ObjFuncC, AcceptIdx


# %% ##################################################################
# Organize data
UTM = np.transpose([XYZ_Inv.UTMX[:] - min(XYZ_Inv.UTMX[:]),
                    XYZ_Inv.UTMY[:] - min(XYZ_Inv.UTMY[:])])

TrueUTM = np.asarray(TrueUTM) - np.asarray([min(XYZ_Inv.UTMX[:]), min(XYZ_Inv.UTMY[:])])
TrueIdx = np.asarray([find_soundings(UTM, U, np.inf)[1][0] for U in TrueUTM])               # Indices of soundings closest to sediment cores

DOI_Con = np.asarray(XYZ_Inv.DOI_CONSERVATIVE)

# Extract resistivity (rho) and depth arrays; cutoff below maxDepth
Depth, Rho = ut.get_res_model(XYZ_Inv)

Elev = np.ones(np.shape(DOI_Con)) * np.median(XYZ_Inv.ELEVATION)

DataRes = XYZ_Inv.RESDATA.to_numpy()

del XYZ_Inv

Depth = Depth[np.logical_not(np.logical_or(UTM[:, 0] > MaxX, UTM[:, 1] > MaxY))]
Rho = Rho[np.logical_not(np.logical_or(UTM[:, 0] > MaxX, UTM[:, 1] > MaxY))]
DOI_Con = DOI_Con[np.logical_not(np.logical_or(UTM[:, 0] > MaxX, UTM[:, 1] > MaxY))]
Elev = Elev[np.logical_not(np.logical_or(UTM[:, 0] > MaxX, UTM[:, 1] > MaxY))]
DataRes = DataRes[np.logical_not(np.logical_or(UTM[:, 0] > MaxX, UTM[:, 1] > MaxY))]

UTM = UTM[np.logical_not(np.logical_or(UTM[:, 0] > MaxX, UTM[:, 1] > MaxY))]

DepthCut, RhoCut = ut.depth_thresh(Depth, Rho, SoundingMaxDepth)

# Convert resistivity to log10 resistivity
LogRho = np.log10(Rho)
LogRhoCut = np.log10(RhoCut)

nS = np.shape(UTM)[0]           # Number of soundings (points)
nL = np.shape(DepthCut)[1]

# %% ##################################################################
# Set up flags for soundings to exclude
# Un-flag soundings too close to edge of survey area

EdgeRadius = 100        # Radius to search around soundings
EdgeQuantile = 0.35   # Remove soundings w/ # of other soundings in FlagRadius below this quantile

EdgeFlag = ut.edge_flag(UTM, EdgeRadius, EdgeQuantile, False)

# TitleStr = 'Exclusion_Zone_PlanView'
# ut.color_plot_layer(UTM, TitleStr, EdgeFlag, [0, 1], 'viridis', 'Flag Value (1=keep; 0=reject)', Sz, 'False')

# %% ##################################################################
# Exclude high data residual soundings
DataResCutoff = 0.99
DataResFlag = ut.data_fit_flag(DataRes, DataResCutoff, False)

# TitleStr = 'Data_Residual_PlanView'
# ut.color_plot_layer(UTM, TitleStr, DataRes, [np.min(DataRes), np.max(DataRes)], 'magma', 'Data Residual', Sz, 'False')

# TitleStr = 'Data_Residual_Cutoff_PlanView'
# ut.color_plot_layer(UTM, TitleStr, DataResFlag, [0, 1], 'viridis', 'Flag Value (1=keep; 0=reject)', Sz, 'False')

# %% ##################################################################
# ASSESS LATERAL TRANSITION ZONES
# Plan: Find all soundings in radius around each sounding, assess total variance
#       Plot histogram of variances, define cutoff to exclude samples
#       Perform extensive sanity checks!

VarCalcDist_Start = 15      # Effective measurement footprint (m)
VarCalcDist = [VarCalcDist_Start + y*np.tan(IndAngle*np.pi/180) for y in DepthCut[0, :]]

LogVarAll = np.zeros((nS, nL))
for i in range(nS):
    Pt = UTM[i, :]
    Dist = sp.cdist(UTM, [Pt], metric='euclidean')
    LogVarAll[i, :] = [np.var(LogRhoCut[np.where(Dist < VarCalcDist[j])[0], j], axis=0)
                       for j in range(nL)]

ThickCut = DepthCut - np.append(np.zeros((nS, 1)), DepthCut[:, 0: -1], axis=1)

WCLV = np.sum(LogVarAll * ThickCut, 1)

HistVar = WCLV[np.where(EdgeFlag * DataResFlag)[0]]
HistPlotParams['Edges'] = 41
HistPlotParams['XLabel'] = 'Cumulaive Log10 Lateral Variance'
HistPlotParams['YLabel'] = '# Counts'
HistPlotParams['Title'] = 'Weighted_Cumulative_Lateral_Variance'
HistPlotParams['Quantiles'] = []
# ut.hist_gko(HistVar, HistPlotParams)


MinLogVarAll = 0
MaxLogVarAll = np.quantile(LogVarAll, 0.95)
IncLogVarAll = 10**(np.ceil(np.log10(MaxLogVarAll))-2)/4
HistPlotParams['Edges'] = np.arange(MinLogVarAll, MaxLogVarAll, IncLogVarAll)

HistPlotParams['Title'] = r'Var($\log_{10}\rho$) Histograms'
HistPlotParams['XLabel'] = r'Var($\log_{10}\rho$ $(\Omega$m))'
HistPlotParams['YLabel'] = 'Depth (m)'
HistPlotParams['Quantiles'] = [0.25]
HistPlotParams['FaceColor'] = [0, 0, 1, 0.75]     # [R, G, B, Alpha]

# ut.sounding_hist(LogVarAll.T, DepthCut[0], HistPlotParams)

WCLV_Flags = np.zeros(nS)
WCLV_Flags[np.where(WCLV < np.quantile(WCLV, WCLV_Thresh))] = 1


# %% ##################################################################
# Show total cancelled area


AllFlags = np.ones(nS).astype(int)       # 1 = use sounding, 2 = do not use sounding
# AllFlags = WCLV_Flags * EdgeFlag * DataResFlag
AllFlags = EdgeFlag * DataResFlag

TitleStr = 'All_Flags_PlanView'
ut.color_plot_layer(UTM, TitleStr, AllFlags, [0, 1], 'viridis',
                    'Flag Value (1=keep; 0=reject)', Sz, 'False')


# %% ##################################################################

# PCA!!!

nPC = 3  # number of pca components

pca = PCA(n_components=nPC)      # Set up PCA
standscal = StandardScaler()           # Process to normalize variables to same scale

PC_Train, PC_Score = train_test_split(LogRhoCut, test_size=0.75)

# Run data through standard scaler before PCA
LR_SC = standscal.fit_transform(LogRhoCut)

PC_Score = pca.fit_transform(LR_SC)

print('Mean = ' + str(np.mean(PC_Score).round(3)))
print('Std = ' + str(np.var(PC_Score).round(3)))
print('Eigenvectors, or PC Loadings (a_k) = ' + str(pca.components_))
print('Eigenvalues (lambda) = ' + str(pca.explained_variance_))
print('Explained Variance Ratio (pi_j)= ' + str(pca.explained_variance_ratio_))

OutData = np.concatenate((UTM, RhoCut, DepthCut, np.reshape(EdgeFlag, (nS, 1)),
                          np.reshape(DataResFlag, (nS, 1)), np.reshape(WCLV_Flags,
                          (nS, 1)), PC_Score), axis=1)

OutFileName = 'Clarklind_Flags_PCA.txt'

np.savetxt(outputpath + OutFileName, OutData)
# %% Find Sounding According to Lateral Variance Quantile (only non-flagged soundings allowed)

# SoundingQuant = 1
# Idx = np.arange(nS)
# WCLV_Masked = WCLV * 1
# WCLV_Masked[np.where(1 - AllFlags[:,0])[0]] = np.nan
# WCLV_MaskQuantValue = np.nanquantile(WCLV_Masked, SoundingQuant)
# IdxSelect = np.abs(WCLV - WCLV_MaskQuantValue).argmin()
# WCLV_Quant = np.count_nonzero(WCLV<WCLV_MaskQuantValue) / WCLV.size


# SelectPoint = np.asarray(UTM[IdxSelect, :])
# PlotRangeHorz = np.asarray([800, 00])
# PlotRangeVert = np.asarray([00, 800])
# LineHorz = np.append(SelectPoint - PlotRangeHorz, SelectPoint + PlotRangeHorz).reshape((1, 4))
# LineVert = np.append(SelectPoint - PlotRangeVert, SelectPoint + PlotRangeVert).reshape((1, 4))

# Lines = np.concatenate((LineHorz, LineVert), axis=0)

# XSectionPlotParams['SelectPoint'] = SelectPoint
# XSectionPlotParams['FigName'] = ['EastingXSection', 'NorthingXSection']

# # fLH, axLH = ut.xsection(UTM, Elev, DepthCut, RhoCut, DOI_Con, Lines, XSectionPlotParams)

# Dist, NearSoundIdx = find_soundings(UTM, UTM[IdxSelect], np.max(VarCalcDist))

# RhoPlot = RhoCut[NearSoundIdx].T
# DepthPlot = DepthCut[NearSoundIdx].T
# DOI_Plot = DOI_Con[NearSoundIdx]

# SoundingPlotParams = copy.deepcopy(SoundingPlotParams_Basic)
# SoundingPlotParams['TitleStr'] = 'WCLV Sounding Quantile: ' + str('{:.2f}'.format(SoundingQuant))
# SoundingPlotParams['FigID'] = 'Min WCLV Sounding'
# SoundingPlotParams['LnClr'] = SoundingColorFun('r', Dist)

# # f = ut.tem_plot_1d(RhoPlot, DepthPlot, DOI_Plot, SoundingPlotParams)

# HistPlotParams['Edges'] = 41
# HistPlotParams['XLabel'] = 'Cumulaive Log10 Lateral Variance'
# HistPlotParams['YLabel'] = '# Counts'
# HistPlotParams['Title'] = 'Weighted_Cumulative_Lateral_Variance'
# HistPlotParams['Quantiles'] = [WCLV_Quant]
# # ut.hist_gko(WCLV, HistPlotParam


# %% STEP 1: Divide PC_Score into nDL strata, calc correlation matrix for PC_Score

AllFlagIdx = np.where(AllFlags)[0].astype(int)
UTM_Subset = UTM[AllFlagIdx]
MaxAx = (np.max(UTM, axis=0) - np.min(UTM, axis=0)).argmax()                    # Get coordinate axis with maximum range
UTM_Normalized = (UTM - np.mean(UTM, axis=0))/np.std(UTM[:, MaxAx], axis=0)      # UTM Coordinates normalized so axis with max range has variance of 1
# UTM_Norm_Subset = UTM_Normalized[AllFlagIdx]

# PC_Score_Subset = PC_Score[AllFlagIdx]
# LogRhoCutSubset = LogRhoCut[AllFlagIdx]
# WCLV_Subset = WCLV[AllFlagIdx]

LHS_Params_All = np.concatenate((PC_Score, UTM_Normalized), axis=1)
# LHS_Params_Subset = np.concatenate((PC_Score_Subset, UTM_Norm_Subset), axis=1)

IdxSubset = np.delete(np.asarray(range(nS)), np.where(1 - AllFlags))

# Create sparse array of LHS Parameters

nSS = np.shape(AllFlagIdx)[0]
nCl = int(np.round(nSS/10))
km = kmeans(n_clusters=nCl).fit(UTM_Subset)

Cl_Labels = km.labels_
MinMeanIdx = np.zeros((nCl,)).astype(int)

for i in range(nCl):
    Idx_Cl = np.where(Cl_Labels == i)[0]
    nIdx = np.size(Idx_Cl)

    Dist_All = sp.cdist(UTM_Subset[Idx_Cl], UTM_Subset[Idx_Cl], metric='euclidean')
    MinMeanIdx[i] = Idx_Cl[np.mean(Dist_All, axis=1).argmin()]

# UTM_Norm_Sparse = UTM_Norm_Subset[MinMeanIdx, :]
# PC_Score_Sparse = PC_Score_Subset[MinMeanIdx, :]
# WCLV_Sparse = WCLV_Subset[MinMeanIdx]
# LHS_Params_Sparse = np.concatenate((PC_Score_Sparse, UTM_Norm_Sparse), axis=1)

SparseIdx = AllFlagIdx[MinMeanIdx]
# OutData = np.concatenate((LHS_Params, IdxSubset.reshape(-1,1)), axis = 1)
# OutPath = 'C:\\Users\\gordon.osterman\\Dropbox\\USDA Documents\\Github Documents\\Sampling_Design\\Outputs\\LHS_UTM_Idx.txt'
# np.savetxt(OutPath, OutData)

# %%
# LHS_Params = np.zeros((50**2, 2))
# c = 0
# for i in range(1, 51):
#     for j in range(1, 51):
#             LHS_Params[c] = [i,j]
#             c += 1


nLHS = np.shape(LHS_Params_All)[1]

Q = np.arange(0, (nDL + 0.9)/nDL, 1/nDL)
QDist = np.asarray([np.quantile(LHS_Params_All[:, i], Q) for i in range(nLHS)]).T
QDist[-1] += 1

nRuns = 10000                     # Number of times to repeat LHS procedure
nIter_Run01 = 2000            # Number of iterations to perform during each LHS procedure
nIter_Run02 = 8000           # Number of iterations to perform during each LHS procedure


WA, WB, WC = 1, 2, 75

WW = [WA, WB, WC]

P = 0.10                # Probability of swapping random sample instead of worst
# CoolRate_Run01 = 0.0003
# CoolRate_Run02 = 0.00003

CoolRate_Run01 = 0.95
CoolRate_Run02 = 0.95

StartTemp_Run01 = 1
StartTemp_Run02 = CoolRate_Run01**nIter_Run01
MinTemp = 1e-7


T = tm.time()

SA_Params = {
    'StartIdx': [],
    'nIter': nIter_Run01,
    'nRuns': nRuns,
    'nDL': nDL,
    'P': P,
    'CR': CoolRate_Run01,
    'Ts': StartTemp_Run01,
    'Tm': MinTemp,
    }


Out_Run01 = ModCLHS(SA_Params, QDist, LHS_Params_All, WCLV, SparseIdx, WW)

ObjFuncA_Run01, ObjFuncB_Run01, ObjFuncC_Run01, AcceptIdx_Run01 = Out_Run01


print(' ')

SA_Params = {
    'StartIdx': AcceptIdx_Run01[-1],
    'nIter': nIter_Run02,
    'nRuns': nRuns,
    'nDL': nDL,
    'P': P,
    'CR': CoolRate_Run02,
    'Ts': np.max((StartTemp_Run02, MinTemp)),
    'Tm': MinTemp,
    }

Out_Run02 = ModCLHS(SA_Params, QDist, LHS_Params_All, WCLV, AllFlagIdx, WW)

ObjFuncA_Run02, ObjFuncB_Run02, ObjFuncC_Run02, AcceptIdx_Run02 = Out_Run02

ObjFuncA = np.concatenate([WA*ObjFuncA_Run01, WA*ObjFuncA_Run02], axis=0)*WA
ObjFuncB = np.concatenate([WA*ObjFuncB_Run01, WA*ObjFuncB_Run02], axis=0)*WB
ObjFuncC = np.concatenate([WA*ObjFuncC_Run01, WA*ObjFuncC_Run02], axis=0)*WC
ObjFuncFull = ObjFuncA + ObjFuncB + ObjFuncC

print('Total LHS Time = ' + '{:.2f}'.format(tm.time() - T))
######################################
# %%

BestRunIdx = ObjFuncFull[-1].argsort()[5000]

AcceptIdx_Final = AcceptIdx_Run02[-1, BestRunIdx]
LHS_Final = LHS_Params_All[AcceptIdx_Final]

nQPts_Final = np.zeros((nDL, nLHS))
for i in range(nLHS):
    nQPts_Final[:, i] = np.histogram(LHS_Final[:, i], bins=QDist[:, i])[0]

plt.figure('Objective Function', figsize=(4, 3), dpi=300)
ax = plt.axes(
    xlabel='Iteration',
    ylabel=r'$\phi$ (-)',
)
# plt.rc('font', size=30)
plt.rc('axes', labelsize=12)
plt.plot(ObjFuncA[:, BestRunIdx], 'b--', linewidth=2, label=r'$\phi_1$')
plt.plot(ObjFuncB[:, BestRunIdx], 'r-.', linewidth=2, label=r'$\phi_2$')
plt.plot(ObjFuncC[:, BestRunIdx], 'm:', linewidth=2, label=r'$\phi_3$')
plt.plot(ObjFuncFull[:, BestRunIdx], 'k-', linewidth=2.5, label=r'$\phi_T$')
plt.legend(loc='upper right', framealpha=1, edgecolor='k', fontsize=12, ncol=2)
ax.tick_params(axis='both', labelsize=12)
ax.set_xlim([-10, nIter_Run01 + nIter_Run02])
ax.set_ylim([0, 60])
# ax.set_xticks([0, 10000, 20000])
# ax.set_xticklabels(['0', '10', '20'])
# ax.set_yticks([0, 5, 10 , 15])

plt.tight_layout()

print('Lateral Variance Quantiles')
for i in range(nLHS):
    WCLV_Quantile = np.where(np.sort(WCLV) <
                             WCLV[AcceptIdx_Final][i])[0].size/WCLV.size

    print(f'{WCLV_Quantile:.3f}')
    print(' ')

SavePath = 'C://Users//gordon.osterman//Dropbox//USDA Documents//Github Documents//Sampling_Design//Manuscript Figures//'

plt.savefig(SavePath + 'SimAnnealing.pdf', dpi=300, format='pdf')
plt.savefig(SavePath + 'SimAnnealing.png', dpi=300, format='png')

SoundingIdxLHS = AcceptIdx_Run02[-1]

# Output Format: [AcceptIdx(1) ... AcceptIdx(nDL, ObjFuncA, ObjFuncC, ObjFuncAll)],
#       where all values are output from the last LHS iteration
OutData = np.concatenate((SoundingIdxLHS, np.reshape(ObjFuncA[-1], (nRuns, 1)),
                          np.reshape(ObjFuncB[-1], (nRuns, 1)), np.reshape(ObjFuncC[-1],
                          (nRuns, 1)), np.reshape(ObjFuncFull[-1], (nRuns, 1))), axis=1)

OutFileName = 'Clarklind_mLHS_04_WA01WB02WC75.txt'

np.savetxt(outputpath + OutFileName, OutData)





# Edges = np.arange(0, 40)
# plt.figure()
# ax = plt.axes()
# ax.hist(np.ravel(ObjFuncA*WA), bins=Edges)

# plt.figure()
# ax = plt.axes()
# ax.hist(np.ravel(ObjFuncB*WB), bins=Edges)

# plt.figure()
# ax = plt.axes()
# ax.hist(np.ravel(ObjFuncC*WC), bins=Edges)











# %%
# RANDOM SAMPLING


nSims = 1000                # Number of random sampling simulations to run

MeanDevRand = np.zeros((nSims, 1))
StdDevRand = np.zeros((nSims, 1))
RangePctRand = np.zeros((nSims, 1))
MeanWCLV_Rand = np.zeros((nSims, 1))
GeomOffsetRand = np.zeros((nSims, 1))
ObjFuncARand = np.zeros((nSims, 1))
ObjFuncBRand = np.zeros((nSims, 1))
ObjFuncCRand = np.zeros((nSims, 1))
ObjFuncFullRand = np.zeros((nSims, 1))

SamplingParams = np.concatenate((PC_Score, UTM_Normalized), axis=1)

for i in range(nSims):
    RandIdx = np.random.choice(np.where(AllFlags)[0], nDL, replace=False)    
    MeanDev_Temp, StdDev_Temp, RangePct_Temp, GeomOffset_Temp = ut.validation_metrics(LogRhoCut.T, UTM, RandIdx)

    nQPts = np.zeros((nDL, nLHS))
    for j in range(nLHS):
        nQPts[:, j] = np.histogram(SamplingParams[RandIdx, j], bins=QDist[:,j])[0]

    WCLV_Select = WCLV[RandIdx] 

    ObjFuncAllRand = calc_obj_func(nQPts, SamplingParams[RandIdx], SamplingParams, WCLV_Select, WW)
    
    # MeanDevRand[i] = MeanDev_Temp
    # StdDevRand[i] = StdDev_Temp
    # RangePctRand[i] = RangePct_Temp
    # MeanWCLV_Rand[i] = np.mean(WCLV[RandIdx])
    # GeomOffsetRand[i] = GeomOffset_Temp
    ObjFuncARand[i] = ObjFuncAllRand[0]
    ObjFuncBRand[i] = ObjFuncAllRand[1]
    ObjFuncCRand[i] = ObjFuncAllRand[2]
    ObjFuncFullRand[i] = ObjFuncAllRand[3]
    

# NormMeanDevDL = MeanDevRand/np.max(MeanDevRand)
# NormStdDevDL = StdDevRand/np.max(StdDevRand)
# NormRangePctDL = 1 - RangePctRand/np.max(RangePctRand)
# NormMeanWCLV_DL = MeanWCLV_Rand/np.max(MeanWCLV_Rand)
# NormGeomOffsetDL = 1 - GeomOffsetRand/np.max(GeomOffsetRand)
# BestSimIdx = (NormMeanDevDL + NormStdDevDL + NormRangePctDL + NormMeanWCLV_DL + NormGeomOffsetDL).argmin()

# %%
# VALIDATION!
OutFileName = 'Clarklind_mLHS_00.txt'
# FROM PYTHON
AllResults = np.loadtxt(outputpath + OutFileName)
SoundingIdx = AllResults[:, :nDL].astype(int)
ObjFuncA = AllResults[:, nDL]
ObjFuncB = AllResults[:, nDL + 1]
ObjFuncC = AllResults[:, nDL + 2]
ObjFuncFull = AllResults[:, -1]
# BestRunIdx = ObjFuncFull[-1].argmin()
# SoundingIdx = SoundingIdxLHS


# From MATLAB
# AllResults = np.loadtxt('C:\\Users\gordon.osterman\\Dropbox\\USDA Documents\\Github Documents\\Sampling_Design\\Outputs\\IdxSelect.txt')
SoundingIdx = AllResults[:, :nDL].astype(int)
# ObjFuncA = AllResults[:, nDL + 1].astype(int)
# ObjFuncC = AllResults[:, nDL + 2].astype(int)
# ObjFuncFull = AllResults[:, -1].astype(int)
# # BestRunIdx = ObjFuncFull[-1].argmin()
# SoundingIdx = SoundingIdxLHS

nRuns = np.shape(SoundingIdx)[0]

MeanDev_True, StdDev_True, RangePct_True, GeomOffset_True = ut.validation_metrics(LogRhoCut.T, UTM, TrueIdx)
# MeanWCLV_True = np.mean(WCLV[TrueIdx])
nQPts = np.zeros((nDL, nLHS))


for j in range(nLHS):
    nQPts[:, j] = np.histogram(SamplingParams[TrueIdx, j], bins=QDist[:,j])[0]
WCLV_Select = WCLV[TrueIdx]
ObjFuncAllTrue = calc_obj_func(nQPts, SamplingParams[TrueIdx], SamplingParams, WCLV_Select, WW)
ObjFuncATrue = ObjFuncAllRand[0]
ObjFuncBTrue = ObjFuncAllRand[1]
ObjFuncCTrue = ObjFuncAllRand[2]
ObjFuncFullTrue = ObjFuncAllRand[3]

# %% 
# Check Objective function performance
BestSimIdx = ObjFuncFull.argsort()[0]

Wghts = np.ones(nRuns)/nRuns
HistPlotParams['Weight'] = 'Percentile'
HistPlotParams['YLabel'] = 'Fraction'
HistPlotParams['FaceColor'] = [0, 0, 1, 0.75]
HistPlotParams['EdgeColor'] = 'gray'

HistVar = ObjFuncARand
# QuantPlot = np.count_nonzero(MeanDevRand<MeanDevDL[BestRunIdx]) / MeanDevRand.size
HistPlotParams['XLabel'] = 'LHS ObjFuncA'
HistPlotParams['Title'] = 'LHS ObjFuncA'# + str(np.round(QuantPlot, 2))
HistPlotParams['Edges'] = np.arange(0, 40, 2)
HistPlotParams['XLims'] = [0, 40]
HistPlotParams['Quantiles'] = []
f_objA, ax_objA = ut.hist_gko(HistVar, HistPlotParams)
HistData = ObjFuncA
ax_objA.hist(HistData, HistPlotParams['Edges'], edgecolor='gray', weights=Wghts, fc = [1, 0.65, 0, 0.75], label='LHS')

ymx = ax_objA.get_ylim()[1]
ax_objA.get_children()[0].set_label('Random Sampling')
ax_objA.plot([ObjFuncA[BestRunIdx], ObjFuncA[BestRunIdx]], [0, ymx], 'r--', linewidth=4, label='Best Samples')
ax_objA.plot([ObjFuncARand[BestSimIdx], ObjFuncARand[BestSimIdx]], [0, ymx], 'm--', linewidth=4, label='Best Random')
ax_objA.plot([ObjFuncATrue, ObjFuncATrue], [0, ymx], 'k', linewidth=4, label='True Samples')
ax_objA.legend(handlelength=4)
# f_objA.savefig(figpath + 'LHS ObjFuncA PCA')


HistVar = ObjFuncBRand*WB
# QuantPlot = np.count_nonzero(MeanDevRand<MeanDevDL[BestRunIdx]) / MeanDevRand.size
HistPlotParams['XLabel'] = 'LHS ObjFuncB'
HistPlotParams['Title'] = 'LHS ObjFuncB'# + str(np.round(QuantPlot, 2))
HistPlotParams['Edges'] = np.arange(0, 40, 2)
HistPlotParams['XLims'] = [0, 40]
HistPlotParams['Quantiles'] = []
f_objB, ax_objB = ut.hist_gko(HistVar, HistPlotParams)
HistData = ObjFuncB*WB
ax_objB.hist(HistData, HistPlotParams['Edges'], edgecolor='gray', weights=Wghts, fc = [1, 0.65, 0, 0.75], label='LHS')

ymx = ax_objB.get_ylim()[1]
ax_objB.get_children()[0].set_label('Random Sampling')
# ax_objB.plot([ObjFuncB[BestRunIdx], ObjFuncB[BestRunIdx]], [0, ymx], 'r--', linewidth=4, label='Best Samples')
# ax_objB.plot([ObjFuncBRand[BestSimIdx], ObjFuncBRand[BestSimIdx]], [0, ymx], 'm--', linewidth=4, label='Best Random')
ax_objB.plot([ObjFuncBTrue*WB, ObjFuncBTrue*WB], [0, ymx], 'k', linewidth=4, label='True Samples')
ax_objB.legend(handlelength=4)
# f_objC.savefig(figpath + 'LHS ObjFuncC PCA')


HistVar = ObjFuncCRand
# QuantPlot = np.count_nonzero(MeanDevRand<MeanDevDL[BestRunIdx]) / MeanDevRand.size
HistPlotParams['XLabel'] = 'LHS ObjFuncC'
HistPlotParams['Title'] = 'LHS ObjFuncC'# + str(np.round(QuantPlot, 2))
HistPlotParams['Edges'] = np.arange(0, 40, 2)
HistPlotParams['XLims'] = [0, 40]
HistPlotParams['Quantiles'] = []
f_objC, ax_objC = ut.hist_gko(HistVar, HistPlotParams)
HistData = ObjFuncC
ax_objC.hist(HistData, HistPlotParams['Edges'], edgecolor='gray', weights=Wghts, fc = [1, 0.65, 0, 0.75], label='LHS')

ymx = ax_objC.get_ylim()[1]
ax_objC.get_children()[0].set_label('Random Sampling')
ax_objC.plot([ObjFuncC[BestRunIdx], ObjFuncC[BestRunIdx]], [0, ymx], 'r--', linewidth=4, label='Best Samples')
ax_objC.plot([ObjFuncCRand[BestSimIdx], ObjFuncCRand[BestSimIdx]], [0, ymx], 'm--', linewidth=4, label='Best Random')
ax_objC.plot([ObjFuncCTrue, ObjFuncCTrue], [0, ymx], 'k', linewidth=4, label='True Samples')
ax_objC.legend(handlelength=4)
# f_objC.savefig(figpath + 'LHS ObjFuncC PCA')


HistVar = ObjFuncFullRand
# QuantPlot = np.count_nonzero(MeanDevRand<MeanDevDL[BestRunIdx]) / MeanDevRand.size
HistPlotParams['XLabel'] = 'LHS ObjFuncFull'
HistPlotParams['Title'] = 'LHS ObjFuncFull'# + str(np.round(QuantPlot, 2))
HistPlotParams['Edges'] = np.arange(0, 40, 2)
HistPlotParams['XLims'] = [0, 40]
HistPlotParams['Quantiles'] = []
f_objF, ax_objF = ut.hist_gko(HistVar, HistPlotParams)
HistData = ObjFuncFull
ax_objF.hist(HistData, HistPlotParams['Edges'], edgecolor='gray', weights=Wghts, fc = [1, 0.65, 0, 0.75], label='LHS')

ymx = ax_objF.get_ylim()[1]
ax_objF.get_children()[0].set_label('Random Sampling')
ax_objF.plot([ObjFuncFull[BestRunIdx], ObjFuncFull[BestRunIdx]], [0, ymx], 'r--', linewidth=4, label='Best Samples')
ax_objF.plot([ObjFuncFullRand[BestSimIdx], ObjFuncFullRand[BestSimIdx]], [0, ymx], 'm--', linewidth=4, label='Best Random')
ax_objF.plot([ObjFuncFullTrue, ObjFuncFullTrue], [0, ymx], 'k', linewidth=4, label='True Samples')
ax_objF.legend(handlelength=4)
# f_objF.savefig(figpath + 'LHS ObjFuncFull PCA')

# %%
Stats = [
    'mean',
    'std',
]
StatsPlotParams = {
    'Title' : 'LHS LogRho Stats Comparison',
    'XLabel' : r'$\log_{10}\rho$ $(\Omega$m)',
    'YLabel' : 'Depth (m)',
    'YLim' : [0, SoundingMaxDepth],
    'XLim' : [1, 2],
}

BestSoundingIdx = SoundingIdxLHS[ObjFuncFull.argsort()[0]]

f6, ax6 = ut.validation_charstats(LogRhoCut.T, DepthCut[0], BestSoundingIdx, Stats, StatsPlotParams)
# f6.savefig(figpath + 'LHS Stats Comparison PCA')



MinLogRhoAll = 1 #np.quantile(LogRhoCut, 0.01)
MaxLogRhoAll = 2.1 #np.quantile(LogRhoCut, 0.99)
IncLogRhoAll = 10**np.floor(np.log10((MaxLogRhoAll - MinLogRhoAll)/41))
HistPlotParams['Edges'] = np.arange(MinLogRhoAll, MaxLogRhoAll, IncLogRhoAll)

HistPlotParams['Title'] = r'$\log_{10}\rho$ Histograms'
HistPlotParams['XLabel'] = r'$\log_{10}\rho$ $(\Omega$m)'
HistPlotParams['YLabel'] = 'Depth (m)'
HistPlotParams['Quantiles'] = []

f7, ax7 = ut.validation_hist(LogRhoCut.T, DepthCut[0], BestSoundingIdx, HistPlotParams)
# f7.savefig(figpath + 'LHS Sounding Comparison PCA')




TitleStr = 'LHS LogRho Selections'
f8, ax8 = ut.color_plot_layer(UTM, TitleStr, LogRhoCut[:, 6], [0, 1], 'jet', r'$\log_{10}\rho$ $(\Omega$m)', Sz, 'False')
ax8.scatter(UTM[BestSoundingIdx, 0], UTM[BestSoundingIdx, 1], c = 'w', s = 100, edgecolors = 'k', linewidths=1, marker='v', label='Best')
ax8.scatter(TrueUTM[:, 0], TrueUTM[:, 1], c = 'r', s = 100, edgecolors = 'k', linewidths=1, marker='o', label='True')
ax8.legend(loc='lower left', framealpha=1, edgecolor='k')
# f8.savefig(figpath + 'LHS Selected Soundings PCA')

plt.show()




#%%
# Find Sounding According to Lateral Variance Quantile (only non-flagged soundings allowed)

# SoundingQuant = 1
# Idx = np.arange(nS)
# WCLV_Masked = WCLV * 1
# WCLV_Masked[np.where(1 - AllFlags[:,0])[0]] = np.nan
# WCLV_MaskQuantValue = np.nanquantile(WCLV_Masked, SoundingQuant)
# IdxSelect = np.abs(WCLV - WCLV_MaskQuantValue).argmin()
# WCLV_Quant = np.count_nonzero(WCLV<WCLV_MaskQuantValue) / WCLV.size


IdxSelect = BestSoundingIdx[4]

SelectPoint = np.asarray(UTM[IdxSelect, :])
PlotRangeHorz = np.asarray([800, 00])
PlotRangeVert = np.asarray([00, 800])
LineHorz = np.append(SelectPoint - PlotRangeHorz, SelectPoint + PlotRangeHorz).reshape((1, 4))
LineVert = np.append(SelectPoint - PlotRangeVert, SelectPoint + PlotRangeVert).reshape((1, 4))

Lines = np.concatenate((LineHorz, LineVert), axis=0)

XSectionPlotParams['SelectPoint'] = SelectPoint
XSectionPlotParams['FigName'] = ['EastingXSection', 'NorthingXSection']

fLH, axLH = ut.xsection(UTM, Elev, DepthCut, RhoCut, DOI_Con, Lines, XSectionPlotParams)

Dist, NearSoundIdx = find_soundings(UTM, UTM[IdxSelect], np.max(VarCalcDist))

RhoPlot = RhoCut[NearSoundIdx].T
DepthPlot = DepthCut[NearSoundIdx].T
DOI_Plot = DOI_Con[NearSoundIdx]

SoundingPlotParams = copy.deepcopy(SoundingPlotParams_Basic)
SoundingPlotParams['TitleStr'] = 'WCLV=' + str('{:.3f}'.format(WCLV[IdxSelect]))
SoundingPlotParams['FigID'] = 'Min WCLV Sounding'
SoundingPlotParams['LnClr'] = SoundingColorFun('r', Dist)

f = ut.tem_plot_1d(RhoPlot, DepthPlot, DOI_Plot, SoundingPlotParams)


x=1

# %% ##################################################################
# PC 2D scatter plots

# Clr = ['r', 'b', 'g']
# Mx = np.max(np.abs(LR_PCA))*1.05
# Mn = -Mx

# plt.figure(figsize=(8,8))
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.xlabel('PC 1', fontsize=20)
# plt.ylabel('PC 2', fontsize=20)
# plt.xlim([Mn, Mx])
# plt.ylim([Mn, Mx])

# # plt.scatter(LR_PCA[:, 0], LR_PCA[:, 1], c=Clr[0], s= 10)
# plt.scatter(LR_PCA[:, 0], LR_PCA[:, 1], c=VarFlag, s= 10)
# plt.scatter(PCAVarAll[:, 0], PCAVarAll[:, 1], c='w', s= 100, edgecolors='k', linewidths=1.5)
# for i in range(nDL):
#     plt.scatter(LR_PCA[SoundingIdxAll[i], 0], LR_PCA[SoundingIdxAll[i], 1], c='k', s= 35)

# plt.figure(figsize=(8,8))
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.xlabel('PC 1', fontsize=20)
# plt.ylabel('PC 3', fontsize=20)
# plt.xlim([Mn, Mx])
# plt.ylim([Mn, Mx])

# # plt.scatter(LR_PCA[:, 0], LR_PCA[:, 2], c=Clr[1], s= 10)
# plt.scatter(LR_PCA[:, 0], LR_PCA[:, 2], c=VarFlag, s= 10)
# plt.scatter(PCAVarAll[:, 0], PCAVarAll[:, 2], c='w', s= 100, edgecolors='k', linewidths=1.5)
# for i in range(nDL):
#     plt.scatter(LR_PCA[SoundingIdxAll[i], 0], LR_PCA[SoundingIdxAll[i], 2], c='k', s= 35)

# plt.figure(figsize=(8,8))
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)
# plt.xlabel('PC 2', fontsize=20)
# plt.ylabel('PC 3', fontsize=20)
# plt.xlim([Mn, Mx])
# plt.ylim([Mn, Mx])

# # plt.scatter(LR_PCA[:, 1], LR_PCA[:, 2], c=Clr[2], s= 10)
# plt.scatter(LR_PCA[:, 1], LR_PCA[:, 2], c=VarFlag, s= 10)
# plt.scatter(PCAVarAll[:, 1], PCAVarAll[:, 2], c='w', s= 100, edgecolors='k', linewidths=1.5)
# for i in range(nDL):
#     plt.scatter(LR_PCA[SoundingIdxAll[i], 1], LR_PCA[SoundingIdxAll[i], 2], c='k', s= 35)

# %% ##################################################################
# PC 3D Scatter Plot

# Mx = np.max(np.abs(PC_Score))*1.1
# Mn = -Mx

# fig = plt.figure(figsize=(8,8))
# ax = fig.add_subplot(projection='3d')
# plt.xticks(fontsize=12)
# plt.yticks(fontsize=12)

# ax.set_xlabel('PC 1', fontsize=20)
# ax.set_ylabel('PC 2', fontsize=20)
# ax.set_zlabel('PC 3', fontsize=20)
# # plt.zlabel('PC 3', fontsize=20)
# plt.xlim([Mn, Mx])
# plt.ylim([Mn, Mx])
# # plt.zlim([Mn, Mx])
# ax.set_zlim([Mn, Mx])
# # ax.scatter(LR_PCA[:, 0], LR_PCA[:, 1], LR_PCA[:, 2], c = 'b', s= 8, marker='.')
# ax.scatter(PC_Score[:, 0], PC_Score[:, 1], PC_Score[:, 2], c = WCLV, s= 8, marker='.')

# # ax.scatter(PCAVarAll[:, 0], PCAVarAll[:, 1], PCAVarAll[:, 2], c = 'k', s= 80, alpha=1)

# # plt.show()

# %% ##################################################################
# Plot Selected 1D Soundings

# for i in range(nDL):
#     SoundingIdx = SoundingIdxThresh[i, 0]
#     # SoundingIdx = SoundingIdxMinVar[i, 0]

#     Dist, NearSoundIdx = find_soundings(UTM, UTM[SoundingIdx, :], np.max(VarCalcDist))
#     WCLV_Quant = np.count_nonzero(WCLV<WCLV[SoundingIdx]) / WCLV.size

#     RhoPlot = RhoCut[NearSoundIdx, :].T
#     DepthPlot = DepthCut[NearSoundIdx, :].T
#     DOI_Plot = DOI_Con[NearSoundIdx]
    
#     SoundingPlotParams = copy.deepcopy(SoundingPlotParams_Basic)
#     # Title = 'PC Score ' + ' '.join(['{:0.2}'.format(PC_Score[SoundingIdx,j]) + ',' for j in range(3)]) + ' Q: {:0.2}'.format(WCLV_Quant)
#     Title = 'PC Score ' + ' '.join([str(np.round(PC_Score[SoundingIdx,j], 1)) + ',' for j in range(3)]) + ' Q: {:0.2}'.format(WCLV_Quant)
#     SoundingPlotParams['TitleStr'] = Title
#     SoundingPlotParams['FigID'] = Title
#     SoundingPlotParams['LnClr'] = SoundingColorFun('r', Dist)
#     str('{:.2f}'.format(SoundingQuant))
#     ut.tem_plot_1d(RhoPlot, DepthPlot, DOI_Plot, SoundingPlotParams)

#     HistPlotParams['Edges'] = 41
#     HistPlotParams['XLabel'] = 'Cumulaive Log10 Lateral Variance'
#     HistPlotParams['YLabel'] = '# Counts'
#     HistPlotParams['Title'] = Title + ' Hist'
#     HistPlotParams['Quantiles'] = [WCLV_Quant]
#     ut.hist_gko(WCLV, HistPlotParams)























































# %% # %% ##################################################################
# VALIDATION: Check what soundings look like 

# PlotRangeHorz = np.asarray([800, 00])
# PlotRangeVert = np.asarray([00, 800])

# SoundingIdx = SoundingIdxLHS

# UTM_Ordered_Final = UTM[SoundingIdx]
# SoundingPlotParams =  SoundingPlotParams_Basic
# for i in range(nDL):   
    
#     Dist, NearSoundIdx = find_soundings(UTM, UTM[SoundingIdx[i]], np.max(VarCalcDist))


#     RhoPlot = RhoCut[NearSoundIdx, :].T
#     DepthPlot = DepthCut[NearSoundIdx, :].T
#     DOI_Plot = DOI_Con[NearSoundIdx]

#     Title = 'Sounding_' + str(i+1)
#     SoundingPlotParams['TitleStr'] = Title
#     SoundingPlotParams['FigID'] = Title
#     SoundingPlotParams['LnClr'] = SoundingColorFun('r', Dist)
#     ut.tem_plot_1d(RhoPlot, DepthPlot, DOI_Plot, SoundingPlotParams)

#     SelectPoint = UTM_Ordered_Final[i]
    
#     LineHorz = np.append(SelectPoint - PlotRangeHorz, SelectPoint + PlotRangeHorz).reshape((1, 4))
#     LineVert = np.append(SelectPoint - PlotRangeVert, SelectPoint + PlotRangeVert).reshape((1, 4))

#     Lines = np.concatenate((LineHorz, LineVert), axis=0)

#     XSectionPlotParams['SelectPoint'] = SelectPoint
#     XSectionPlotParams['FigName'] = ['Final EastingXSection_RS' + str(i), 'NorthingXSection_RS' + str(i)]

#     fLH, axLH = ut.xsection(UTM, Elev, DepthCut, RhoCut, DOI_Con, Lines, XSectionPlotParams)
    



# MinLogRhoAll = 1 #np.quantile(LogRhoCut, 0.01)
# MaxLogRhoAll = 2.1 #np.quantile(LogRhoCut, 0.99)
# IncLogRhoAll = 10**np.floor(np.log10((MaxLogRhoAll - MinLogRhoAll)/41))
# HistPlotParams['Edges'] = np.arange(MinLogRhoAll, MaxLogRhoAll, IncLogRhoAll)

# HistPlotParams['Title'] = r'$\log_{10}\rho$ Histograms'
# HistPlotParams['XLabel'] = r'$\log_{10}\rho$ $(\Omega$m)'
# HistPlotParams['YLabel'] = 'Depth (m)'
# HistPlotParams['Quantiles'] = []

# f1, ax1 = ut.validation_hist(LogRhoCut.T, DepthCut[0], SoundingIdx, HistPlotParams)

# Stats = [
#     'mean',
#     'std',
# ]
# StatsPlotParams = {
#     'Title' : 'Stats Comparison',
#     'XLabel' : r'$\log_{10}\rho$ $(\Omega$m)',
#     'YLabel' : 'Depth (m)',
#     'YLim' : [0, SoundingMaxDepth],
#     'XLim' : [1, 2],
# }
# ut.validation_charstats(LogRhoCut.T, DepthCut[0], SoundingIdx[BestIdx], Stats, StatsPlotParams)

# plt.show()

# x=1


# %%
# # Run simulation randomly sampling soundings
# Available_Idx = np.where(AllFlags)[0]

# nSim = 10000

# MinDist = np.zeros((nSim, 1))
# Total_PC_Score = np.zeros((nSim, 1))
# Mean_WCLV = np.zeros((nSim, 1))
# Mean_Dev = np.zeros((nSim, 1))
# Std_Dev = np.zeros((nSim, 1))


# for i in range(nSim):
    
#     RandSoundIdx = np.random.choice(Available_Idx, nDL, replace=False)
    
#     MinDist_Temp, Total_PC_Score_Temp, Mean_WCLV_Temp, Mean_Dev_Temp, Std_Dev_Temp = ut.validation_metrics(UTM, LogRhoCut, PC_Score, WCLV, RandSoundIdx)
#     MinDist[i] = MinDist_Temp[1]
#     Total_PC_Score[i] = Total_PC_Score_Temp
#     Mean_WCLV[i] = Mean_WCLV_Temp
#     Mean_Dev[i] = Mean_Dev_Temp
#     Std_Dev[i] = Std_Dev_Temp

# MinDist_Select, Total_PC_Score_Select, Mean_WCLV_Select, Mean_Dev_Select, Std_Dev_Select = ut.validation_metrics(UTM, LogRhoCut, PC_Score, WCLV, SoundingIdxLHS)
# MinDist_Select = MinDist_Select[1]

# # %%
# HistVar = MinDist
# QuantPlot = np.count_nonzero(MinDist<MinDist_Select) / MinDist.size
# HistPlotParams['Edges'] = 41
# HistPlotParams['XLabel'] = 'Min Offset (m)'
# HistPlotParams['YLabel'] = '# Counts'
# HistPlotParams['Title'] = 'Minimum Offset Metric; Q=' + str(np.round(QuantPlot, 2))
# HistPlotParams['Quantiles'] = [QuantPlot]
# ut.hist_gko(HistVar, HistPlotParams)

# HistVar = Total_PC_Score
# QuantPlot = np.count_nonzero(Total_PC_Score<Total_PC_Score_Select) / Total_PC_Score.size
# HistPlotParams['Edges'] = 41
# HistPlotParams['XLabel'] = 'PC Balance (-)'
# HistPlotParams['YLabel'] = '# Counts'
# HistPlotParams['Title'] = 'PC Score Balance Metric; Q=' + str(np.round(QuantPlot, 2))
# HistPlotParams['Quantiles'] = [QuantPlot]
# ut.hist_gko(HistVar, HistPlotParams)

# HistVar = Mean_WCLV
# QuantPlot = np.count_nonzero(Mean_WCLV<Mean_WCLV_Select) / Mean_WCLV.size
# HistPlotParams['Edges'] = 41
# HistPlotParams['XLabel'] = 'Mean(WCLV) (-)'
# HistPlotParams['YLabel'] = '# Counts'
# HistPlotParams['Title'] = 'Mean Weighted Cumulative Lateral Variance Metric; Q=' + str(np.round(QuantPlot, 3))
# HistPlotParams['Quantiles'] = [QuantPlot]
# ut.hist_gko(HistVar, HistPlotParams)

# HistVar = Mean_Dev
# QuantPlot = np.count_nonzero(Mean_Dev<Mean_Dev_Select) / Mean_Dev.size
# HistPlotParams['Edges'] = 41
# HistPlotParams['XLabel'] = 'Total abs deviation of means'
# HistPlotParams['YLabel'] = '# Counts'
# HistPlotParams['Title'] = 'Total absolute deviation of means; Q=' + str(np.round(QuantPlot, 2))
# HistPlotParams['Quantiles'] = [QuantPlot]
# ut.hist_gko(HistVar, HistPlotParams)

# HistVar = Std_Dev
# QuantPlot = np.count_nonzero(Std_Dev<Std_Dev_Select) / Std_Dev.size
# HistPlotParams['Edges'] = 41
# HistPlotParams['XLabel'] = 'Total abs deviation of std'
# HistPlotParams['YLabel'] = '# Counts'
# HistPlotParams['Title'] = 'Total absolute deviation of standard deviations; Q=' + str(np.round(QuantPlot, 2))
# HistPlotParams['Quantiles'] = [QuantPlot]
# ut.hist_gko(HistVar, HistPlotParams)

# %% OLD CODE
#################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################
#################################################################################################

# %%
# Lesch (2005) methodology for selecting point locations
# Approach 1. Maximize total geometric distance between all selected sampling points
# Approach 2. Maximize minimum geometric distance between selected sampling points

# SoundingIdxAll_Search = np.array([SoundingIdxAll[i][0]for i in range(nDL)])
# UTM_Ordered = np.zeros((nDL, 2))
# UTM_Ordered[0, :] = UTM[SoundingIdxAll[0][0]]
# Idx_Ordered = np.zeros((nDL, 1)).astype(int)

# MaskArray = np.zeros((nDL, 1))
# MaskArray[0] = 1
# UTM_Search = UTM[SoundingIdxAll_Search]
# # COMPLETE ONE CYCLE THROUGH ALL SOUNDINGS; ASSUME "BEST" SOUNDING TO START
# UTM_Start = UTM[SoundingIdxAll_Search[0]]
# for i in range(1, nDL):
#     AllDist = [np.linalg.norm(UTM_Search[i,:] - UTM_Start) for i in range(nDL)]
#     AllDistMaxIdx = np.ma.masked_array(AllDist, MaskArray).argmax()

#     Idx_Ordered[i] = AllDistMaxIdx
#     UTM_Ordered[i, :] = UTM_Search[AllDistMaxIdx]
#     UTM_Start = UTM_Ordered[AllDistMaxIdx, :]
#     MaskArray[AllDistMaxIdx] = 1
    

# # UTM_Ordered[-1, :] = UTM_Search
# # Idx_Ordered [-1] = np.where(UTM_Ordered[-1, :] == UTM_Keep[SoundingIdxAll_Search])[0][1]


# # Plot original set of points
# TitleStr = DataFileName + '; Original Sequence (wh=first, bk=last)'
# FigNum = 'ResMap_StartPtsSeq_PlanView'
# RhoRng = [10, 120]
# Clr = cm.get_cmap('Greys', nDL)
# f, ax = ut.color_plot_layer(RhoCut, UTM, 6, TitleStr, RhoRng, Log, 20, FigNum)
# plt.figure(f)
# plt.scatter(UTM_Ordered[:, 0], UTM_Ordered[:, 1], c = Clr(np.linspace(0, 1, 8)), s = 100, edgecolors = 'k', linewidths=1, marker='v')

# plt.figure('All_Flags_StartPtsSeq_PlanView')
# plt.axes()
# plt.scatter(UTM[:, 0], UTM[:, 1], c=AllFlags, s=10)
# plt.xlabel('UTM Easting')
# plt.ylabel('UTM Northing ')
# plt.title(DataFileName + ' All Flags Original Seq')
# c = plt.colorbar(fraction=0.1)
# c.ax.set_ylabel('Flag Value (1=keep; 0=reject)')
# plt.scatter(UTM_Check[:, 0], UTM_Check[:, 1], c = Clr(np.linspace(0, 1, 8)), s = 100, edgecolors = 'k', linewidths=1, marker='v')


# # %%
# #  Iteratively select new points and test total distance, max minimum distance
# nIt = 2000
# StartDist = np.linalg.norm(UTM_Ordered - UTM_Ordered[:,None], axis=-1)                      # Euclidean distance btwn all points


# AllMaxAvgDist = np.zeros((nIt, 1))
# AllMaxAvgDist[0] = np.mean(StartDist[np.triu_indices(nDL, k=1)])               # Maximum Arithmetic Average distance

# AllMaxGeoDist = np.zeros((nIt, 1))
# AllMaxGeoDist[0] = st.gmean(StartDist[np.triu_indices(nDL, k=1)])              # Maximum Geometric Average distance

# AllMaxMinDist = np.zeros((nIt, 1))
# AllMaxMinDist[0] = StartDist[np.triu_indices(nDL, k=1)].min()                  # Max minimum distance

# MaxAvgDist = np.zeros((nIt, 1))
# MaxAvgDist[0] = AllMaxAvgDist[0]

# MaxGeoDist = np.zeros((nIt, 1))
# MaxGeoDist[0] = AllMaxGeoDist[0]

# MaxMinDist = np.zeros((nIt, 1))
# MaxMinDist[0] = AllMaxMinDist[0]

# UTM_Ordered_MaxAvg = np.zeros((nIt, nDL, 2))
# UTM_Ordered_MaxAvg[0, :, :] = UTM_Ordered

# UTM_Ordered_MaxGeo = np.zeros((nIt, nDL, 2))
# UTM_Ordered_MaxGeo[0, :, :] = UTM_Ordered

# UTM_Ordered_MaxMin = np.zeros((nIt, nDL, 2))
# UTM_Ordered_MaxMin[0, :, :] = UTM_Ordered

# for i in range(1, nIt):
#     IdxA = np.where(np.random.randint(0, nDL) == Idx_Ordered[:, 0])[0][0]
#     IdxB = np.random.randint(0, len(SoundingIdxAll[IdxA]))
    
#     UTM_MaxAvg_Temp = UTM_Ordered_MaxAvg[i-1, :, :]
#     UTM_MaxAvg_Temp[IdxA, :] = UTM[SoundingIdxAll[IdxA][IdxB], :]

#     UTM_MaxGeo_Temp = UTM_Ordered_MaxGeo[i-1, :, :]
#     UTM_MaxGeo_Temp[IdxA, :] = UTM[SoundingIdxAll[IdxA][IdxB], :]

#     UTM_MaxMin_Temp = UTM_Ordered_MaxMin[i-1, :, :]
#     UTM_MaxMin_Temp[IdxA, :] = UTM[SoundingIdxAll[IdxA][IdxB], :]

#     NewDistAll = np.linalg.norm(UTM_MaxAvg_Temp - UTM_MaxAvg_Temp[:,None], axis=-1)
#     NewAvgDist = np.mean(NewDistAll[np.triu_indices(nDL, k=1)])         # Average euclidean distance
#     NewGeoDist = st.gmean(NewDistAll[np.triu_indices(nDL, k=1)])        # Average geometric distance 
#     NewMaxMin = NewDistAll[np.triu_indices(nDL, k=1)].min()             

#     MaxAvgDist[i] = MaxAvgDist[i-1]
#     MaxGeoDist[i] = MaxGeoDist[i-1]
#     MaxMinDist[i] = MaxMinDist[i-1]

#     UTM_Ordered_MaxAvg[i, :, :] = UTM_Ordered_MaxAvg[i-1, :, :]
#     UTM_Ordered_MaxGeo[i, :, :] = UTM_Ordered_MaxGeo[i-1, :, :]
#     UTM_Ordered_MaxMin[i, :, :] = UTM_Ordered_MaxMin[i-1, :, :]


#     Rnd = np.random.randint(0, 100) # Try to implement some kind of pseudo MCMC thing to avoid local minima
#     MC_Thresh = 99
#     if NewAvgDist >= MaxAvgDist[i]:
#         MaxAvgDist[i] = NewAvgDist
#         UTM_Ordered_MaxAvg[i, :, :] = UTM_MaxAvg_Temp

#     if NewGeoDist >= MaxGeoDist[i]:
#         MaxGeoDist[i] = NewGeoDist
#         UTM_Ordered_MaxGeo[i, :, :] = UTM_MaxGeo_Temp

#     if NewMaxMin >= MaxMinDist[i]:
#         MaxMinDist[i] = NewMaxMin
#         UTM_Ordered_MaxMin[i, :, :] = UTM_MaxMin_Temp

#     if not (NewMaxMin >= MaxMinDist[i]) and not (NewMaxMin >= MaxMinDist[i]) and (Rnd > MC_Thresh):
#         UTM_Ordered_MaxAvg[i, :, :] = UTM_MaxAvg_Temp
#         UTM_Ordered_MaxGeo[i, :, :] = UTM_MaxMin_Temp
#         UTM_Ordered_MaxMin[i, :, :] = UTM_MaxMin_Temp

#     AllMaxAvgDist[i] = NewAvgDist
#     AllMaxGeoDist[i] = NewGeoDist
#     AllMaxMinDist[i] = NewMaxMin

# # Plot Arithmetic Avg Distance at each iteration
# plt.figure('MaxArithMean')
# plt.axes(
#     xlabel = 'Iteration',
#     ylabel = 'Distance (m)',
#     title = 'Max Arithmetic Mean Distance = {:.1f}'.format(MaxAvgDist[-1][0]),
# )
# plt.plot(AllMaxAvgDist, 'b')
# plt.plot(MaxAvgDist, 'r')

# # Plot Geometric Avg Distance at each iteration
# plt.figure('MaxGeomMean')
# plt.axes(
#     xlabel = 'Iteration',
#     ylabel = 'Distance (m)',
#     title = 'Max Geometric Mean Distance = {:.1f}'.format(MaxGeoDist[-1][0]),
# )
# plt.plot(AllMaxGeoDist, 'b')
# plt.plot(MaxGeoDist, 'r')
# plt.title('Max Geometric Mean Distance = {:.1f}'.format(MaxGeoDist[-1][0]));

# # Plot Max Minimum Distance at each iteration
# plt.figure('MaxMin')
# plt.axes(
#     xlabel = 'Iteration',
#     ylabel = 'Distance (m)',
#     title = 'Max Minimum Distance = {:.1f}'.format(MaxMinDist[-1][0]),
# )
# plt.plot(AllMaxMinDist, 'b')
# plt.plot(MaxMinDist, 'r')

# #%%

# MaxAvgIdx = np.argmax(AllMaxAvgDist)
# UTM_Ordered_MaxAvg_Final = UTM_Ordered_MaxAvg[MaxAvgIdx, :, :]

# MaxGeoIdx = np.argmax(AllMaxGeoDist)
# UTM_Ordered_MaxGeo_Final = UTM_Ordered_MaxGeo[MaxGeoIdx, :, :]

# MaxMinIdx = np.argmax(AllMaxMinDist)
# UTM_Ordered_MaxMin_Final = UTM_Ordered_MaxMin[MaxMinIdx, :, :]

# # Plot Final Max Total Distance Samples
# FigNum = 'Max Artithmetic Avg Distance'
# TitleStr = DataFileName + '; Max Artithmetic Avg Distance'
# Clr = cm.get_cmap('Greys', nDL)
# favg, ax = ut.color_plot_layer(RhoCut, UTM, 6, TitleStr, RhoRng, Log, 20, FigNum)
# plt.figure(favg)
# plt.scatter(UTM_Ordered_MaxAvg_Final[:, 0], UTM_Ordered_MaxAvg_Final[:, 1], c = Clr(np.linspace(0, 1, 8)), s = 100, edgecolors = 'k', linewidths=1, marker='v')

# # Plot Final Max Total Distance Samples
# FigNum = 'Max Geometric Avg Distance'
# TitleStr = DataFileName + '; Max Geometric Avg Distance'
# Clr = cm.get_cmap('Greys', nDL)
# fgeo, ax = ut.color_plot_layer(RhoCut, UTM, 6, TitleStr, RhoRng, Log, 20, FigNum)
# plt.figure(fgeo)
# plt.scatter(UTM_Ordered_MaxGeo_Final[:, 0], UTM_Ordered_MaxGeo_Final[:, 1], c = Clr(np.linspace(0, 1, 8)), s = 100, edgecolors = 'k', linewidths=1, marker='v')


# # Plot Final Max Total Distance Samples
# FigNum = 'Max Minimum Distance'
# TitleStr = DataFileName + '; Max Minimum Distance'
# Clr = cm.get_cmap('Greys', nDL)
# fmin, ax = ut.color_plot_layer(RhoCut, UTM, 6, TitleStr, RhoRng, Log, 20, FigNum)
# plt.figure(fmin)
# plt.scatter(UTM_Ordered_MaxMin_Final[:, 0], UTM_Ordered_MaxMin_Final[:, 1], c = Clr(np.linspace(0, 1, 8)), s = 100, edgecolors = 'k', linewidths=1, marker='v')

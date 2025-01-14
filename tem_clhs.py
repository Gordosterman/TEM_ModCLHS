'''
This script takes a transient EM (TEM) model and selects a set of 1D sampling points co-located with the TEM soundings based on user criteria. It is
    intended to replicate the procedures followed in Osterman, Lesch and Bradford (under review for Computers and Geoscience)
    DOI: https://doi.org/10.1016/j.cageo.2024.105582

In its basic form, the script assumes:
        - The TEM model comes from a dense, 3D survey grid, although 2D lines are technically acceptable
        - The TEM model is formated in a .xyz file, output from AarhusWorkbench, and functions exist here to load these files (as of 2024)
            - Alternate formats may be used; however, the user will need to format their data to match the required input
        - The sampling points come from optimizing an objective function that aims to:
            1. Maximally stratify the samples across the defined feature space (resistivity model layers--or principal components of those layers--
                                                                                and the geographic coordinates of the TEM soundings)
            2. Minimize the difference between the covariance of the resistivity model and the covariance of the sampled points
            3. Select points with minimal lateral variance in the surrounding resistivity model
        - The intended outcome from optimizing the defined objective function is to
            1. Replicate the statistics of the resistivity model with the samples (objective function elements 1 and 2)
                - These first criteria satisfy the so-called Conditioned Latin Hypercube Sampling (cLHS) assumption, see Minasny and McBratney (2006),
                    DOI: https://doi.org/10.1016/j.cageo.2005.12.009
            2. Select model regions where the 3D footprint of the TEM sounding is most likely to correlate with 1D footprint of the sampling strategy
        - The output of this script are sets of sampling points, where the number of sampling points per set and the number of sets are user-defined
            - Finding a global minimum of the objective function is impractical, so it is recommended to re-run the algorithm several times and
                select the sampling points that best meet the user objectives

'''

# %% ##################################################################
# Open packages

# from locale import normalize
import os
import sys

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans as kmeans

import scipy.spatial.distance as sp

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import path
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

import cmasher as cmr
import time as tm
# import multiprocessing as mp

import copy

# print(sys.path)
fpath = os.path.join(os.path.dirname(__file__), 'utils')
datapath = os.path.join(os.path.dirname(__file__), 'Data\\')
figpath = os.path.join(os.path.dirname(__file__), 'Figures\\')
outputpath = os.path.join(os.path.dirname(__file__), 'Outputs\\')
sys.path.append(fpath)

np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

LogResCMap = mpl.colors.ListedColormap(cmr.rainforest(np.linspace(0.1, 0.9, 256)))
InclCMap = mpl.colors.ListedColormap(mpl.colormaps['viridis'](np.linspace(0.10, 1, 2)))
WCLV_CMap = mpl.colormaps['plasma']

# %% ##################################################################
#

# def SoundingColorFun(MainColor, Metric):
#     import numpy as np
#     import matplotlib.colors as cl
#     nS = np.shape(Metric)[0]
#     LnClr = np.ndarray((nS, 3))
#     if isinstance(MainColor, list) | isinstance(MainColor, tuple):
#         LnClr[0] = list(MainColor)
#     elif isinstance(MainColor, str):
#         LnClr[0] = list(cl.to_rgb(MainColor))

#     MinGray = 0.2

#     ScalMetric = (Metric - np.min(Metric))/(np.max(Metric - np.min(Metric)))*(1 - MinGray)
#     LnClr[1:] = [np.asarray([1, 1, 1]) * ScalMetric[i] for i in range(1, nS)]

#     return LnClr

# def find_in_quantile(Pts, Qts):
#     nPts = np.shape(Pts)[1]
#     nQts = np.shape(Pts)[0]
#     nQPts = np.zeros((nDL, nFS)).astype(int)
#     Qtl = np.zeros((nDL, nFS)).astype(int)
#     for i in range(nQts):
#         for j in range(nPts):
#             nQPts[i, j] = np.sum((Pts[:, j] >= Qts[i, j]) & (Pts[:, j] < Qts[i + 1, j]))

#             Idx = np.where((Pts[:, j] >= Qts[i, j]) & (Pts[:, j] < Qts[i + 1, j]))
#             Qtl[Idx, j] = i

#     # Qtl: Tells which quantile in Qts that the given point Pts is found in
#     # nQPs: Tells how many points are within given quantile
#     return Qtl, nQPts

# def find_soundings(UTM_All, UTM_Sounding, Range):
#     '''
#     # Find all points within Euclidian distance of a specified point
#     # Inputs:
#         # UTM_All: all N points to sample distance at ([N x 2] array of floats)
#         # UTM_Sounding distance is assessed from ([1 x 2] array of floats)
#         # Range: search distance (float)
#     # Outputs:
#         # Dist: Distances of m points within Range (array of floats)
#         # Idx: Indices of original point array UTM_All within search radius (array of int)
#     '''
#     Dist_All = sp.cdist(UTM_All, [UTM_Sounding], metric='euclidean').flatten()
#     Idx = np.where(Dist_All <= Range)[0]
#     Idx = Idx[np.argsort(Dist_All[Idx])]
#     Dist = Dist_All[Idx]
#     return Dist, Idx


def load_xyz_file(FilePath):
    '''
    # Loads .xyz resistivity model file exported from Aarhus Workbench (v6 or later)
    Parameters
    ----------
    FilePath : File path
        Path to .xyz resistivity model file

    Returns
    -------
    XYZ_File : Pandas dataframe
        Pandas dataframe containing all data from .xyz file
    '''

    BlankCol = 'Blank'
    file_getheader = pd.read_csv(FilePath, header=0)
    nH = len(file_getheader.loc[file_getheader[file_getheader.columns[0]].str.contains("/")])
    file_colnames = pd.read_csv(FilePath, header=nH, delim_whitespace=True)
    fhead = pd.Index.tolist(file_colnames.columns)[1:]
    fhead.append(BlankCol)
    XYZ_File = pd.read_csv(FilePath, header=nH, delim_whitespace=True)
    XYZ_File.columns = fhead
    XYZ_File.drop(BlankCol, axis=1, inplace=True)
    return XYZ_File


def get_res_model(XYZ_Data):
    '''
    # Extracts resistivity and depth information from loaded .xyz file
    Parameters
    ----------
    XYZ_Data : Input XYZ Data File loaded from load_xyz_file
        XYZ Data file exported from AarhusWorkbench (v6 or later)
    Returns
    -------
    Depth : Array of float
        Depth to bottom of resistivity model cell for each sounding (meters)
    Rho : Array of float
        Resistivity of each model cell for each sounding (Ohm meters)
    '''
    Rho_Col_Idx = [cl for cl in XYZ_Data.columns
                   if 'RHO_' in cl and 'STD' not in cl]  # Find rows associated with resistivity model
    Dep_Col_Idx = [cl for cl in XYZ_Data.columns
                   if 'DEP_BOT' in cl and 'DEP_BOT_STD' not in cl]  # Find rows associated with depth to cell bottoms

    Rho = XYZ_Data[Rho_Col_Idx].to_numpy()  # Array of resistivity model cells
    Depth = np.append(XYZ_Data[Dep_Col_Idx].to_numpy(),
                      np.inf*np.ones([np.shape(Rho)[0], 1]), 1)  # Array of depth to bottom of model cells
    return Depth, Rho


def calc_obj_func(nQPts, FS_Select, FS_All, WCLV_Select, WW):
    '''
    # Calculate the weighted cLHS objective function from selected points

    Parameters
    ----------
    nQPts : Array of int
        ###
    FS_Select : [N x M] array of floats
        The feature M space values of the N selected points.
    FS_All : [N x P] array of floats
        All feature space values.
    WCLV_Select : array of M floats
        Weighted cumulative lateral variance of selected points.
    WW : 3x1 Float array
        Weighting factors for objective function parameter.

    Returns
    -------
    ObjFunAll : 4x1 Array of floats
        All unwieghted objective function parameters and the weighted total objective function.

    '''

    nSamples = nQPts.shape[0]
    CC = np.corrcoef(FS_All.T)
    TT = np.corrcoef(FS_Select.T)
    ObjFuncA = np.abs(nQPts - 1).sum()/nSamples
    ObjFuncB = np.abs(CC - TT).sum()/nSamples
    ObjFuncC = WCLV_Select.max()/nSamples

    ObjFuncSum = np.dot(WW, np.array((ObjFuncA, ObjFuncB, ObjFuncC)))
    ObjFunAll = np.asarray([ObjFuncA, ObjFuncB, ObjFuncC, ObjFuncSum])
    return ObjFunAll


def edge_flag(UTM, Poly, EdgePolygon, EdgeRadius, EdgeQuantile, PlotHist):
    '''

    # Flags soundings outside prescribed area
    Parameters
    ----------
    UTM : Nx2 Float
        Vector of UTM coordinates for all soundings .
    Poly : Bool
        If true: Use prescribed EdgePolygon
        If false: Use EdgeRadius
    EdgePolygon : Array of float
        Series of points forming a polygon, all soundings contained w/in EdgePolygon are retained.
    EdgeRadius : Float
        Set search radius around each point.
    EdgeQuantile : Float
        DESCRIPTION.
    PlotHist : Bool
        Plots histogram of residuals and cutoff if true.

    Returns
    -------
    EdgeFlag : Vector of Bool
        If True: keep sounding
        If False: reject sounding

    '''

    nS = UTM.shape[0]  # Number of soundings
    if Poly:  # Get soundings within defined polygon
        EdgeFlag = [EdgePolygon.contains_point(i) for i in UTM]
        return EdgeFlag
    else:  # Get number of soundings w/in specified distance of each sounding
        nDist = np.zeros(nS).astype(int)  # Number of other soundings within EdgeRadius of each sounding
        EdgeFlag = np.ones(nS).astype(bool)  # Initialize all flags as True
        for i in range(nS):
            Dist = np.linalg.norm(UTM - UTM[i, :], axis=1)  # Assess distances between all points
            nDist[i] = (Dist < EdgeRadius).sum()  # Find number of points within prescribed distance

        EdgeCutoff = np.quantile(nDist, EdgeQuantile)  # Set cutoff for number of soundings required within EdgeRadius

        EdgeFlag[nDist < EdgeCutoff] = False  # Set flags for each point

    return EdgeFlag


def data_fit_flag(DataRes, DataResCutoff, Percentile, PlotHist):
    '''
    # Flags soundings with data residuals higher than given cutoff
    Parameters
    ----------
    DataRes : Float
        Vector of N data residuals for each sounding.
    DataResCutoff : Float
        Cutoff value to flag soundings.
    Percentile : Bool
        If True: DataResCutoff given as percentile
        If False: DataResCutoff given as residual value
    PlotHist : Bool
        Plots histogram of residuals and cutoff if true.

    Returns
    -------
    DataResFlag : Bool
        Vector of N flags:
        If True: keep sounding
        If False: reject sounding.

    '''

    nS = DataRes.size           # Number of soundings
    if Percentile:
        DataResCutoff = np.quantile(DataRes, DataResCutoff)     # Re-initialize DataResCutoff from quantile to Data Residual value

    DataResFlag = np.ones(nS).astype(bool)                  # Initialize all flags as True
    DataResFlag[DataRes > DataResCutoff] = False              # Set all soundings with high data residuals as false (reject)

    if PlotHist:
        # Histogram of all data residuals w/ cutoff given
        plt.figure(dpi=300, figsize=[4, 3])
        plt.axes(
            xlabel='Data Residual',
            ylabel='# Counts',
            title='Data Residual Histogram'
            )
        h = plt.hist(DataRes, 41)
        plt.plot([DataResCutoff, DataResCutoff], [0, np.max(h[0])], c='r')

        plt.show()

    return DataResFlag


def color_plot_layer(UTM, TitleStr, Clr, ClrRng, ClrMap, ClrBarTitle, Log):
    '''
    # Plots layer of resistivity model as colored scatter plot

    Parameters
    ----------
    UTM : Array of floats
        UTM coordinates of each sounding.
    TitleStr : String
        Title string for plot
    Clr : Array of floats
        Data plotted as color.
    ClrRng : Floats
        Range of colorbar.
    ClrMap : Colormap name
        Colormap to use for plotting
    ClrBarTitle : String
        Colorbar title.
    Log : Bool
        If True: Plot on log scale.
        If False: Plot on linear scale

    Returns
    -------
    None.

    '''

    plt.figure(TitleStr, dpi=300)
    plt.axes()
    plt.scatter(UTM[:, 0], UTM[:, 1], s=10, c=Clr, cmap=ClrMap)

    plt.xlabel('UTM Easting (m)')
    plt.ylabel('UTM Northing (m)')
    plt.title(TitleStr, fontsize=14)
    c = plt.colorbar(fraction=0.1)
    c.ax.set_ylabel(ClrBarTitle)
    if Log:
        Ticks, TickLabels = LogTicks(ClrRng[0], ClrRng[1], 4, 1)
        plt.clim(np.log10(Ticks[0]), np.log10(Ticks[-1]))
        c.ax.set_ylim([np.log10(Ticks[0]), np.log10(Ticks[-1])])
        c.ax.set_yticks(np.log10(Ticks))
        c.ax.set_yticklabels(TickLabels)
    plt.tight_layout()
    plt.show()


def SparseFlagSelect(UTM, Flags, RedFact):
    '''
    # Selects a subset of approximately equally spaced soundings across set of flagged soundings
    Parameters
    ----------
    UTM : Array of floats
        UTM coordinates of each sounding.
    Flags : Array of booleans
        Flags for each sounding; True = sounding available for sampling; False = sounding unavailable for sampling
    RedFact : float
        Factor controlling how much to reduce sounding set by (RedFact>1)

    Returns
    -------
    SparseIdx : Array of booleans
        Flags for each sounding indicating sparse sounding set to sample from

    '''
    nS = np.shape(UTM)[0]
    UTM_Subset = UTM[Flags]  # Get coordinates of soundings flagged for use

    # Create sparse array of LHS Parameters
    nSS = Flags.sum(0)  # Number of soundings flagged for use
    nCl = (nSS/rFact).astype(int)  # Determine a number of clusters
    KM = kmeans(n_clusters=nCl, n_init='auto').fit(UTM_Subset)  # K-means clustering on the soundings flagged for use

    Cl_Labels = KM.labels_  # Label of cluster each sounding belongs to
    MinMeanIdx = np.zeros((nCl,)).astype(int)  # Indices of soundings minimizing the mean distance to other soundings in a given cluster

    for i in range(nCl):
        Idx_Cl = np.where(Cl_Labels == i)[0]   # Find soundings withing given cluster
        Dist_All = sp.cdist(UTM_Subset[Idx_Cl],
                            UTM_Subset[Idx_Cl], metric='euclidean')  # Distances between all soundings in each cluster
        MinMeanIdx[i] = Idx_Cl[Dist_All.mean(1).argmin()]

    SparseIdx = np.zeros(nS).astype(bool)
    SparseIdx[np.where(Flags)[0][MinMeanIdx]] = True  # Sparse set sounding indices to prime simulated anealling
    return SparseIdx


def LogTicks(Min, Max, N, D):
    '''
    # Outputs a range of approximately log10-spaced values, but rounded according to user-input rules; best for log-spaced tick marks
    Parameters
    ----------
    Min : Float
        Minimum value in range
    Max : Float
        Maximum value in range
    N : Int
        Number of increments
    D : Int >=0
        Factor indicating which place to round to; 0 = round according to largest digit, 1 = round according to second largest digit
            - Example for Min = 326.2: D = 0, rounded value = 300; D = 1, rounded value = 320; D = 2, rounded value = 326 (always lower for Min)

    Returns
    -------
    TicksRound: Array of floats
        Rounded, log10-spaced array

    TicksRound: Array of strings
        String of TicksRounded
    '''

    TicksRaw = np.logspace(np.log10(Min), np.log10(Max), N, base=10)
    # TicksRaw = np.append(TicksRaw, Max)

    T = 10**int(np.floor(np.log10(Min)) - D)

    TicksRound = np.round(np.floor(Min/T)*T, -int(np.log10(T)))
    for m in TicksRaw[1:-1]:
        T = 10**int(np.floor(np.log10(m)) - D)
        TicksRound = np.append(TicksRound, np.round(np.round(m/T)*T, -int(np.log10(T))))

    T = 10**int(np.floor(np.log10(Max)) - D)
    TicksRound = np.append(TicksRound, np.round(np.ceil(Max/T)*T, -int(np.log10(T))))

    TicksLabel = []
    for m in TicksRound:
        if np.mod(m, 1) != 0:
            N = len(str(m).split('.')[1])
            FormatStr = '.' + str(N) + 'f'
            TicksLabel.append(format(m, FormatStr))
        else:
            TicksLabel.append(format(m, '.0f'))

    return TicksRound, TicksLabel


def ModCLHS(SA_Params, FS_All, QDist, WCLV_All, SubsetFlags, LockIdx):
    '''
    # Set up and execute cLHS simulated anealling to select optimal sampling points from input feature space
    # Inputs:
        # SA_Params: Dictionary of simulated annealing (SA) inputs
            # nSchedules: number M of SA schedules to perform on single run (array of M ints)
            # StartIdx: Starting indices of first sample for each schedule (array of M ints)
                # N: number of samples
            # nIter: Number of iterations in each schedule (array of M ints)
            # nRuns: Number of SA runs to perform (int)
            # nDL: Number of design levels = N (int)
            # P: SA probability of swapping worst samples instead of single random sample (float, 0<=P<=1)
            # StartTemp: SA starting temperature (float, 0<StartTemp<=1)
            # MinTemp: SA minimum temperature (### TO DEAL WITH WEIRD WARNING PYTHON IS THROWING; OTHER FIX???, float)
            # CoolRate: SA exponential cooling rate, typically keep closer around 0.9-0.99 (float)
            # WW: Weights for objective function calculations (3 floats)
        # QDist: Edge values for each N strata for Q feature space elements ((N + 1) x Q floats)
        # FS_All: Feature space parameters for entire P points in dataset (P x Q array of floats)
        # WCLV_All: All weighted cumulative lateral variance values entire dataset (array of P floats)
        # SubsetFlags: Specifies what subsets of the feature space should be sampled from (array of strings)
        # LockIdx: Indices to sample that do not change (array of 1 to M-1 ints)
    # Outputs: ### CHECK THESE!
        # ObjFuncA: First objective function parameter for each of I iterations for R runs (I x R array floats)
        # ObjFuncB: Second objective function parameter for each of I iterations for R runs (I x R array floats)
        # ObjFuncC: Third objective function parameter for each of I iterations for R runs (I x R array floats)
        # AcceptIdx: Accepted indices at each of I iterations for R runs (I x R array floats)
    '''

    # Clr = np.array([[100, 143, 255],
    #                 [120, 94, 240],
    #                 [220, 38, 127],
    #                 [254, 97, 0],
    #                 [255, 176, 0]
    #                 ])/255
    # HistClrA = [0.7, 0.7, 0.7]
    # HistClrB = [0.4, 0.4, 0.4]
    # f_temp = plt.figure(figsize=[5, 4], dpi=300)
    # plt.ion()
    # ax1 = f_temp.add_subplot(3, 2, 1)
    # plt.xlabel('PC1')
    # ax1.set_yticks([])
    # ax2 = f_temp.add_subplot(3, 2, 2)
    # plt.xlabel('PC2')
    # ax2.set_yticks([])
    # ax3 = f_temp.add_subplot(3, 2, 3)
    # plt.xlabel('PC3')
    # ax3.set_yticks([])
    # ax4 = f_temp.add_subplot(3, 2, 4)
    # plt.xlabel('PC4')
    # ax4.set_yticks([])
    # ax5 = f_temp.add_subplot(3, 2, 5)
    # plt.xlabel('X')
    # ax5.set_yticks([])
    # ax6 = f_temp.add_subplot(3, 2, 6)
    # plt.xlabel('Y')
    # ax6.set_yticks([])
    # f_temp.supylabel('Counts')
    # plt.tight_layout()

    # BinsA = np.arange(-8, 8, 0.5)
    # BinsB = np.arange(-5, 5, 0.2)

    # h1 = ax1.hist(FS_All[:, 0], BinsA, fc=HistClrA)
    # h2 = ax2.hist(FS_All[:, 1], BinsA, fc=HistClrA)
    # h3 = ax3.hist(FS_All[:, 2], BinsA, fc=HistClrA)
    # h4 = ax4.hist(FS_All[:, 3], BinsA, fc=HistClrA)
    # h5 = ax5.hist(FS_All[:, 4], BinsB, fc=HistClrA)
    # h6 = ax6.hist(FS_All[:, 5], BinsB, fc=HistClrA)

    # ax1.hist(FS_All[SubsetFlags[1], 0], BinsA, fc=HistClrB)
    # ax2.hist(FS_All[SubsetFlags[1], 1], BinsA, fc=HistClrB)
    # ax3.hist(FS_All[SubsetFlags[1], 2], BinsA, fc=HistClrB)
    # ax4.hist(FS_All[SubsetFlags[1], 3], BinsA, fc=HistClrB)
    # ax5.hist(FS_All[SubsetFlags[1], 4], BinsB, fc=HistClrB)
    # ax6.hist(FS_All[SubsetFlags[1], 5], BinsB, fc=HistClrB)
    # for n in range(QDist.shape[0]):
    #     ax1.plot([QDist[n, 0], QDist[n, 0]], [0, h1[0].max()], 'k--')
    #     ax2.plot([QDist[n, 1], QDist[n, 1]], [0, h2[0].max()], 'k--')
    #     ax3.plot([QDist[n, 2], QDist[n, 2]], [0, h3[0].max()], 'k--')
    #     ax4.plot([QDist[n, 3], QDist[n, 3]], [0, h4[0].max()], 'k--')
    #     ax5.plot([QDist[n, 4], QDist[n, 4]], [0, h5[0].max()], 'k--')
    #     ax6.plot([QDist[n, 5], QDist[n, 5]], [0, h6[0].max()], 'k--')

    nSchedules = SA_Params['nSchedules']  # DO I NEED TO REPEAT FROM ABOVE?###
    StartIdx = SA_Params['StartIdx']
    nIter = SA_Params['nIter']
    nRuns = SA_Params['nRuns']
    nDL = SA_Params['nDL']
    P = SA_Params['P']
    StartTemp = SA_Params['Ts']
    CoolRate = SA_Params['CR']

    nFS = FS_All.shape[1]

    TotalIter = np.sum(nIter)  # Total number of iterations across all schedules

    ObjFuncA = np.zeros((TotalIter, nRuns))  # Objective function A (phiA): how well stratified samples are
    ObjFuncB = np.zeros((TotalIter, nRuns))  # Objective function B (phiB): (model covariance) - (sampled model cells covariance)
    ObjFuncC = np.zeros((TotalIter, nRuns))  # Objective function C (phiC): lateral variance metric
    ObjFuncFull = np.zeros((TotalIter, nRuns))  # Total weighted objective function

    AcceptIdx = np.zeros((TotalIter, nRuns, nDL)).astype(int)  # Indices of nDL sampled points at nIter steps of the sim. ann. schedule for all nRuns
    LockIdx = np.array(LockIdx)                                 # Ensure LockIdx is numpy array
    nLock = LockIdx.size

    MinTemp = 1e-7                      # Minimum temperature (trying to avoid weird warning; not working!###)

    ScheduleIdx = np.zeros((nRuns, nSchedules, nDL)).astype(int)
    for s in range(nSchedules):
        AcceptIdx_Temp = np.zeros((nIter[s], nDL)).astype(int)
        ObjFuncA_Temp = np.zeros(nIter[s])
        ObjFuncB_Temp = np.zeros(nIter[s])
        ObjFuncC_Temp = np.zeros(nIter[s])
        ObjFuncFull_Temp = np.zeros(nIter[s])

        SampleIdx = np.where(SubsetFlags[s])[0].astype(int)  # Get indices of soundings flagged for use

        for r in range(nRuns):
            t = tm.time()   # Start time of run (time)
            nIdx = SampleIdx.shape[0]  # Number of indices in SubsetIdx
            IdxPerm = copy.copy(SampleIdx)  # Permuations of SubsetIdx; ### DO I NEED TO DO copy.copy???
            np.random.shuffle(IdxPerm)  # Shuffle IdxPerm to initialize pool of random samples to draw from

            if s == 0:
                if (np.size(StartIdx[s]) == nRuns) & (nRuns > 1):
                    IdxS = StartIdx[s, r]
                else:
                    WCLV_25q = np.quantile(WCLV_All[SampleIdx], 0.25)
                    WCLV_25q_Idx = SampleIdx[np.where(WCLV_All[SampleIdx] < WCLV_25q)[0]]  # Get 25th quantile WCLV samples
                    IdxS = np.random.choice(WCLV_25q_Idx, nDL, replace=False).astype(int)
                    if np.any(LockIdx):
                        IdxS[0:LockIdx.size] = LockIdx

            else:
                IdxS = ScheduleIdx[r, s - 1, :]

            # if np.any(StartIdx[s]):
            #     if (np.size(StartIdx[s]) == nRuns) & (nRuns > 1):
            #         IdxS = StartIdx[s, r]
            #     else:
            #         if np.size(StartIdx[s]) > 1:
            #             IdxS = StartIdx[s, 0]

            # ### NEED SOME FUNCTION TO ENSURE CONSISTENCY BETWEEN STARTING INDICES AND LOCKED INDICES
            # else:
            #     WCLV_25q = np.quantile(WCLV_All[SubsetIdx], 0.25)
            #     WCLV_25q_Idx = SubsetIdx[np.where(WCLV_All[SubsetIdx[s]] < WCLV_25q)[0]]  # Get 25th quantile WCLV samples
            #     IdxS = np.random.choice(WCLV_25q_Idx, nDL, replace=False).astype(int)
            #     if LockIdx.size > 0:
            #         IdxS[0:LockIdx.size] = LockIdx

            MtchIdx = [np.where(IdxPerm == [i])[0][0] for i in IdxS[nLock:]]  # Find indices of IdxS in IndexPerm
            IdxR = np.delete(IdxPerm, MtchIdx).astype(int)  # Remove IdxS indices from IndexPerm so indices cannot be sampled twice

            # IdxS = IdxPerm[:nDL].astype(int)  # Selected Indices to start iterations

            # AcceptIdx[0, r] = IdxS  # Initial set of accepted indices
            FS_Select = FS_All[IdxS]  # Feature space values at initial accepted indices

            # for n in range(FS_Select.shape[0]):
            #     ax1.plot([FS_Select[n, 0], FS_Select[n, 0]], [0, h1[0].max()], c=Clr[n], lw=2)
            #     ax2.plot([FS_Select[n, 1], FS_Select[n, 1]], [0, h2[0].max()], c=Clr[n], lw=2)
            #     ax3.plot([FS_Select[n, 2], FS_Select[n, 2]], [0, h3[0].max()], c=Clr[n], lw=2)
            #     ax4.plot([FS_Select[n, 3], FS_Select[n, 3]], [0, h4[0].max()], c=Clr[n], lw=2)
            #     ax5.plot([FS_Select[n, 4], FS_Select[n, 4]], [0, h5[0].max()], c=Clr[n], lw=2)
            #     ax6.plot([FS_Select[n, 5], FS_Select[n, 5]], [0, h6[0].max()], c=Clr[n], lw=2)

            nQPts = np.zeros((nDL, nFS)).astype(int)
            for i in range(nFS):
                nQPts[:, i] = np.histogram(FS_Select[:, i], bins=QDist[:, i])[0]  # Find how many samples fall in each strata given in QDist

            RandValA = np.random.rand(nIter[s])  # Uniform random value in range [0, 1] to determine random sample or worst samples are swapped
            RandValB = np.random.rand(nIter[s])  # Uniform random value in range [0, 1] to determine if proposed change to sample set is accepted

            CoolTemp = StartTemp[s]   # Initialize cooling temperature

            for k in range(nIter[s]):

                if k == 0:  # First iteration: parameters set here so initial samples always accepted
                    Metro = 1  # Initial metropolis transition probability
                    ObjDiff = 0  # No difference in objective functions between steps at this point
                    WCLV_Select = WCLV_All[IdxS]  # WCLV at the inital step
                    ObjFuncAll_Select = calc_obj_func(nQPts, FS_Select, FS_All, WCLV_Select, WW)  # Initial objective function calculation

                else:  # For all nIter steps after first

                    if RandValA[k] < P:  # Pick random sample, swap with reserve indices
                        RandSample = np.random.randint(nLock, nDL)  # Sample to be swapped
                        TempIdx = IdxS[RandSample]  # Temporarily store sample to be swapped
                        RandIdx = np.random.randint(nIdx - nDL)  # Index of sample to be swapped from reserve pool
                        IdxS[RandSample] = IdxR[RandIdx]  # Swap random reserve sample into Selected Pool
                        IdxR[RandIdx] = TempIdx  # Swap de-selected sample into reserve pool

                    else:  # Swap worst samples (samples associated w/ too many/too few quantiles)
                        # Dif = np.abs(nQPts - 1).sum(axis=1)  # Total number of over/undersampled strata for each feature space variable
                        # TotalMax = Dif.max().astype(int)
                        # MaxIdx = np.where(Dif == TotalMax)[0]  # Samples to be swapped

                        # Dif = np.abs(nQPts - 1).T  # Total number of over/undersampled strata for each feature space variable
                        # TotalMax = Dif.max().astype(int)
                        # MaxIdx = np.array(np.where(Dif == TotalMax))  # Samples to be swapped

                        # N_Strata = np.zeros(nDL).astype(int)         # ### CHECK AND COMMENT THIS SECTION ###
                        y = np.zeros((nDL, FS_All.shape[1], nDL)).astype(int)       # For all of N samples, find which quantile bin Q of all feature space components F are sampled
                        for j in range(nDL):
                            y[j, :, :] = np.array([np.histogram(FS_Select[j, i], bins=QDist[:, i])[0] for i in range(FS_All.shape[1])])
                        y_tot = y.sum(axis=0)       # Collapse y into FxQ matrix; gives number of times each quantile bin is sampled (rows all sum to number of samples)
                        N_Strata = np.abs(y_tot - 1)        # 0 = proper number of samples in given quantile bin; >0 too many/too few samples in given quantile bin
                        # MaxSamp = np.array(np.where(N_Strata == N_Strata.max()))[:, 0]  # Sampling sites that most over-sample quantile bins
                        MaxSamp = np.argwhere(N_Strata.max(0) == N_Strata.max()).flatten()  # Sampling sites that most over-sample quantile bins
                        # MaxIdx = np.array(np.where(N_Strata == N_Strata.max())).T
                        RepSampIdx = np.unique(MaxSamp)
                        # for i in MaxSamp:
                        #     # Sample_Idx = np.where(y[:, i[0], i[1]] == 1)[0]
                        #     # if np.any(Sample_Idx):
                        #     RepIdxAll.append(Sample_Idx)
                        # RepIdx = np.unique(RepIdxAll)
                        if np.any(LockIdx):
                            RepSampIdx = np.setdiff1d(RepSampIdx, np.arange(0, nLock))
                        # MaxIdx = np.where(N_Strata == N_Strata.max())[0]
                        # TempIdx = IdxS[MaxIdx]  # Temporarily store sample to be swapped

                        TempIdx = IdxS[RepSampIdx]
                        RandIdx = np.random.randint(nIdx - nDL, size=RepSampIdx.size)  # Index of sample to be swapped from reserve pool
                        IdxS[RepSampIdx] = IdxR[RandIdx]  # Swap random reserve sample into Selected Pool
                        IdxR[RandIdx] = TempIdx  # Swap de-selected sample into reserve pool

                    FS_Select = FS_All[IdxS]

                    # Calcuate Objective Function

                    nQPts = np.zeros((nDL, nFS))
                    for i in range(nFS):
                        nQPts[:, i] = np.histogram(FS_Select[:, i], bins=QDist[:, i])[0]

                    WCLV_Select = WCLV_All[IdxS]

                    ObjFuncAll_Select = calc_obj_func(nQPts, FS_Select, FS_All, WCLV_Select, WW)

                    # Perform Annealing Schedule
                    ObjDiff = ObjFuncAll_Select[3] - ObjFuncFull_Temp[k - 1]            # ObjDiff < 0: New Total ObjFunc lower (better) than previous
                    Metro = np.exp(-ObjDiff/CoolTemp) + np.random.rand()*CoolTemp

                    CoolTemp = np.max([CoolTemp*CoolRate, MinTemp])
                # Generate uniform random number between 0 and 1, compare to Metro

                Idx_Temp = k

                if (RandValB[k] < Metro or ObjDiff < 0):  # Accept new selected samples
                    AcceptIdx_Temp[Idx_Temp] = IdxS
                    ObjFuncA_Temp[Idx_Temp] = ObjFuncAll_Select[0]
                    ObjFuncB_Temp[Idx_Temp] = ObjFuncAll_Select[1]
                    ObjFuncC_Temp[Idx_Temp] = ObjFuncAll_Select[2]
                    ObjFuncFull_Temp[Idx_Temp] = ObjFuncAll_Select[3]
                else:  # Reject new selected samples; use samples from previous iteration
                    AcceptIdx_Temp[Idx_Temp] = AcceptIdx_Temp[k - 1]
                    ObjFuncA_Temp[Idx_Temp] = ObjFuncA_Temp[k - 1]
                    ObjFuncB_Temp[Idx_Temp] = ObjFuncB_Temp[k - 1]
                    ObjFuncC_Temp[Idx_Temp] = ObjFuncC_Temp[k - 1]
                    ObjFuncFull_Temp[Idx_Temp] = ObjFuncFull_Temp[k - 1]


                # Clr = np.array([[100, 143, 255],
                #                 [120, 94, 240],
                #                 [220, 38, 127],
                #                 [254, 97, 0],
                #                 [255, 176, 0]
                #                 ])/255
                # HistClrA = [0.7, 0.7, 0.7]
                # HistClrB = [0.4, 0.4, 0.4]
                # f_temp = plt.figure(figsize=[5, 4], dpi=300)
                # plt.ion()
                # ax1 = f_temp.add_subplot(3, 2, 1)
                # plt.xlabel('PC1')
                # ax1.set_yticks([])
                # ax2 = f_temp.add_subplot(3, 2, 2)
                # plt.xlabel('PC2')
                # ax2.set_yticks([])
                # ax3 = f_temp.add_subplot(3, 2, 3)
                # plt.xlabel('PC3')
                # ax3.set_yticks([])
                # ax4 = f_temp.add_subplot(3, 2, 4)
                # plt.xlabel('PC4')
                # ax4.set_yticks([])
                # ax5 = f_temp.add_subplot(3, 2, 5)
                # plt.xlabel('X')
                # ax5.set_yticks([])
                # ax6 = f_temp.add_subplot(3, 2, 6)
                # plt.xlabel('Y')
                # ax6.set_yticks([])
                # f_temp.supylabel('Counts')
                # plt.tight_layout()

                # BinsA = np.arange(-8, 8, 0.5)
                # BinsB = np.arange(-5, 5, 0.2)

                # h1 = ax1.hist(FS_All[:, 0], BinsA, fc=HistClrA)
                # h2 = ax2.hist(FS_All[:, 1], BinsA, fc=HistClrA)
                # h3 = ax3.hist(FS_All[:, 2], BinsA, fc=HistClrA)
                # h4 = ax4.hist(FS_All[:, 3], BinsA, fc=HistClrA)
                # h5 = ax5.hist(FS_All[:, 4], BinsB, fc=HistClrA)
                # h6 = ax6.hist(FS_All[:, 5], BinsB, fc=HistClrA)

                # ax1.hist(FS_All[SubsetFlags[1], 0], BinsA, fc=HistClrB)
                # ax2.hist(FS_All[SubsetFlags[1], 1], BinsA, fc=HistClrB)
                # ax3.hist(FS_All[SubsetFlags[1], 2], BinsA, fc=HistClrB)
                # ax4.hist(FS_All[SubsetFlags[1], 3], BinsA, fc=HistClrB)
                # ax5.hist(FS_All[SubsetFlags[1], 4], BinsB, fc=HistClrB)
                # ax6.hist(FS_All[SubsetFlags[1], 5], BinsB, fc=HistClrB)
                # for n in range(QDist.shape[0]):
                #     ax1.plot([QDist[n, 0], QDist[n, 0]], [0, h1[0].max()], 'k--')
                #     ax2.plot([QDist[n, 1], QDist[n, 1]], [0, h2[0].max()], 'k--')
                #     ax3.plot([QDist[n, 2], QDist[n, 2]], [0, h3[0].max()], 'k--')
                #     ax4.plot([QDist[n, 3], QDist[n, 3]], [0, h4[0].max()], 'k--')
                #     ax5.plot([QDist[n, 4], QDist[n, 4]], [0, h5[0].max()], 'k--')
                #     ax6.plot([QDist[n, 5], QDist[n, 5]], [0, h6[0].max()], 'k--')
                # for n in range(FS_Select.shape[0]):
                #     ax1.plot([FS_Select[n, 0], FS_Select[n, 0]], [0, h1[0].max()], c=Clr[n], lw=2, alpha=0.65)
                #     ax2.plot([FS_Select[n, 1], FS_Select[n, 1]], [0, h2[0].max()], c=Clr[n], lw=2, alpha=0.65)
                #     ax3.plot([FS_Select[n, 2], FS_Select[n, 2]], [0, h3[0].max()], c=Clr[n], lw=2, alpha=0.65)
                #     ax4.plot([FS_Select[n, 3], FS_Select[n, 3]], [0, h4[0].max()], c=Clr[n], lw=2, alpha=0.65)
                #     ax5.plot([FS_Select[n, 4], FS_Select[n, 4]], [0, h5[0].max()], c=Clr[n], lw=2, alpha=0.65)
                #     ax6.plot([FS_Select[n, 5], FS_Select[n, 5]], [0, h6[0].max()], c=Clr[n], lw=2, alpha=0.65)

                # print(ObjFuncAll_Select[-1])

                # plt.figure(dpi=300, figsize=[5, 3])
                # plt.axes()
                # plt.scatter(UTM[:, 0], UTM[:, 1], s=10, c=AllFlags, cmap=InclCMap)
                # plt.scatter(UTM[IdxS, 0], UTM[IdxS, 1], s=100, c=Clr, marker='v', edgecolor='k')
                # plt.xlabel('UTM Easting (m)')
                # plt.ylabel('UTM Northing (m)')
                # c = plt.colorbar(fraction=0.1)
                # c.ax.set_ylabel('Flag Value')

                # plt.tight_layout()
                # plt.show()
                # ppp=1

            if s == 0:
                sIdx = np.arange(0, nIter[s])
            else:
                sIdx = np.arange(nIter[s - 1], nIter[s] + nIter[s - 1])

            AcceptIdx[sIdx, r] = AcceptIdx_Temp
            ObjFuncA[sIdx, r] = ObjFuncA_Temp
            ObjFuncB[sIdx, r] = ObjFuncB_Temp
            ObjFuncC[sIdx, r] = ObjFuncC_Temp
            ObjFuncFull[sIdx, r] = ObjFuncFull_Temp

            ScheduleIdx[r, s, :] = AcceptIdx_Temp[-1]  # Reinitialize starting indices for next schedule
            print('Run #: ' + str(r) + '; Runtime = ' + '{:.2f}'.format(tm.time() - t))

    return ObjFuncA, ObjFuncB, ObjFuncC, AcceptIdx


# %%
# INPUT VARIABLES

# FIXED OPTIONS (only change for advanced users)
VarCalcDist_Start = 15  # Effective measurement footprint (m) (for 3x3 tTEM system, TEMcompany)
IndAngle = 30  # Induction "smoke-ring" expansion angle for TEM sounding

DataFilePath = datapath + 'tTEM Inversion XYZ Files'    # Data path is relative to script path


# USER OPTIONS

# DataFileName = 'Robson_tTEM_Inv_Original_Smooth'  # Input tTEM .xyz file name
DataFileName = 'Clarklind tTEM Inversion'  # Input tTEM .xyz file name
OutFileName = 'TEST'  # Output .txt file name
SoundingMaxDepth = 35  # Maximum sounding depth to use in feature space

nDL = 5  # Number of samples (DL=design levels)


# DEFINE BOUNDARIES OF STUDY AREA (soundings outside this region are completely excluded from the sampling design)
# Note: the UTM coordinates are always translated so the smallest X/Y points are at 0, this reduces the number of digits to plot

# BoundaryPoly = path.Path([[-10, 390], [600, 390], [600, 425], [750, 425],
#                           [950, 650], [950, 1200], [-10, 1200], [-10, 390])  # Polygon defining boundaries of study area (first pt must equal last pt)

BoundaryPoly = path.Path([[-40, -40], [1000, -40], [1000, 500], [-40, 500], [-40, -40]])  # CLARKLIND

# DEFINE PARTIAL-EXCLUSION CRITERIA (soundings that cannot be sampled but that will impact the selection of the sampled soundings)

# Edge criteria: soundings too close to the edge of the survey area should be subject to partial-exclusion
EdgePolyFlag = True
#   Option 1. Define inclusion polygon; all soudings inside polygon are included, those outside are partially excluded

# EdgePolygon = path.Path([[100, 450], [300, 410], [580, 410], [700, 470],
#                          [750, 590], [820, 625], [900, 740], [890, 1110], [100, 1060], [100, 390]])  # ROBSON FRONT 100

EdgePolygon = path.Path([[40, 40], [760, 40], [760, 360], [40, 360], [40, 40]])  # CLARKLIND


# EdgePolygon = path.Path([[75, 75], [700, 50], [700, 325], [100, 350], [75, 75]])
#   Option 2. Determine number of soundings around each sounding; if there are too few (i.e., sounding is close to the edge of the survey grid),
#             then that sounding is partially excluded; this option is best for dense, regular grids
EdgeRadius = 100  # Radius to search around soundings
EdgeQuantile = 0.35  # Remove soundings w/ # of other soundings in FlagRadius below this quantile

# Data residual criteria: partially-exclude the soundings with the highest data residuals
DataRes_Cutoff = 0.99  # Set the cutoff quantile (0-1); soundings with data residuals higher than this quantile are partially excluded


# DEFINE FEATURE SPACE
PC_Flag = True  # True (default): use principal components of resistivity model in feature space; False: use full resistivity model
nPC = 4  # number of Principal Components (PCs); adjust to so that >90% of variance is explained


# DEFINE SIMULATED ANNEALING PARAMETERS
nRuns = 100  # Number of times to repeat LHS procedure
nSchedules = 2  # Number of simulated annealing schedules to perform per LHS run
nIter = [2000, 8000]  # Number of iterations to perform during each simulated anealling schedule (size must equal nSchedules)

WA, WB, WC = 1, 2, 75  # Weighting factors for objective function calculation
WW = [WA, WB, WC]

P = 0.10  # Probability of swapping random sample instead of worst
CoolRate = 0.95  # Simulated annealing cooling rate

StartTemp = [1, CoolRate**nIter[0]]  # Simulated anealling starting temperature, first schedule (size must equal nSchedules)
StartIdx = [[], -1]   # Specify starting indices for each run (size must equal number of runs)
                        # []: Start at random location
                        # -1: Start at final location of previous SA schedule
                        # Otherwise, specify sampling indices (# = nDL int)

Subsets = ['Sparse', 'Flags']  # Subsets of the feature space to sample from (size must equal nSchedules)
                                   # 'Sparse': Reduce the flagged feature space by a reduction factor; helps prime the simulated annealing
                                   # 'Flags': Only sample from flagged samples
rFact = 10  # For sparse sampling, specify the reduction factor

# LockIdx = [300, 100, 200]
LockIdx = []   # Indices of soundings to ALWAYS sample from (cannot be changed during simulated annealing);
                   # Length <= nDL; if empty, no sample indices are locked (default)

# %%
# LOAD DATA
XYZ_Inv = load_xyz_file(DataFilePath + '\\' + DataFileName + '.xyz')

# %% ##################################################################
# ORGANIZE DATA

UTM_All = np.transpose([XYZ_Inv.UTMX.values - XYZ_Inv.UTMX.values.min(),
                        XYZ_Inv.UTMY.values - XYZ_Inv.UTMY.values.min()])

DOI_Con_All = np.asarray(XYZ_Inv.DOI_CONSERVATIVE)

# Extract resistivity (rho), depth, elevation and data residual arrays
Depth_All, Rho_All = get_res_model(XYZ_Inv)
Elev_All = XYZ_Inv.ELEVATION.to_numpy()
DataRes_All = XYZ_Inv.RESDATA.to_numpy()

# Set flags to remove soundings outside of BoundaryPoly
BoundaryFlag = [BoundaryPoly.contains_point(i) for i in UTM_All]

# Re-initialize variables with only included soundings
UTM = UTM_All[BoundaryFlag]
Depth = Depth_All[BoundaryFlag]
Rho = Rho_All[BoundaryFlag]
Elev = Elev_All[BoundaryFlag]
DataRes = DataRes_All[BoundaryFlag]

LogRho = np.log10(Rho)    # Convert resistivity to log10 resistivity

DepthCut = Depth[:, Depth[0] <= SoundingMaxDepth]
RhoCut = Rho[:, np.where(Depth[0] <= SoundingMaxDepth)[0]]
LogRhoCut = np.log10(RhoCut)

nS = np.shape(UTM)[0]  # Number of soundings (points)
nL = np.shape(DepthCut)[1]  # Number of resistivity model layers

LayerNo = 3  # Select layer of resistivity model to plot
color_plot_layer(UTM, 'Resistivity Model, Layer: ' + str(LayerNo), LogRho[:, LayerNo],
                 np.quantile(RhoCut, (0.02, 0.98)), LogResCMap, 'Log10 Resistivity', True)

# %% ##################################################################
# SET UP INCLUSION FLAGS
# Un-flag soundings 1. too close to edge of survey area, and 2. that have high data residuals

EdgeFlag = edge_flag(UTM, EdgePolyFlag, EdgePolygon, EdgeRadius, EdgeQuantile, False)

DataResFlag = data_fit_flag(DataRes, DataRes_Cutoff, True, False)

AllFlags = EdgeFlag * DataResFlag  # True: include sounding, False: partial-exclusion

color_plot_layer(UTM, 'All Flags', AllFlags, [0, 1], InclCMap,
                 'Flag Value (1=keep; 0=reject)', False)

# %% ##################################################################
# ASSESS LATERAL TRANSITION ZONES
# Plan: Find all soundings in radius around each sounding, assess total variance
#       Plot histogram of variances, define cutoff to exclude samples
#       NOTE: THIS ASSUMES EFFECTIVELY FLAT TERRAIN!

TanIndAng = np.tan(IndAngle*np.pi/180)
VarCalcDist = [VarCalcDist_Start + y*TanIndAng for y in DepthCut[0, :]]  # Search radius at each model depth interval

LogVarAll = np.zeros((nS, nL))
for i in range(nS):
    Dist = np.linalg.norm(UTM - UTM[i], axis=1)  # Assess distance between all points
    LogVarAll[i, :] = [LogRhoCut[Dist < VarCalcDist[j], j].var() for j in range(nL)]  # variance of all points within radius at each depth interval

ThickCut = np.append(np.zeros((nS, 1)), np.diff(DepthCut), axis=1)  # Translate depths to interval bottom to thicknesses

WCLV = (LogVarAll * ThickCut).sum(axis=1)  # Weighted cumulative lateral variance metric for all soundings

color_plot_layer(UTM, '', WCLV, [0, 1], WCLV_CMap, r'$V_L$', False)


# %% ##################################################################
# BUILD THE FEATURE SPACE
# The feature space defines the parameters that will be sampled, in our case the resistivity model and the geographic coordinates of the model
#   - Either the principal components that explain 90+% of the log-resistivity model variance or the log-resistivity model are used
#   - The UTM coordinates must be normalized to the scale of either the PCs or the log-resistivity model for the feature space covariance calculation

if PC_Flag:
    # PRINCIPAL COMPONENT ANALYSIS (PCA)
    # Use principal component analysis to reduce multi-layer resistivity model to smaller number of orthogonal principal components
    pca = PCA(n_components=nPC)  # Set up PCA
    standscal = StandardScaler()  # Process to normalize variables to mean = 0, std = 1

    LR_SC = standscal.fit_transform(LogRhoCut)  # Run model through standard scaler before PCA to normalize

    PC_Score = pca.fit_transform(LR_SC)  # PC scores

    # print('Mean = ' + str(np.mean(PC_Score).round(3)))
    # print('Std = ' + str(np.var(PC_Score).round(3)))
    # print('Eigenvectors, or PC Loadings (a_k) = ' + str(pca.components_))
    # print('Eigenvalues (lambda) = ' + str(pca.explained_variance_))
    print('Explained Variance Ratios (pi_j)= ' + str(pca.explained_variance_ratio_))
    ExpVar = pca.explained_variance_ratio_.sum(axis=0).round(3) * 100
    print('Total explained variance for ' + str(nPC) + ' PCs = ' + str(ExpVar) + '%')

    MaxAx = (UTM.max(0) - UTM.min(0)).argmax()  # Get coordinate axis with maximum range
    UTM_Normalized = (UTM - UTM.mean(0))/UTM[:, MaxAx].std(0)*PC_Score.std()  # UTM Coordinates normalized so max axis has same variance as PC Scores

    FeatureSpace = np.concatenate((PC_Score, UTM_Normalized), axis=1)  # Build full feature space from PC scores and Normalized coordinates

else:
    # Use the original log-resistivity model to define the feature space
    MaxAx = (UTM.max(0) - UTM.min(0)).argmax()  # Get coordinate axis with maximum range
    UTM_Normalized = (UTM - UTM.mean(0))/UTM[:, MaxAx].std(0)*LogRhoCut.std()  # UTM Coordinates normalized so max axis has same variance as LogRho

    FeatureSpace = np.concatenate((LogRhoCut, UTM_Normalized), axis=1)  # Build full feature space from PC scores and Normalized coordinates

nFS = FeatureSpace.shape[1]  # Number of feature space components

Q = np.arange(0, 1.001, 1/nDL)  # Quantiles of the feature space
QDist = np.quantile(FeatureSpace, Q, axis=0)  # Quantile bin bounds for each feature space component
QDist[-1] += 0.01  # May improve stability? Can't remember how though... ###


# %%
# RUN SIMULATED ANNEALING SCHEDULE(S)

# 'nSchedule': Number of types of simulated annealing schedules to perform per run (changing SA parameters)
# 'StartIdx': Specify starting indices for each run (# entries must =1 or nSteps)
#           []: Start at random location
#           -1: Start at final location of previous SA schedule
#           Otherwise, specify sampling indices (# = nDL int)
# 'nIter': Number of iterations for each schedule (Single entry applies to all schedules, otherwise specify all)
# 'nRuns': Number of times to repeat entire set of simulated annealing schedules (one int)
# 'nDL': Number of samples (design levels, one int)
# 'P': # Probability of swapping random sample instead of worst sample(s)
# 'CR': Cooling rate (Single entry applies to all schedules)
# 'Ts': Starting temperature (Single entry applies to all schedules, otherwise specify all)

SA_Params = {
    'nSchedules': nSchedules,
    'StartIdx': StartIdx,
    'nIter': nIter,
    'nRuns': nRuns,
    'nDL': nDL,
    'P': P,
    'CR': CoolRate,
    'Ts': StartTemp,
    'WW': WW,
    }

SubsetFlags = []
for i in Subsets:
    if i.lower() == 'sparse':
        SparseFlags = SparseFlagSelect(UTM, AllFlags, rFact)
        SubsetFlags = np.append(SubsetFlags, SparseFlags).astype(bool)
    elif i.lower() == 'flags':
        SubsetFlags = np.append(SubsetFlags, AllFlags).astype(bool)
    else:
        SubsetFlags = np.append(SubsetFlags, np.ones(nS)).astype(bool)
SubsetFlags = SubsetFlags.reshape((nSchedules, nS))

T = tm.time()
SA_RunOutput = ModCLHS(SA_Params, FeatureSpace, QDist, WCLV, SubsetFlags, LockIdx)
print('Total LHS Time = ' + '{:.2f}'.format(tm.time() - T))

ObjFuncA_Run, ObjFuncB_Run, ObjFuncC_Run, AcceptIdx_Run = SA_RunOutput      # Objective functions and Accepted indices from each run

# Calculate weighted objective functions
ObjFuncA = ObjFuncA_Run*WA
ObjFuncB = ObjFuncB_Run*WB
ObjFuncC = ObjFuncC_Run*WC
ObjFuncFull = ObjFuncA + ObjFuncB + ObjFuncC


# %%
#

Idx_Plot = 1  # SA run to use (0 = lowest (best) ObjFun value; -1 (last index) = highest ObjFun value)


BestRunIdx = ObjFuncFull[-1].argsort()[Idx_Plot]  # Find lowest final objective function iteration value from all nRuns
print(ObjFuncFull[-1][BestRunIdx])
AcceptIdx_Final = AcceptIdx_Run[-1, BestRunIdx]  # Indices of resistivity model soundings associated with the lowest objective function value
FS_Final = FeatureSpace[AcceptIdx_Final]  # The feature space scores associated with the lowest objective function value


Clr = np.array([[100, 143, 255],
                [120, 94, 240],
                [220, 38, 127],
                [254, 97, 0],
                [255, 176, 0]
                ])/255
HistClrA = [0.7, 0.7, 0.7]
HistClrB = [0.4, 0.4, 0.4]
f_temp = plt.figure(figsize=[5, 4], dpi=300)
plt.ion()
ax1 = f_temp.add_subplot(3, 2, 1)
plt.xlabel('PC1')
ax1.set_yticks([])
ax2 = f_temp.add_subplot(3, 2, 2)
plt.xlabel('PC2')
ax2.set_yticks([])
ax3 = f_temp.add_subplot(3, 2, 3)
plt.xlabel('PC3')
ax3.set_yticks([])
ax4 = f_temp.add_subplot(3, 2, 4)
plt.xlabel('PC4')
ax4.set_yticks([])
ax5 = f_temp.add_subplot(3, 2, 5)
plt.xlabel('X')
ax5.set_yticks([])
ax6 = f_temp.add_subplot(3, 2, 6)
plt.xlabel('Y')
ax6.set_yticks([])
f_temp.supylabel('Counts')
plt.tight_layout()

BinsA = np.arange(-8, 8, 0.5)
BinsB = np.arange(-5, 5, 0.2)

h1 = ax1.hist(FeatureSpace[:, 0], BinsA, fc=HistClrA)
h2 = ax2.hist(FeatureSpace[:, 1], BinsA, fc=HistClrA)
h3 = ax3.hist(FeatureSpace[:, 2], BinsA, fc=HistClrA)
h4 = ax4.hist(FeatureSpace[:, 3], BinsA, fc=HistClrA)
h5 = ax5.hist(FeatureSpace[:, 4], BinsB, fc=HistClrA)
h6 = ax6.hist(FeatureSpace[:, 5], BinsB, fc=HistClrA)

ax1.hist(FeatureSpace[SubsetFlags[1], 0], BinsA, fc=HistClrB)
ax2.hist(FeatureSpace[SubsetFlags[1], 1], BinsA, fc=HistClrB)
ax3.hist(FeatureSpace[SubsetFlags[1], 2], BinsA, fc=HistClrB)
ax4.hist(FeatureSpace[SubsetFlags[1], 3], BinsA, fc=HistClrB)
ax5.hist(FeatureSpace[SubsetFlags[1], 4], BinsB, fc=HistClrB)
ax6.hist(FeatureSpace[SubsetFlags[1], 5], BinsB, fc=HistClrB)
for n in range(QDist.shape[0]):
    ax1.plot([QDist[n, 0], QDist[n, 0]], [0, h1[0].max()], 'k--')
    ax2.plot([QDist[n, 1], QDist[n, 1]], [0, h2[0].max()], 'k--')
    ax3.plot([QDist[n, 2], QDist[n, 2]], [0, h3[0].max()], 'k--')
    ax4.plot([QDist[n, 3], QDist[n, 3]], [0, h4[0].max()], 'k--')
    ax5.plot([QDist[n, 4], QDist[n, 4]], [0, h5[0].max()], 'k--')
    ax6.plot([QDist[n, 5], QDist[n, 5]], [0, h6[0].max()], 'k--')

for n in range(FS_Final.shape[0]):
    ax1.plot([FS_Final[n, 0], FS_Final[n, 0]], [0, h1[0].max()], c=Clr[n], lw=2, alpha=0.65)
    ax2.plot([FS_Final[n, 1], FS_Final[n, 1]], [0, h2[0].max()], c=Clr[n], lw=2, alpha=0.65)
    ax3.plot([FS_Final[n, 2], FS_Final[n, 2]], [0, h3[0].max()], c=Clr[n], lw=2, alpha=0.65)
    ax4.plot([FS_Final[n, 3], FS_Final[n, 3]], [0, h4[0].max()], c=Clr[n], lw=2, alpha=0.65)
    ax5.plot([FS_Final[n, 4], FS_Final[n, 4]], [0, h5[0].max()], c=Clr[n], lw=2, alpha=0.65)
    ax6.plot([FS_Final[n, 5], FS_Final[n, 5]], [0, h6[0].max()], c=Clr[n], lw=2, alpha=0.65)

plt.figure(dpi=300, figsize=[5, 3])
plt.axes()
plt.scatter(UTM[:, 0], UTM[:, 1], s=10, c=AllFlags, cmap=InclCMap)
plt.scatter(UTM[AcceptIdx_Final, 0], UTM[AcceptIdx_Final, 1], s=100, c=Clr, marker='v', edgecolor='k')
plt.xlabel('UTM Easting (m)')
plt.ylabel('UTM Northing (m)')
c = plt.colorbar(fraction=0.1)
c.ax.set_ylabel('Flag Value')

plt.tight_layout()
plt.show()

# %%
# PLOTTING FUNCTIONALITY TO OBSERVE OBJECTIVE FUNCTION BEHAVIOR

Idx_Plot = 5  # SA run to use (0 = lowest (best) ObjFun value; -1 (last index) = highest ObjFun value)

BestRunIdx = ObjFuncFull[-1].argsort()[Idx_Plot]  # Find lowest final objective function iteration value from all nRuns

# AcceptIdx_Final = AcceptIdx_Run[-1, BestRunIdx]  # Indices of resistivity model soundings associated with the lowest objective function value
LHS_Final = FeatureSpace[AcceptIdx_Final]  # The feature space scores associated with the lowest objective function value

# Plot the evolution of the lowest final objective function

f = plt.figure(figsize=(3.5, 2), dpi=300)
ax = plt.axes(
    xlabel='Iteration',
    ylabel=r'$\phi$ (-)',
)

plt.rc('axes', labelsize=12)
plt.plot(ObjFuncA[:, BestRunIdx], 'b--', linewidth=1.3, label=r'$W_1\phi_1$')
plt.plot(ObjFuncB[:, BestRunIdx], 'r-.', linewidth=1.3, label=r'$W_2\phi_2$')
plt.plot(ObjFuncC[:, BestRunIdx], 'm:', linewidth=1.3, label=r'$W_3\phi_3$')
plt.plot(ObjFuncFull[:, BestRunIdx], 'k-', linewidth=2, label=r'$\phi_T$')
plt.legend(loc='upper right', framealpha=1, edgecolor='k', fontsize=10, ncol=2)
ax.tick_params(axis='both', labelsize=10)
ax.set_xlim([-10, np.sum(nIter)])
ax.set_ylim([0, 15])
plt.tight_layout()


print('Lateral Variance Percentiles')
for i in range(nDL):
    WCLV_Quantile = np.where(np.sort(WCLV) <
                             WCLV[AcceptIdx_Final][i])[0].size/WCLV.size

    print('Sample ' + str(i + 1) + ': ' + f'{WCLV_Quantile*100:.1f}' + '%')
    print(' ')

SoundingIdxLHS = AcceptIdx_Run[-1]


# Plot histograms of the feature space parameters (all and subsets), the strata limits, and the sampled points

if PC_Flag:
    SamplingParams = np.concatenate((PC_Score, UTM_Normalized), axis=1)
else:
    SamplingParams = np.concatenate((LogRho, UTM_Normalized), axis=1)

AbsMax = np.max([FeatureSpace.max(), np.abs(FeatureSpace.max())])
Edges = np.arange(-AbsMax, AbsMax, 0.4)

sb = [321, 323, 325, 322, 324]
q = np.linspace(0, 1, nDL + 1)
Qtl = np.quantile(SamplingParams, q, axis=0)

fL1 = plt.figure('LHS Example', figsize=[5, 1.5*nDL], dpi=300)

WghtA = np.ones((nS, 1))/nS
WghtB = np.ones((int(AllFlags.sum()), 1))/AllFlags.sum()

CMap = plt.colormaps['tab20c']

nSub = SamplingParams.shape[1]
for j in range(nFS):
    ax = plt.subplot(nSub*100 + 11 + j)
    plt.hist(SamplingParams[:, j], Edges, weights=WghtA, fc=[0.7, 0.7, 0.7])
    plt.hist(SamplingParams[np.where(AllFlags)[0], j], Edges, weights=WghtB, fc='y')

    if j < nSub - 2:
        if PC_Flag:
            plt.title(f'PC {j+1:02d} Scores')
        else:
            plt.title(f'Model Layer {j:02d}')
    elif j == nSub - 2:
        plt.title('Scaled UTM X')
    elif j == nSub - 1:
        plt.title('Scaled UTM Y')
        plt.xlabel('Feature Space Score (-)', fontsize=10)

    YMax = ax.get_ylim()[1]
    for i in Qtl[:, j]:
        plt.plot([i, i], [0, YMax], 'k--', linewidth=2)

    for i in range(nDL):
        X = [LHS_Final[i, j], LHS_Final[i, j]]
        Y = [0, YMax]
        plt.plot(X, Y, c=CMap(i))


fL1.tight_layout()

h1 = Rectangle((0, 0), 1, 1, color=[0.7, 0.7, 0.7])
h2 = Rectangle((0, 0), 1, 1, color='y')
h3 = Line2D([0, 0], [0, 0], linewidth=2, linestyle='--', color='k')
l1 = 'All soundings'
l2 = 'Accepted soundings'
l3 = 'Latin Hypercube Strata'
ax.legend(handles=[h1, h2, h3], labels=[l1, l2, l3],
          loc='upper center', bbox_to_anchor=(0.5, -0.5),
          ncol=1, framealpha=1, edgecolor='None', fontsize=10)


# Plot the statistics of the entire model and sampled soundings at each depth interval

DepthVect = np.concatenate(([0], DepthCut[0]))

DataMean = np.mean(LogRhoCut.T, axis=1)
DataMeanDL = np.mean(LogRhoCut.T[:, AcceptIdx_Final], axis=1)
RhoVect_Mean = np.concatenate((DataMean, [DataMean[-1]]))
RhoVect_MeanDL = np.concatenate((DataMeanDL, [DataMeanDL[-1]]))

DataStd = np.std(LogRhoCut.T, axis=1)
DataStdDL = np.std(LogRhoCut.T[:, AcceptIdx_Final], axis=1)
RhoVect_Std = np.concatenate((DataStd, [DataStd[-1]]))
RhoVect_StdDL = np.concatenate((DataStdDL, [DataStdDL[-1]]))


f = plt.figure(figsize=(2.5, 4), dpi=300)
ax = plt.axes()

ax.set_ylabel('Depth (m)')
ax.set_xlabel(r'Resistivity $(\Omega$m)')

ax.set_xlim([np.log10(5), np.log10(40)])
ax.set_xticks([np.log10(5), np.log10(10), np.log10(20), np.log10(40)])
ax.set_xticklabels(['5', '10', '20', '40'])
ax.set_ylim([0, 31.5])
ax.tick_params(axis='both', labelsize=8)

plt.step(RhoVect_Mean, DepthVect, where='pre', color='k',
         linestyle='-', linewidth=2, label='Mean All')
plt.step(RhoVect_MeanDL, DepthVect, where='pre', color='r',
         linestyle='-', linewidth=2, label='Mean DL')

plt.step(RhoVect_Mean + RhoVect_Std, DepthVect, where='pre', color='k',
         linestyle='-.', linewidth=1.25, label='Std All')
plt.step(RhoVect_MeanDL + RhoVect_StdDL, DepthVect, where='pre', color='r',
         linestyle='-.', linewidth=1.25, label='Std DL')

plt.step(RhoVect_Mean - RhoVect_Std, DepthVect, where='pre', color='k',
         linestyle='-.', linewidth=1.25)
plt.step(RhoVect_MeanDL - RhoVect_StdDL, DepthVect, where='pre', color='r',
         linestyle='-.', linewidth=1.25)

ax.invert_yaxis()

LabelText = str(bytes([bytes('a', 'utf-8')[0] + i + 1]))[2] + '.'
plt.xlabel(r'Resistivity $(\Omega)$ m')
plt.ylabel(r'Depth (m)')
# plt.title(TitleStr[i], fontsize=10)

h1 = Line2D([0, 0], [0, 0], linewidth=2, linestyle='-', color='k')
h2 = Line2D([0, 0], [0, 0], linewidth=2, linestyle='--', color='k')
h3 = Line2D([0, 0], [0, 0], linewidth=2, linestyle='-', color='r')
h4 = Line2D([0, 0], [0, 0], linewidth=2, linestyle='--', color='r')


l1 = r'Model Mean'
l2 = r'Model $\pm1$ St. Dev.'
l3 = r'cLHS Mean'
l4 = r'cLHS $\pm1$ St. Dev.'
ax.legend(handles=[h1, h2, h3, h4], labels=[l1, l2, l3, l4],
          loc='upper center', bbox_to_anchor=(.5, -.15), ncol=2,
          framealpha=1, edgecolor='None', fontsize=10)

#  Plot locations of selected sampling points

plt.figure(dpi=300)
plt.axes()
plt.scatter(UTM[:, 0], UTM[:, 1], s=10, c=AllFlags, cmap=InclCMap)
plt.scatter(UTM[AcceptIdx_Final, 0], UTM[AcceptIdx_Final, 1], s=100, c='r', marker='v')
plt.xlabel('UTM Easting (m)')
plt.ylabel('UTM Northing (m)')
c = plt.colorbar(fraction=0.1)
c.ax.set_ylabel('Flag Value')

plt.tight_layout()
plt.show()
# %%
# SAVE RESULTS

# First, save all simulated annealing inputs to dataframe
# Format: [UTMX, UTMY, All_Flags, Feature Space inputs]
ColNames = ['UTMX (m)', 'UTMY (m)', 'InclusionFlags']  # Set up column names for dataframe
if PC_Flag:
    [ColNames.append('PC' + str('{:02d}'.format(i))) for i in range(nPC)]
else:
    [ColNames.append('ResLayer' + str('{:02d}'.format(i))) for i in range(nL)]
ColNames.extend(['NormUTMX', 'NormUTMY'])

UTM_Corr = UTM + np.array([XYZ_Inv.UTMX.values.min(), XYZ_Inv.UTMY.values.min()])  # Use original UTM coordinates
SimAnn_Inputs = np.concatenate((UTM_Corr, np.reshape(AllFlags.astype(int), (-1, 1)),
                                FeatureSpace.round(4)), axis=1)  # Set up dataframe inputs

SimAnn_DF = pd.DataFrame(SimAnn_Inputs, columns=ColNames)
SA_Input_Ext = '_SimAnnInputs.csv'
SimAnn_DF.to_csv(outputpath + OutFileName + SA_Input_Ext)


# Output Format: [AcceptIdx(1) ... AcceptIdx(nDL), ObjFuncA, ObjFuncB, ObjFuncC, ObjFuncAll)],
#       where all values are output from the last LHS iteration
SimAnnResults = np.concatenate((SoundingIdxLHS.astype(int), WA*ObjFuncA[-1].reshape((nRuns, 1)),
                                WB*ObjFuncB[-1].reshape((nRuns, 1)), WC*ObjFuncC[-1].reshape((nRuns, 1)),
                                ObjFuncFull[-1].reshape((nRuns, 1))), axis=1)
ColNames = []
[ColNames.append('Sounding' + str('{:02d}'.format(i)) + 'Idx') for i in range(nDL)]
ColNames.extend(['WghtObjA', 'WghtObjB', 'WghtObjC', 'TotalObj'])
SimAnnResults_DF = pd.DataFrame(SimAnnResults, columns=ColNames)
SA_Results_Ext = '_SimAnnResults.csv'
SimAnnResults_DF.to_csv(outputpath + OutFileName + SA_Results_Ext)

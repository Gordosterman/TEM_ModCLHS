# Flag soundings based on various criteria
# TO DO:
#   Comment up code
#   

import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance as sp

def edge_flag(UTM, EdgeRadius, EdgeQuantile, PlotHist):

    nS = np.shape(UTM)[0]
    EdgeFlag = np.ones(nS).astype(int)       # 1 = use sounding, 2 = do not use sounding

    # Get number of soundings w/in specified distance of each sounding
    nDist = np.zeros(nS)
    for i in range(nS):
        Pt = UTM[i, :]
        Dist = sp.cdist(UTM, [Pt], metric='euclidean')
        nDist[i] = np.sum(Dist<EdgeRadius)

    EdgeCutoff = np.quantile(nDist, EdgeQuantile)
    
    if PlotHist:
        # Histogram of total soundings w/in specified radius
        # Edges = np.arange(np.min(nDist), np.max(nDist))
        plt.figure('Distance_Histogram')
        plt.axes(
            xlabel = '# of Soundings',
            ylabel = '# Counts',
            title = '# of Soundings w/in Offset Distance ' + str(EdgeRadius) + 'm'
            )
        h = plt.hist(nDist, 41)
        plt.plot([EdgeCutoff, EdgeCutoff], [0, np.max(h[0])], c='r')

        plt.show()

    EdgeFlag[nDist < EdgeCutoff] = 0

    return EdgeFlag

def data_fit_flag(DataRes, DataResCutoff, PlotHist):

    nS = np.shape(DataRes)[0]
    DataResCutoff = np.quantile(DataRes, DataResCutoff)
    DataResFlag = np.ones(nS).astype(int)
    DataResFlag[DataRes>DataResCutoff] = 0

    if PlotHist:
        # Histogram of total soundings w/in specified radius
        # Edges = np.arange(np.min(nDist), np.max(nDist))
        plt.figure('Distance_Histogram')
        plt.axes(
            xlabel = 'Data Residual',
            ylabel = '# Counts',
            title = 'Data Residual Histogram'
            )
        h = plt.hist(DataRes,41)
        plt.plot([DataResCutoff, DataResCutoff], [0, np.max(h[0])], c='r')

        plt.show()

    return DataResFlag

def lateral_variance_flag():
    return 9
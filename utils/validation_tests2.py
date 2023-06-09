## COMMENT CODE



from matplotlib.pyplot import xlim


def validation_hist(Data, Depth, Idx, HistPlotParams):
    import numpy as np
    import utils as ut
    
    f, ax = ut.sounding_hist(Data, Depth, HistPlotParams)

    nS = len(Idx)

    DepthVect = np.concatenate(([0], Depth))
    for i in range(nS):
        SoundingIdx = Idx[i]
        RhoVect = np.concatenate((Data[:, SoundingIdx], [Data[-1, SoundingIdx]]))
        ax.step(RhoVect, DepthVect, where='pre', color='r', linewidth=1, linestyle='dashed')

    return f, ax


def validation_charstats(Data, Depth, Idx, Stats, StatsPlotParams):
    import numpy as np
    import matplotlib.pyplot as plt
    


    Title = StatsPlotParams['Title']
    DepthVect = np.concatenate(([0], Depth))

    f = plt.figure(Title)
    ax = plt.axes(
        title = StatsPlotParams['Title'],
        xlabel = StatsPlotParams['XLabel'],
        ylabel = StatsPlotParams['XLabel'],
        xlim = StatsPlotParams['XLim'],
        ylim = StatsPlotParams['YLim'],
        )

    Stats = [string.lower() for string in Stats]

    
    DataMean = np.mean(Data, axis=1)
    DataMeanDL = np.mean(Data[:, Idx], axis=1)
    RhoVect_Mean = np.concatenate((DataMean, [DataMean[-1]]))
    RhoVect_MeanDL = np.concatenate((DataMeanDL, [DataMeanDL[-1]]))
    if 'mean' in Stats:
        ax.step(RhoVect_Mean, DepthVect, where='pre', color='r', linestyle='-', linewidth=2, label='Mean All')
        ax.step(RhoVect_MeanDL, DepthVect, where='pre', color='b', linestyle='-', linewidth=2, label='Mean DL')

    if 'median' in Stats:
        DataMedian = np.median(Data, axis=1)
        DataMedianDL = np.median(Data[:, Idx], axis=1)
        RhoVect_Median = np.concatenate((DataMedian, [DataMedian[-1]]))
        RhoVect_MedianDL = np.concatenate((DataMedianDL, [DataMedianDL[-1]]))
        ax.step(RhoVect_Median, DepthVect, where='pre', color='r', linestyle='--', linewidth=2, label = 'Med All')
        ax.step(RhoVect_MedianDL, DepthVect, where='pre', color='b', linestyle='--', linewidth=2, label = 'Med DL')

    if 'std' in Stats:
        DataStd = np.std(Data, axis=1)
        DataStdDL = np.std(Data[:, Idx], axis=1)
        RhoVect_Std = np.concatenate((DataStd, [DataStd[-1]]))
        RhoVect_StdDL = np.concatenate((DataStdDL, [DataStdDL[-1]]))
        ax.step(RhoVect_Mean + RhoVect_Std, DepthVect, where='pre', color='r', linestyle='-.', linewidth=2, label = 'Std All')
        ax.step(RhoVect_MeanDL + RhoVect_StdDL, DepthVect, where='pre', color='b', linestyle='-.', linewidth=2, label = 'Std DL')

        ax.step(RhoVect_Mean - RhoVect_Std, DepthVect, where='pre', color='r', linestyle='-.', linewidth=2)
        ax.step(RhoVect_MeanDL - RhoVect_StdDL, DepthVect, where='pre', color='b', linestyle='-.', linewidth=2)

    if 'range' in Stats:
        DataMin = np.quantile(Data, 0.05, axis=1)
        DataMinDL = np.min(Data[:, Idx], axis=1)
        DataMax = np.quantile(Data, 0.95, axis=1)
        DataMaxDL = np.max(Data[:, Idx], axis=1)

        RhoVect_Min = np.concatenate((DataMin, [DataMin[-1]]))
        RhoVect_MinDL = np.concatenate((DataMinDL, [DataMinDL[-1]]))
        RhoVect_Max = np.concatenate((DataMax, [DataMax[-1]]))
        RhoVect_MaxDL = np.concatenate((DataMaxDL, [DataMaxDL[-1]]))

        ax.step(RhoVect_Min, DepthVect, where='pre', color='r', linestyle=':', linewidth=2, label='Max/Min')
        ax.step(RhoVect_MinDL, DepthVect, where='pre', color='b', linestyle=':', linewidth=2)

        ax.step(RhoVect_Max, DepthVect, where='pre', color='r', linestyle=':', linewidth=2)
        ax.step(RhoVect_MaxDL, DepthVect, where='pre', color='b', linestyle=':', linewidth=2)

    ax.invert_yaxis()
    ax.legend()

    return f, ax  

def validation_metrics(Data, UTM, Idx):
    import numpy as np
    import scipy.stats.mstats as st

    # Total abs deviation of means
    # Mean abs deviation of std
    # Total spanned range
    # Geometric Offset
    DataDL = Data[:, Idx]
    MinData = np.quantile(Data, 0.00, axis=1)
    MaxData = np.quantile(Data, 1, axis=1)
    

    MeanDev = np.sum(np.abs(np.mean(Data, axis=1) - np.mean(DataDL, axis=1)))
    StdDev = np.sum(np.abs(np.var(Data, axis=1) - np.var(DataDL, axis=1)))
    RangePct = np.mean(np.divide(np.max(DataDL, axis=1) - np.min(DataDL, axis=1),(MaxData - MinData)))

    AllDist = np.linalg.norm(UTM[Idx] - UTM[Idx, None], axis=-1)
    GeomMeanDist = st.gmean(AllDist[np.triu_indices(len(Idx), k=1)])
    
    return MeanDev, StdDev, RangePct, GeomMeanDist
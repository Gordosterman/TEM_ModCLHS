
import numpy as np
import matplotlib.pyplot as plt
import scipy.spatial.distance as sp

def hist_gko(Var, HistPlotParams):

    f = plt.figure(HistPlotParams['Title'])
    ax = plt.axes(
        xlabel = HistPlotParams['XLabel'],
        ylabel = HistPlotParams['YLabel'],
        title = HistPlotParams['Title'],
        xlim = HistPlotParams['XLims']
    )
    
    FaceColor = HistPlotParams['FaceColor']
    EdgeColor = HistPlotParams['EdgeColor']

    Edges = HistPlotParams['Edges']
    if HistPlotParams['Weight'].casefold() == 'percentile':
        Wghts = np.ones_like(Var) / len(Var)
    else:
        Wghts = np.ones_like(Var)
    
    h = ax.hist(Var, Edges, edgecolor = EdgeColor, weights=Wghts, fc = FaceColor)
    QunantPlot = HistPlotParams['Quantiles']
    nQ = np.shape(QunantPlot)[0]
    for i in range(nQ):
        LineX = np.quantile(Var, QunantPlot[i])
        ax.plot([LineX, LineX], [0, np.max(h[0])], c='k', label=str(QunantPlot[i]) + 'th Quantile')
    
    return f, ax

def sounding_hist(Var, Depth, HistPlotParams):
    # Histograms of total variances w/in distance threshold around each point @ each depth
    # Edges = np.arange(np.min(LogVarAll), np.max(LogVarAll), )

    nL = np.shape(Var)[0]

    
    Edges = HistPlotParams['Edges']
    nB = len(Edges)
    AllLayerHist = [np.histogram(Sounding, bins=Edges)[0] for Sounding in Var]


    pX = np.repeat([Edges], repeats=nL+1, axis = 0)
    DepthPlot = np.append([0], Depth)
    pY = np.repeat(np.reshape(DepthPlot, (nL+1, 1)), repeats = nB, axis=1)

    MedLogVarLyr = np.median(Var, axis=1)         # Median variance of logrho at each layer
    MedLogVarSnd = np.median(Var, axis=0)         # Median variance of logrho for each sounding
    MedLogVarTot = np.median(Var)                 # Median variance of logrho of all layers
    
    QunantPlot = HistPlotParams['Quantiles']
    nQ = np.shape(QunantPlot)[0]
    
    # 
    # NS_LogVarAll = [ut.norm_score(Sounding, True, False) for Sounding in LogVarAll.T]
    # NS_LogVarQuartile = np.quantile(NS_LogVarAll, VarQuantileThresh, axis=1)
    # 
    # 
    f = plt.figure(HistPlotParams['Title'])
    ax = plt.axes(
        title = HistPlotParams['Title'],
        xlabel = HistPlotParams['XLabel'],
        ylabel = HistPlotParams['XLabel'],
        )
    ax.pcolormesh(pX, pY, AllLayerHist)

    for i in range(nQ):
        QuantileLyr = np.quantile(Var, QunantPlot[i], axis=1)
        ax.plot(QuantileLyr, Depth, 'm--', linewidth=4, label=str(QunantPlot[i]) + 'th Quantile')
    
    if nQ>0: ax.legend()
    
    ax.invert_yaxis()
    
    return f, ax
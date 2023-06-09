## COMMENT CODE



def tem_plot_1d(Rho, Depth, DOI, PlotParams):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.colors import Normalize

    MaxDepth =  PlotParams['MaxDepth']
    TitleStr =  PlotParams['TitleStr']
    RhoRng =    PlotParams['RhoRng']
    Log =       PlotParams['Log']
    FigID =     PlotParams['FigID']
    LnWidth =   PlotParams['LnWidth']
    LnClr =     PlotParams['LnClr']

    
    nS = Depth.shape[1] if Depth.ndim > 1 else 1

    
    if isinstance(LnClr, str): LnClr += LnClr[-1] * (nS - np.size(LnClr))          # Fill in any missing color values w/ last given string
    if isinstance(LnClr, tuple):
        LnClr = list(LnClr)
    if isinstance(LnClr, list):
        if np.size(LnWidth) < nS: LnWidth = LnWidth * np.ones(nS,1)

    if isinstance(LnWidth, int) | isinstance(LnWidth, float):
        LnWidth = [LnWidth]
    if isinstance(LnWidth, tuple):
        LnWidth = list(LnWidth)
    if np.shape(LnWidth)[0] < nS: [LnWidth.append(LnWidth[-1]) for i in range(nS - np.shape(LnWidth)[0])]

    f = plt.figure(FigID)
    ax = plt.axes()
    for i in range(nS - 1, -1, -1):
        DepthVect = Depth if nS == 1 else Depth[:, i]
        RhoVect = Rho if nS == 1 else Rho[:, i]

        DepthVect = np.concatenate(([0], DepthVect))
        RhoVect = np.concatenate((RhoVect, [RhoVect[-1]]))
        if nS == 1: DOI = [DOI]

        Idx = np.where(DepthVect<DOI[i])[0].argmax() + 1
        Above_DOI = np.arange(0, Idx)
        Below_DOI = np.arange(Idx, len(DepthVect))
        ax.step(RhoVect[Above_DOI], DepthVect[Above_DOI], where='pre', color=LnClr[i], linewidth=LnWidth[i])
        ax.step(RhoVect[Below_DOI], DepthVect[Below_DOI], where='pre', color=LnClr[i], linewidth=LnWidth[i], linestyle='dashed')
      

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel('Resistivity ' + r'($\Omega$m)', fontsize=14)
    plt.ylabel('Depth (m)', fontsize=14)
    plt.title(TitleStr, fontsize=16)
    plt.xlim(RhoRng)
    plt.ylim([0, MaxDepth])
    
    ax.invert_yaxis()
    plt.xscale('log') if Log else plt.xscale('linear')

    return f
        
    

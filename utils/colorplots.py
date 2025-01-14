# Color plots of 3D resistivity models

# TO DO:
#   Comment up code
#

def color_plot_layer(UTM, TitleStr, Clr, ClrRng, ClrMap, ClrBarTitle, Sz, Log):
    # import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    # from matplotlib.colors import Normalize

    f = plt.figure(TitleStr)
    ax = plt.axes()
    plt.scatter(UTM[:, 0], UTM[:, 1], s=Sz, c=Clr, cmap=ClrMap)
    if Log:
        LogNorm(vmin=ClrRng[0], vmax=ClrRng[1])
    plt.xlabel('UTM Easting')
    plt.ylabel('UTM Northing ')
    plt.title(TitleStr)
    c = plt.colorbar(fraction=0.1)
    c.ax.set_ylabel(ClrBarTitle)

    return f, ax


def color_plot_layer_binary(UTM, TitleStr, Clr, ClrRng, ClrMap, ClrBarTitle, Sz, Log):
    # import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    # from matplotlib.colors import Normalize

    f = plt.figure(TitleStr)
    ax = plt.axes()
    plt.scatter(UTM[:, 0], UTM[:, 1], s=Sz, c=Clr, cmap=ClrMap)
    if Log:
        LogNorm(vmin=ClrRng[0], vmax=ClrRng[1])
    plt.xlabel('UTM Easting')
    plt.ylabel('UTM Northing ')
    plt.title(TitleStr)
    c = plt.colorbar(fraction=0.1)
    c.ax.set_ylabel(ClrBarTitle)

    return f, ax

# Inputs:
#   -UTM Coords + Elevation
#   -All soundings data
#   -DOI
#   -Start Coords, end coords (only lines to start)
#   -Search radius
#   -Plotting Elements
#       -Max Depth to plot
#       -Max Elevation to plot
#       -Aspect Ratio
#       -Width of soundings
#       -Lin/Log Scale
#


def xsection(UTM, Elev, DepthSound, RhoSound, DOI, Lines, PlotParams):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.colors import Normalize

    SoundWidth = PlotParams['SoundWidth']
    FigSize = PlotParams['FigSize']
    SelectPoint = PlotParams['SelectPoint']
    RhoRng = PlotParams['RhoRng']
    SearchRadius = PlotParams['SearchRadius']
    NormDef = LogNorm(vmin=RhoRng[0], vmax=RhoRng[1]) if PlotParams['LinLog'] == 'Log' else Normalize(vmin=RhoRng[0], vmax=RhoRng[1])

    nLines = np.shape(Lines)[0]
    SortUTMRng = np.zeros((nLines, 2, 2))
    for ln in range(nLines):
        LineCoords = Lines[ln]
        # AllDist = (np.cross(LineCoords[1]- LineCoords[0], LineCoords[0] - Coords[:, 0:2]))/np.linalg.norm(LineCoords[1]- LineCoords[0])
        # CloseIdx = np.where(np.abs(AllDist)<SearchRadius)[0]
        # A = LineCoords[0, :]
        # B = LineCoords[1, :]
        PtA = LineCoords[0: 2]
        PtB = LineCoords[2:]
        AllDist = (np.cross(PtB - PtA, PtA - UTM))/np.linalg.norm(LineCoords[1] - LineCoords[0])
        CloseIdx = np.where(np.abs(AllDist) < SearchRadius)[0]

        SelectPointDist = np.dot(SelectPoint - PtA, PtB - PtA)/np.linalg.norm(PtB - PtA)

        SoundingDist = np.dot(UTM[CloseIdx] - PtA, PtB - PtA)/np.linalg.norm(PtB - PtA)

        SelectPointDist = SelectPointDist - np.min(SoundingDist)
        SoundingDist = SoundingDist - np.min(SoundingDist)

        UTMPlot = UTM[CloseIdx]
        RhoPlot = RhoSound[CloseIdx, :]
        ElevPlot = Elev[CloseIdx]
        DOIPlot = DOI[CloseIdx]

        nS, nL = np.shape(RhoPlot)
        DepthPlot = np.append(np.zeros((nS, 1)), DepthSound[CloseIdx, :], axis=1)

        MaxElev = PlotParams['MaxElev'] if np.any(PlotParams['MaxElev']) else np.max(ElevPlot)
        MaxDepth = -1*PlotParams['MaxDepth'] if np.any(PlotParams['MaxDepth']) else np.min(-1*DepthPlot+ElevPlot.reshape(nS,1))

        SortSoundingIdx = np.argsort(SoundingDist)
        SortSounding = SoundingDist[SortSoundingIdx]
        SortDOI = DOIPlot[SortSoundingIdx]
        SortUTM = UTMPlot[SortSoundingIdx]

        SortUTMRng[ln] = [[SortUTM[0, 0], SortUTM[-1, 0]], [SortUTM[0, 1], SortUTM[-1, 1]]]

        XRange = [-5, np.max(SoundingDist)+5]
        YRange = [MaxDepth, MaxElev + 5]

        FigName = PlotParams['FigName'][ln] + ' Line' + str(ln)
        f1 = plt.figure(FigName + 'XSection', figsize=FigSize)
        ax1 = plt.axes(
            xlim=XRange,
            ylim=YRange,
            xlabel='Distance (m)',
            ylabel='Depth (m)',
            title=FigName
            )

        for i in range(len(SoundingDist)):
            X = SoundWidth * np.ones((nL + 1, 2))*[-1, 1] + SoundingDist[i]
            Y = np.array([DepthPlot[i, :], DepthPlot[i, :]]).T*-1 + ElevPlot[i] + DepthPlot[i, 0]
            C = RhoPlot[i, :].reshape(nL, 1)

            plt.pcolormesh(X, Y, C, cmap='viridis', shading='flat', norm=NormDef)

        c = plt.colorbar(fraction=0.1)
        c.ax.set_ylabel('Resistivity '+r'$(\Omega$m)')

        plt.plot([SelectPointDist, SelectPointDist], [MaxDepth, MaxElev], c='k', linestyle='--', linewidth=4)

        plt.plot(SelectPointDist + 10 + DepthPlot[0, :]*np.tan(PlotParams['SensAng']*np.pi/180), DepthPlot[0, :].T*-1 + ElevPlot[i], c='k', linestyle='--', linewidth=2)
        plt.plot(SelectPointDist - (10 + DepthPlot[0, :]*np.tan(PlotParams['SensAng']*np.pi/180)), DepthPlot[0, :].T*-1 + ElevPlot[i], c='k', linestyle='--', linewidth=2)

        plt.plot(SortSounding, SortDOI*-1 + ElevPlot, c='gray')

    if PlotParams['ShowPlanView']:
        Lyr = PlotParams['PlanViewLayer']
        plt.figure('X-Section PlanView')
        plt.axes(
            xlabel='UTM Easting',
            ylabel='UTM Northing ',
            title='X-Section PlanView'
        )
        p1 = plt.scatter(UTM[:, 0], UTM[:, 1], c=RhoSound[:, Lyr], s=20, cmap='jet', norm=NormDef)
        for ln in range(nLines):
            TxtX = SortUTMRng[ln, 0, 0] + 0.05 * (SortUTMRng[ln, 1, 1] - SortUTMRng[ln, 1, 0])
            TxtY = SortUTMRng[ln, 1, 0] + 0.05 * (SortUTMRng[ln, 0, 1] - SortUTMRng[ln, 0, 0])

            plt.plot(SortUTMRng[ln, 0], SortUTMRng[ln, 1], c='k', linewidth=4, linestyle='--')
            plt.text(TxtX, TxtY, 'L' + str(ln), size='large', fontweight='bold')
        plt.scatter(SelectPoint[0], SelectPoint[1], c='w', s=40, edgecolors='k', zorder=1000)
        c = plt.colorbar(p1, fraction=0.1)
        c.ax.set_ylabel(r'$\rho$ $(\Omega$m)')

    return f1, ax1

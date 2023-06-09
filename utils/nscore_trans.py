from sklearn.preprocessing import PowerTransformer
from sklearn.preprocessing import QuantileTransformer
from sklearn.model_selection import train_test_split


# TO DO:
#   Comment up code
#   Plot original data, transformed data
#   Add functionality for reverse transform class
#   Add functionality for selecting transformer method???

# norm_score:
#   Perform normal-score transformation on dataset
#   
#   Input:
#       Data: Dataset to normal-score transform, ndarray
#       QT: True = use quantile transform (robust but needs lots of data)
#           False = use yeo-johnson transform methodology
#       CompHist: True = plot Data and normal score Data histograms
#           Note, Data is 
#   Output: 
#       XYZ_File: Dataframe containing all columns from .xyz file

def norm_score(Data, QT, CompHist):
    import numpy as np
    import matplotlib.pyplot as plt

    if QT:
        ns = QuantileTransformer(
            n_quantiles=500, output_distribution="normal",
            )
    else:
        ns = PowerTransformer(method='yeo-johnson')

    # normdist = rng.normal(size=[10000, 1])
    Data_Ravel = Data.ravel().reshape(-1, 1)
    Train_Data, Test_Data = train_test_split(Data_Ravel, test_size=0.25)
    NS_Data = ns.fit(Train_Data).transform(Data_Ravel)
    
    if CompHist:
        
        
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle('Normal Score Histograms')
        ax1.hist(Data_Ravel, bins=40, label='0-mean input data', alpha=0.5, color='b')
        ax2.hist(NS_Data, bins=40, label='normal-score data', alpha=0.5, color='r')
        ax1.set_title('input data')
        ax2.set_title('normal-score data')
        plt.draw()

    NS_Data_Reshape = np.reshape(NS_Data, np.shape(Data))

    return NS_Data_Reshape

# def inv_norm_score_qt(NS_Data, CompHist):
#     import numpy as np
#     import matplotlib.pyplot as plt

#     ns = QuantileTransformer(
#             n_quantiles=500, output_distribution="normal",
#             )
#     NS_Data_Ravel = NS_Data.ravel().reshape(-1, 1)
#     Inv_NS_Data = ns.inverse_transform(NS_Data)

#     if CompHist:
        
#         plt.figure()
#         plt.axes()
#         plt.hist(Inv_NS_Data - np.nanmean(Inv_NS_Data), bins=40, label='0-mean inv-ns data', alpha=0.5, color='b')
#         plt.hist(NS_Data_Ravel, bins=40, label='normal-score data', alpha=0.5, color='r')
#         plt.legend()
    
#     return Inv_NS_Data
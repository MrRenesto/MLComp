import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings

from DataProcessing import get_raw_data

df_train = get_raw_data('../../Data/train_features.csv')
df_result = get_raw_data('../../Data/train_label.csv')
def_test = get_raw_data('../../Data/test_features.csv')


for i in range(32):

    var = 'feature_' + str(i)
    data = pd.concat([df_result['label'], df_train[var]], axis=1)
    data.plot.scatter(x=var, y='label', ylim=(-1, 2))

    '''
    # Adding labels and a title to the plot
    plt.title(f'Scatter Plot of {var} vs label')
    plt.xlabel(var)
    plt.ylabel('label')
    
    # Display the plot
    plt.show()
    '''

    sns.set()
    sns.pairplot(data, height = 2.5)

    plt.savefig(f'ResultDataExploration/SnsPlot/page_{i}.png', dpi=300)  # Save the current page as an image


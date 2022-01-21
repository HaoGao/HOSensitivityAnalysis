import numpy as np
import pandas as pd
from typing import Sequence

def create_sa_bounds(sa1: bool, log_unif_prior: bool):

    # Fixed Parameter values for SA versus Pressure Analysis
    As_FIXED = 0.69
    Bs_FIXED = 1.11
    Afs_FIXED = 0.31
    Bfs_FIXED = 2.58
    alpha_endo_FIXED = -60.
    alpha_epi_FIXED = 90.

    # set the upper/lower bounds for the sensitibity analysis.

    A_LOWER = .05 # >= 0.1
    A_UPPER = 10 # <= 10

    B_LOWER = .05 # >= 0.1
    B_UPPER = 10 # <= 30

    Af_LOWER = .05 # >= 0.1
    Af_UPPER = 10 # <= 30

    Bf_LOWER = .05 # >= 0.1
    Bf_UPPER = 10 # <= 30

    As_LOWER = .05 # >= 0.1
    As_UPPER = 10 # <= 10

    Bs_LOWER = .05 # >= 0.1
    Bs_UPPER = 10 # <= 30

    Afs_LOWER = .05 # >= 0.1
    Afs_UPPER = 10 # <= 10

    Bfs_LOWER = .05 # >= 0.1
    Bfs_UPPER = 10 # <= 30

    PRESSURE_LOWER = 4.0 # >= 4
    PRESSURE_UPPER = 30.0 # <= 30

    alpha_endo_LOWER = -90 # >= -90
    alpha_endo_UPPER = 0 # <= 0

    alpha_epi_LOWER = 0 # >= 0
    alpha_epi_UPPER = 90 # <= 90


    if log_unif_prior:
        A_LOWER = np.log(A_LOWER) # >= 0.05
        A_UPPER = np.log(A_UPPER) # <= 10

        B_LOWER = np.log(B_LOWER) # >= 0.1
        B_UPPER = np.log(B_UPPER) # <= 30

        Af_LOWER = np.log(Af_LOWER) # >= 0.1
        Af_UPPER = np.log(Af_UPPER) # <= 30

        Bf_LOWER = np.log(Bf_LOWER) # >= 0.1
        Bf_UPPER = np.log(Bf_UPPER) # <= 30

        As_LOWER = np.log(As_LOWER) # >= 0.05
        As_UPPER = np.log(As_UPPER) # <= 10

        Bs_LOWER = np.log(Bs_LOWER) # >= 0.1
        Bs_UPPER = np.log(Bs_UPPER) # <= 30

        Afs_LOWER = np.log(Afs_LOWER) # >= 0.05
        Afs_UPPER = np.log(Afs_UPPER) # <= 10

        Bfs_LOWER = np.log(Bfs_LOWER) # >= 0.1
        Bfs_UPPER = np.log(Bfs_UPPER) # <= 30

    if sa1:
        return [[A_LOWER, A_UPPER],
                 [B_LOWER, B_UPPER],
                 [Af_LOWER, Af_UPPER],
                 [Bf_LOWER, Bf_UPPER],
                 [As_LOWER, As_UPPER],
                 [Bs_LOWER, Bs_UPPER],
                 [Afs_LOWER, Afs_UPPER],
                 [Bfs_LOWER, Bfs_UPPER],
                 [PRESSURE_LOWER, PRESSURE_UPPER],
                 [alpha_endo_LOWER, alpha_endo_UPPER],
                 [alpha_epi_LOWER, alpha_epi_UPPER]]


    if log_unif_prior:
        As_FIXED = np.log(As_FIXED)
        Bs_FIXED = np.log(Bs_FIXED)
        Afs_FIXED = np.log(Afs_FIXED)
        Bfs_FIXED = np.log(Bfs_FIXED)


    # Boundary list for SA where some inputs kept fixed
    DELTA = 1e-4

    return             [[A_LOWER, A_UPPER],
                       [B_LOWER, B_UPPER],
                       [Af_LOWER, Af_UPPER],
                       [Bf_LOWER, Bf_UPPER],
                       [As_FIXED - DELTA, As_FIXED + DELTA],
                       [Bs_FIXED - DELTA, Bs_FIXED + DELTA],
                       [Afs_FIXED - DELTA, Afs_FIXED + DELTA],
                       [Bfs_FIXED - DELTA, Bfs_FIXED + DELTA],
                       [PRESSURE_LOWER, PRESSURE_UPPER],
                       [alpha_endo_FIXED - DELTA, alpha_endo_FIXED + DELTA],
                       [alpha_epi_FIXED - DELTA, alpha_epi_FIXED + DELTA]]

def normalise_data(data: np.array, col_means=None, col_std=None):
    """
    Normalises data to mean zero, unit variance

    This function takes in NxD np array data, and returns
    normalised data where each column has zero mean and
    unit variance

    If col_means and col_std are passed in, the normalisation
    is done wrt these inputs. If not, the col_means and col_std
    of the inputted data are calculdated and the nused for the
    normalisation
    """

    if (col_means is None) or (col_std is None):
        means = data.mean(axis=0)
        std = data.std(axis=0)
        return_stats = True
    else:
        means = col_means
        std = col_std
        return_stats = False

    norm_data = (data-means) / std

    if return_stats:
        return norm_data, means, std

    return norm_data

def standardise_data(data: np.array, min_value=None, max_minus_min_value=None):
    """
    Standardises data to lie within [0,1]

    This function takes in NxD np array data, and returns
    standardised data where all the values of data lie within [0,1]

    If col_means and col_std are passed in, the standardisation
    is done wrt these inputs. If not, the col_means and col_std
    of the inputted data are calculdated and the nused for the
    standardisation
    """
    if (min_value is None) or (max_minus_min_value is None):
        min_value = data.min(axis=0)
        max_minus_min_value = data.max(axis=0) - min_value
        return_parameters = True
    else:
        return_parameters = False

    standardised_data = (data - min_value) / max_minus_min_value

    if return_parameters:
        return standardised_data, min_value, max_minus_min_value
    return standardised_data

def load_and_normalise_data(sa_outputs_list: Sequence[str],
                            start_index: int = 0, data_count: int = 100,
                            Xmean=None, Xstd=None, Ymean=None, Ystd=None):
    """
    Loads train/test data, and normalisese to mean zero / unit variance

    This functions loads training input/output data from the simulator.

    Inputs are the eleven input parameters, and outputs are as specified in "sa_outputs_list"

    Data is normalised with respect to mean zero and unit variance
    """

    # load data
    if Xmean is None:
        full_data = pd.read_csv(f'trainingData/trainData.txt')
        # select only first NN data points to run CV on
        full_data = full_data.iloc[start_index:(start_index+data_count),:]
    else:
        full_data = pd.read_csv(f'trainingData/testData.txt')
        full_data = full_data.iloc[start_index:(start_index+data_count),:]

    # extract training inputs, and normalise
    X = full_data.values[:,:11] # first eleven columns are input parameters

    # log the eight material parameters
    X[:,:8] = np.log(X[:,:8])

    # normalise data
    if Xmean is None:
        Xnorm, Xmean, Xstd = normalise_data(X)
    else:
        Xnorm = normalise_data(X, Xmean, Xstd)

    # extract training outputs, and normalise
    Y = full_data[sa_outputs_list].values

    if Ymean is None:
        Ynorm, Ymean, Ystd = normalise_data(Y)
    else:
        Ynorm = normalise_data(Y, Ymean, Ystd)

    # return normalised data
    return Xnorm, Xmean, Xstd, Ynorm, Ymean, Ystd

import gpflow
import numpy as np
from absl import logging
from typing import Sequence

# Nugget term added to training data for numerical stability in GP matrix operations
INITIAL_NOISE_VARIANCE = 5e-2

def fit_gp(Xtrain: np.array, Ytrain: np.array, kernel_fn: gpflow.kernels) -> gpflow.models.GPR:
    """
    Fits a GP regression model to training data

    This function takes (NxD) input np.array Xtrain, and trains a GPR model to predict
    (Nx1) np.array Ytrain, with kernel function specified by "kernel_fn"
    """

     # Construct a Model
    emulator = gpflow.models.GPR(data=(Xtrain,Ytrain), kernel = kernel_fn, mean_function=None)

    # initial guesses for model hyperparameters
    emulator.likelihood.variance.assign(INITIAL_NOISE_VARIANCE)

    # extract and log pre-trained likelihood variance
    pre_trained_var = gpflow.utilities.leaf_components(emulator)['GPR.likelihood.variance'].numpy()
    logging.info(f'Pre-Trained Likelihood Variance: {pre_trained_var}')

    # Optimise Model hyperparameters
    opt = gpflow.optimizers.Scipy()
    opt_logs = opt.minimize(emulator.training_loss,
                            emulator.trainable_variables,
                            options=dict(maxiter=500))

    # extract and log trained likelihood variance and trained kernel lengthscales
    trained_var = gpflow.utilities.leaf_components(emulator)['GPR.likelihood.variance'].numpy()
    trained_lengthscales = gpflow.utilities.leaf_components(emulator)['GPR.kernel.lengthscales'].numpy()

    logging.info(f'Trained Likelihood Variance: {trained_var}')
    logging.info(f'Trained Kernel LengthScales:{trained_lengthscales}\n')

    return emulator

def validate_gp_one_output(Xtrain: np.array, Xtest: np.array,
                           Ytrain: np.array, Ytest: np.array,
                           Ymean: float, Ystd: float) -> float:
    """
    Validates predictions of trained GPR on out of sample test data

    This function fits a GPR model to (Xtrain, Ytrain). It then predicts
    Ytest given inputs Xtest. The predictions are un-normalised to original
    scale usign Ymean and Ystd. The validation is performed by calculating
    the Q2 coefficient of the GPR predictions against the true outputs Ytest
    """

    # add noise to training data for numerical stability
    Ytrain += (INITIAL_NOISE_VARIANCE**.5)*np.random.standard_normal(Ytrain.shape)

    # convert outputs to 2d array for gpFlow
    Ytrain = Ytrain.reshape(-1,1)

    # now fit gp to data
    kernel = gpflow.kernels.SquaredExponential(lengthscales=[2.]*Xtrain.shape[1])
    gp_model = fit_gp(Xtrain, Ytrain, kernel)

    # make predictions,
    Ypred, _ = gp_model.predict_y(Xtest)
    Ypred = Ypred.numpy().reshape(-1)

    # un-normalise to original scale
    Ypred = (Ypred * Ystd) + Ymean
    Ytest = (Ytest * Ystd) + Ymean

    # calculate Q2 coefficient
    prss = np.sum((Ypred-Ytest)**2)
    tss = np.sum((Ytest - Ytest.mean())**2)
    Q2 = 1 - prss/tss

    return Q2

def validate_gp(outputs_list: Sequence[str],
                start_index: int,
                num_train: int):
        """
        Validates GP predictions on each output quantity of interest

        This function validates the prediction of the trained GP emulator against
        out of sample data, for each output quantity of interest specified in "outputs_list"

        "num_train_value" specifies the number of training data points to fit the GP emulator on

        "start_index" specifies the index first point in the training data set
        the emulator should be fit on

        "start_index" >= 0, and "num_train_value" <= (2000 - start_index)
        """


        from data_utils import load_and_normalise_data
        from directory_functions import setup_save_dir
        import pandas as pd

        q2_array_save_dir, _ = setup_save_dir(start_index, num_train, False, 'emulValidation')

        num_test_value=200

        # load (normalised) training data
        XTrainNorm, Xmean, Xstd, YTrainNorm, Ymean, Ystd = load_and_normalise_data(outputs_list,
                                                                                   start_index,
                                                                                   num_train)
        # load test data (normalised wrt training data mean / variance statistics)
        XTestNorm, _, _, YTestNorm, _, _ = load_and_normalise_data(outputs_list,
                                                                   0, num_test_value,
                                                                   Xmean, Xstd,
                                                                   Ymean, Ystd)

        # np array to hold the Q2 values for each output
        outputs_count = len(outputs_list)
        q2_array = np.zeros((outputs_count, 1), dtype=np.float64)

        for i in range(outputs_count):
            q2_array[i] = validate_gp_one_output(XTrainNorm, XTestNorm,
                                            YTrainNorm[:,i], YTestNorm[:,i],
                                            Ymean[i], Ystd[i])

        # convert to pandas dataframe for nicer formatting and save
        q2_df = pd.DataFrame(q2_array, columns=['Q2'], index=outputs_list)
        q2_df.to_csv(q2_array_save_dir + '/q2_results.txt')



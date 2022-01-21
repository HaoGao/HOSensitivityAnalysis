from emulator_utils import *

def test_gp_fit():
    """
    Test the function "gp_fit" defined in emulator_utils
    """
    import numpy as np
    from SALib.test_functions import Ishigami
    from data_utils import normalise_data
    import matplotlib.pyplot as plt

    Ntrain = 500
    Ntest = 250
    D = 3
    mean = [0.]*D
    std = [1.]*D
    np.random.seed(5)
    Xtrain = np.random.normal(mean, std, size=(Ntrain, D))
    np.random.seed(11)
    Xtest = np.random.normal(mean, std, size=(Ntest, D))

    Ytrain = Ishigami.evaluate(Xtrain)
    Ytest = Ishigami.evaluate(Xtest)

    Ytrain_norm, y_mean, y_std = normalise_data(Ytrain)
    Ytest_norm = normalise_data(Ytest, y_mean, y_std)

    # check that these normalisations worked
    assert np.allclose(Ytrain_norm.mean(0), np.zeros(D))
    assert np.allclose(Ytrain_norm.std(0), np.ones(D))

    # check that normalisation on test data worked
    assert not np.any((np.abs(Ytest_norm.mean(0) - np.zeros(D))) > 0.1)
    assert not np.any((np.abs(Ytest_norm.std(0) - np.ones(D))) > 0.1)

    # add noise for numerical stability
    Ytrain_norm += (INITIAL_NOISE_VARIANCE**.5)*np.random.standard_normal(Ytrain_norm.shape)
    Ytest_norm += (INITIAL_NOISE_VARIANCE**.5)*np.random.standard_normal(Ytest_norm.shape)

    # convert to 2d arrays
    Ytrain_norm = Ytrain_norm.reshape(-1,1)
    Ytest_norm = Ytest_norm.reshape(-1,1)

    # now fit gp to data
    kernel = gpflow.kernels.SquaredExponential(lengthscales=[2.]*D)
    gp_model = fit_gp(Xtrain, Ytrain_norm, kernel)

    # predict on test data
    Ytest_pred, var = gp_model.predict_y(Xtest)

    # find MSE
    mse = (np.mean((Ytest_pred - Ytest_norm)**2))
    print(f'MSE:{mse}')
    assert mse < INITIAL_NOISE_VARIANCE*2

    # save a plot of true versus predicted values
    plt.plot(Ytest_pred[:,0], Ytest_norm[:,0], 'bo')
    plt.plot(Ytest_pred[:,0], Ytest_pred[:,0], 'r')
    plt.title(f'MSE: {mse:.3f}')
    plt.savefig('testingResults/gpTestTrueVersusPred.pdf')
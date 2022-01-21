from data_utils import *


def test_normalisation_and_standardisation():
    """
    Test the function "normalise_data" defined in data_utils.py
    """
    Ntrain = 2500
    Ntest = 2500
    D = 5
    low = np.arange(1,D+1)
    high = low*2
    raw_train_data = np.random.uniform(low, high, size=(Ntrain,D))
    raw_test_data = np.random.uniform(low, high, size=(Ntest,D))

    # check that raw data is the correct size
    assert raw_train_data.shape == (Ntrain,D)
    assert raw_test_data.shape == (Ntest,D)

    # check that normalisation works on training data
    norm_train_data, norm_mean, norm_std = normalise_data(raw_train_data)
    assert np.allclose(norm_train_data.mean(0), np.zeros(D))
    assert np.allclose(norm_train_data.std(0), np.ones(D))

    # check that normalisation works on test data
    norm_test_data = normalise_data(raw_test_data, norm_mean, norm_std)
    assert not np.any((np.abs(norm_test_data.mean(0) - np.zeros(D))) > 0.1)
    assert not np.any((np.abs(norm_test_data.std(0) - np.ones(D))) > 0.1)

    # check that standardisation works on training data
    standard_train_data, train_min, train_high = standardise_data(raw_train_data)
    assert np.allclose(standard_train_data.min(0), np.zeros(D))
    assert np.allclose(standard_train_data.max(0), np.ones(D))

    # check that standardisation works on training data
    standard_test_data = standardise_data(raw_test_data, train_min, train_high)
    assert not np.any((np.abs(standard_test_data.min(0) - np.zeros(D))) > 0.1)
    assert not np.any((np.abs(standard_test_data.max(0) - np.ones(D))) > 0.1)


def test_load_and_normalise_data():

    outputs_list = ['Vol', 'C11', 'R11', 'L11']
    # test data is log-uniformly distributed so must start test at index 1000, after which training data is log-uniformly distributed
    start_value = 1000 
    num_train_value = 1000
    num_test_value=200

    XTrainNorm, Xmean, Xstd, YTrainNorm, Ymean, Ystd = load_and_normalise_data(outputs_list, start_value, num_train_value)

    XTestNorm, _, _, YTestNorm, _, _ = load_and_normalise_data(outputs_list, 0, num_test_value, Xmean, Xstd, Ymean, Ystd)
    
    Din = XTrainNorm.shape[-1]
    Dout = YTrainNorm.shape[-1]

    # check the shapes of the data are what we want
    assert XTrainNorm.shape[0] == num_train_value
    assert YTrainNorm.shape[0] == num_train_value

    assert XTestNorm.shape[0] == num_test_value
    assert YTestNorm.shape[0] == num_test_value

    # check the training data normalisation worked

    assert np.allclose(XTrainNorm.mean(0), np.zeros(Din))
    assert np.allclose(XTrainNorm.std(0), np.ones(Din))

    assert np.allclose(YTrainNorm.mean(0), np.zeros(Dout))
    assert np.allclose(YTrainNorm.std(0), np.ones(Dout))

    # check the test data normalisatoin (approximately) worked

    assert not np.any((np.abs(XTestNorm.mean(0) - np.zeros(Din))) > 0.1)
    assert not np.any((np.abs(XTestNorm.std(0) - np.ones(Din))) > 0.1)

    assert not np.any((np.abs(YTestNorm.mean(0) - np.zeros(Dout))) > 0.1)
    assert not np.any((np.abs(YTestNorm.std(0) - np.ones(Dout))) > 0.15)

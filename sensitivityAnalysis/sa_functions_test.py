from sa_functions import *

def test_sa_implementation():
    """
    Test whether SA implementation from "sa_functions.py" match the results from the
    example here: https://salib.readthedocs.io/en/latest/basics.html
    """
    # load necessary libraries
    from SALib.sample import saltelli
    from SALib.analyze import sobol
    from SALib.test_functions import Ishigami

    # from the above link, we have the true S1/ST values
    true_s1  = np.array([ 0.30644324,  0.44776661, -0.00104936 ])
    true_sT = np.array([ 0.56013728, 0.4387225, 0.24284474])

    # create SA problem
    sa_input_param_names = ['x1', 'x2', 'x3']
    boundary_list = [[-np.pi, np.pi],
                     [-np.pi, np.pi],
                     [-np.pi, np.pi]]
    problem = {
        'num_vars': len(sa_input_param_names),
        'names': sa_input_param_names,
        'bounds': boundary_list
    }

    # find values to run
    param_values = saltelli.sample(problem, 1000, calc_second_order=False)

    # now evaluate using Ishigami function
    saltelli_predictions = Ishigami.evaluate(param_values)

    Si = sobol.analyze(problem, saltelli_predictions, print_to_console=False, calc_second_order=False)

    assert np.sum(np.abs(true_s1 - Si['S1'])) < 0.05
    assert np.sum(np.abs(true_sT - Si['ST'])) < 0.05

def test_run_emulator_sa_sample():
    """
    Test whether function "run_emulator_sa_sample" against toy example problem
    shown here: https://salib.readthedocs.io/en/latest/basics.html
    """

   # load necessary libraries
    from SALib.sample import saltelli
    from SALib.analyze import sobol
    from SALib.test_functions import Ishigami

    # from the above link, we have the true S1/ST values
    true_S  = np.array([ 0.30644324,  0.44776661, -0.00104936 ])
    true_T = np.array([ 0.56013728, 0.4387225, 0.24284474])

    # create training data
    Ntrain = 500
    D = 3
    lower = [-np.pi]*D
    upper = [np.pi]*D
    np.random.seed(5)
    Xtrain = np.random.uniform(lower, upper, size=(Ntrain, D))
    Xtrain_norm, Xmean, Xstd = normalise_data(Xtrain)

    Ytrain = Ishigami.evaluate(Xtrain)
    Ytrain_norm, y_mean, y_std = normalise_data(Ytrain)

    # check that these normalisations worked
    assert np.allclose(Ytrain_norm.mean(0), np.zeros(D))
    assert np.allclose(Ytrain_norm.std(0), np.ones(D))

    # add noise for numerical stability
    Ytrain_norm += (INITIAL_NOISE_VARIANCE**.5)*np.random.standard_normal(Ytrain_norm.shape)

    # convert to 2d array for Gpflow
    Ytrain_norm = Ytrain_norm.reshape(-1,1)
    
    # Train GP model
    kernel = gpflow.kernels.SquaredExponential(lengthscales=[2.]*D)
    gp_model = fit_gp(Xtrain_norm, Ytrain_norm, kernel)

    # create SA problem
    sa_input_param_names = ['x1', 'x2', 'x3']
    boundary_list = [[-np.pi, np.pi],
                     [-np.pi, np.pi],
                     [-np.pi, np.pi]]
    problem = {
        'num_vars': len(sa_input_param_names),
        'names': sa_input_param_names,
        'bounds': boundary_list
    }

    sampled_S, sampled_T = run_emulator_sa_sample(gp_emulator=gp_model,
                                                  sa_problem=problem,
                                                  Xmean=Xmean,
                                                  Xstd=Xstd,
                                                  samples_count=250,
                                                  sa_fidelity_value=5000,
                                                  log_X=False,
                                                  log_unif_prior=False)
    
    assert np.sum(np.abs(true_S - sampled_S.mean(0))) < 0.1
    assert np.sum(np.abs(true_T - sampled_T.mean(0))) < 0.1

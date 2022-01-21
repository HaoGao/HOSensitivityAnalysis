# Script to run a Sensitivity Analysis of the Holzapfel-Ogden Model

from data_utils import *
from emulator_utils import *
from directory_functions import *


import pandas as pd
import matplotlib.pyplot as plt

from SALib.sample import saltelli
from SALib.analyze import sobol

from absl import app
from absl import logging
from typing import Sequence

DELTA=1e-4

def run_emulator_sa_sample(gp_emulator: gpflow.models.GPR,
                           sa_problem: dict,
                           Xmean: np.array,
                           Xstd: np.array,
                           samples_count: int,
                           sa_fidelity_value: int,
                           log_X: bool,
                           log_unif_prior: bool):
    """
    Run SA using a GP emulator, where instead of using a point estimate, we sample from the
    posterior predictive distribution of the GP, giving us an ensemble of sensitivity index values
    """
    # Sample sobol values
    param_values = saltelli.sample(sa_problem, sa_fidelity_value, calc_second_order=False)

    # normalise sobol values for gp prediction
    if log_X and not log_unif_prior:
        param_values[:,:8] = np.log(param_values[:,:8])
    param_values = (param_values - Xmean) / Xstd

    # make "samples_count" samples from trained GP over "param_values" input locations
    samples = gp_emulator.predict_f_samples(param_values, samples_count).numpy()

    # placeholders to hold total-effect and first-order sensitivity indices
    T_indices_values = np.zeros((samples_count,Xmean.shape[0]))
    S_indices_values = np.zeros((samples_count,Xmean.shape[0]))

    # perform SA for each posterior sample from GP
    for sample_index in range(samples_count):

        # extract "sample_index" posterior GP sample
        gp_predictions = samples[sample_index,:, 0]

        # perform SA given the GP predictions
        Si = sobol.analyze(sa_problem, gp_predictions, print_to_console=False, calc_second_order=False)

        # store values of first order and total-effect indices
        S_indices_values[sample_index,:] = Si['S1']
        T_indices_values[sample_index,:] = Si['ST']


    return S_indices_values, T_indices_values

def run_sa1(sa_outputs_list: Sequence[str],
            start_index: int,
            num_train: int,
            input_param_names: Sequence[str],
            sa_fidelity_value: int,
            log_unif_prior: bool,
            samples_count: int):
    """
    Function to run SA1 of HO model, where all 11 input parameters are considered random variables
    """

    # load data
    Xnorm, Xmean, Xstd, Ynorm, _, _ = load_and_normalise_data(sa_outputs_list, start_index, num_train)

    # create parameter boundary list for performing SA
    boundary_list = create_sa_bounds(sa1=True, log_unif_prior=log_unif_prior)

    # define SA analysis problem dictionary
    problem = {
        'num_vars': len(input_param_names),
        'names': input_param_names,
        'bounds': boundary_list
    }

    # set up directory so save results to
    _, data_save_dir = setup_save_dir(start_index, num_train, log_unif_prior, 'SA1')


    for i, output_i in enumerate(sa_outputs_list):
        logging.info(f'Starting SA1 for {output_i}')

        ## Train GP model
        # add nugget to outputs for numerical stability
        YtrainL = Ynorm[:,i]
        YtrainL += (INITIAL_NOISE_VARIANCE**.5)*np.random.standard_normal(YtrainL.shape)

        # convert to 2d array for Gpflow
        YtrainL = YtrainL.reshape(-1,1)

        # train GP model
        kernel = gpflow.kernels.SquaredExponential(lengthscales=[2.]*Xnorm.shape[1])
        gp_model = fit_gp(Xnorm, YtrainL, kernel)

        # perform SA to find first-order (S) and total-effect (T) sensitivity indices
        S, T = run_emulator_sa_sample(gp_model, problem,
                                      Xmean, Xstd,
                                      samples_count, sa_fidelity_value,
                                      True, log_unif_prior)

        # write results to save directory
        np.save(f'{data_save_dir}/{output_i}_S_samples.npy', S)
        np.save(f'{data_save_dir}/{output_i}_T_samples.npy', T)

def run_sa_against_pressure(Xtrain: np.array,
                            Ytrain: np.array,
                            fixed_EDP_values: Sequence[float],
                            input_boundaries,
                            samples_count: int,
                            sa_fidelity_value: int,
                            log_X: bool,
                            log_unif_prior: bool,
                            Xmean: np.array,
                            Xstd: np.array,
                            input_param_names: Sequence[str]):
    """
    Runs series of SA for HO model with fixed values of EDP
    """
    # add nugget to outputs for numerical stability
    Ytrain += (INITIAL_NOISE_VARIANCE**.5)*np.random.standard_normal(Ytrain.shape)

    # convert to 2d array for Gpflow
    Ytrain = Ytrain.reshape(-1,1)

    # train GP model
    kernel = gpflow.kernels.SquaredExponential(lengthscales=[2.]*Xtrain.shape[1])
    gp_model = fit_gp(Xtrain, Ytrain, kernel)


    # initialise placeholder list
    sT_samples_list = []

    # loop over each pressure values
    for pressure_value in fixed_EDP_values:

        # update input_boundaries
        input_boundaries[8] = [pressure_value-DELTA, pressure_value+DELTA]

        # create SA problem
        # define SA analysis problem dictionary
        problem = {
            'num_vars': len(input_param_names),
            'names': input_param_names,
            'bounds': input_boundaries
        }

        # run SA
        _, sT_indices_values = run_emulator_sa_sample(gp_model, problem,
                                                      Xmean, Xstd,
                                                      samples_count, sa_fidelity_value,
                                                      log_X, log_unif_prior)

        # write to placeholder variable
        sT_samples_list.append(sT_indices_values)


    return sT_samples_list

def run_sa2(sa_outputs_list: Sequence[str],
            start_index: int,
            num_train: int,
            input_param_names: Sequence[str],
            sa_fidelity_value: int,
            log_unif_prior: bool,
            fixed_EDP_values: Sequence[float],
            samples_count: int,
            fixed_pressure_params: Sequence[str]):
    """
    Function to run SA2 of HO model, where total-effect sensitivity indices for material
    paramters a, b, af, bf are computed as a function of EDP value
    """

    # load data
    Xnorm, Xmean, Xstd, Ynorm, _, _ = load_and_normalise_data(sa_outputs_list, start_index, num_train)

    # set up directory so save results to
    _, data_save_dir = setup_save_dir(start_index, num_train, log_unif_prior, 'SA2')

    # create list of upper / lower bounds for each input parameter for SA
    boundary_list = create_sa_bounds(sa1=False, log_unif_prior=log_unif_prior)


    def compute_sample_quantiles(samples_list: Sequence,
                                 quantile: float) -> pd.DataFrame:

        EDP_values_count =  len(fixed_EDP_values)
        samples_list_quantile_values = np.zeros((EDP_values_count, len(input_param_names)))

        # loop over each fixed EDP value, and calculate specified quantile value
        for i in range(EDP_values_count):
                samples_i = samples_list[i]

                # if quantile = 0.5, return mean instead of median
                if quantile == 0.5:
                    quantile_values = samples_i.mean(0)
                else:
                    quantile_values = np.quantile(samples_i, quantile, 0)

                samples_list_quantile_values[i,:] = quantile_values

        # return DataFrame of index quantile values for first four parmaters (a, b, af and bf)
        return pd.DataFrame(samples_list_quantile_values.T[:4,:],
                            columns = fixed_EDP_values,
                            index = fixed_pressure_params)


    for i, output_i in enumerate(sa_outputs_list):
        logging.info(f'Running Fixed-Pressure SA for Output:{output_i}\n')
        sT_samples_list_i = run_sa_against_pressure(Xnorm, Ynorm[:,i],
                                                    fixed_EDP_values, boundary_list,
                                                    samples_count, sa_fidelity_value,
                                                    True, log_unif_prior,
                                                    Xmean, Xstd,
                                                    input_param_names)

        # calculate mean / 97.5 percentile and 2.5 percentile of sampled total-effect index values for input parameters a, b af and bf
        mean_sens_results = compute_sample_quantiles(sT_samples_list_i, 0.5)
        upper_sens_results = compute_sample_quantiles(sT_samples_list_i, 0.975)
        lower_sens_results = compute_sample_quantiles(sT_samples_list_i, 0.025)

        # save the pandas arrays
        mean_sens_results.to_csv(f'{data_save_dir}/{output_i}_meanSamples_{sa_fidelity_value}_{log_unif_prior}.txt', index=True)
        upper_sens_results.to_csv(f'{data_save_dir}/{output_i}_upperSamples_{sa_fidelity_value}_{log_unif_prior}.txt', index=True)
        lower_sens_results.to_csv(f'{data_save_dir}/{output_i}_lowerSamples_{sa_fidelity_value}_{log_unif_prior}.txt', index=True)




######################################
### Section 1: Shell input variables
### Use flags imported above to set some
### script variables that can be directly
### changed from the shell when executing
### this script
######################################

from absl import app
from absl import flags
from absl import logging

from typing import Sequence


flags.DEFINE_enum(
    'run_type', 'run_sa1', ['run_sa1', 'plot_sa1', 'run_sa2', 'plot_sa2',  'emulator_validation', 'plot_sa1_sample', 'plot_all_sa'],
    help = 'Run SA1, SA2, or make plots for paper'
    )
flags.DEFINE_boolean(
    'log_unif_prior', False,
    help = 'Whether to run SA with log-unif prior, or log prior'
    )
flags.DEFINE_list(
    'sa_2_output', ['Vol', 'C11'],
    help = 'List of outputs to run SA2 for (Vol and/or C11)'
    )
flags.DEFINE_integer(
    'start_value', 0,
    lower_bound=0, upper_bound=1999,
    help = 'The starting index of the training data for fitting the GP emulator'
    )
flags.DEFINE_integer(
    'num_train', 2000,
    lower_bound = 1, upper_bound = 2000,
    help = 'The number of data points to train GP emulator, starting at start_value'
    )
FLAGS = flags.FLAGS

######################################
### Section 2: Hard-coded input variables
### Set up some input variables which are
### hard-coded for all SA runs
######################################

# The number of GP samples to use for SA2 when we estimate
# the sensitivity indices via sampling
SAMPLES_COUNT: int = 250

# This determines how many quasi-monte carlo points are used
# when calculating the sensitivity indices. Higher SOBOL_VALUE
# means more accurate estimates, at higher comp. expense
SOBOL_VALUE: int = 5000

# names of the input parameters in the training data .txt file
INPUTS_NAMES: Sequence[str] = ['a', 'b', 'af', 'bf', 'as', 'bs', 'afs','bfs', 'pressure', 'alpha.endo', 'alpha.epi'] # list of strings

# These are the names used when labelling the Sensitivity Analysis Plots
INPUT_PARAM_NAMES: Sequence[str] = [r'$a$', r'$b$',
                                    r'$a_{\rm{f}}$', r'$b_{\rm{f}}$',
                                    r'$a_{\rm{s}}$', r'$b_{\rm{s}}$',
                                    r'$a_{\rm{fs}}$',r'$b_{\rm{fs}}$',
                                    r'EDP',
                                    r'$\alpha_{endo}$', r'$\alpha_{epi}$']
INPUTS_TRAIN_DIM = len(INPUTS_NAMES)

# Decide which outputs we want to run SA1 on
SA1_OUTPUTS_LIST: Sequence[str] = ['Vol', 'R11', 'C11', 'L11', 'Apex_Twist']

# Dictionary holding output QoI labels from the training data .txt file (keys) 
# and the corresponding label we use in the SA plots (values)
SA_OUTPUTS_LIST_NAMES: dict = {'Vol': 'LVV',
                               'C11': r'$\varepsilon_{cc}^*$',
                               'R11': r'$\varepsilon_{rr}^*$',
                               'L11': r'$\varepsilon_{ll}^*$',
                               'Apex_Twist':'Torsion'}

# Set the parameters to be considered for SA2
SA2_PARAMS: Sequence[str] = ['a', 'b', 'af', 'bf']
# gives up to pressure 25 mmHg
FIXED_EDP_VALUES: Sequence[float] = [5. + i*2.5 for i in range(9)] 

######################################
### Section 3: Run Analysis
######################################

def main(_):

    if FLAGS.run_type == 'emulator_validation':
        import emulator_utils as eu
        eu.validate_gp(outputs_list=SA1_OUTPUTS_LIST,
                       start_index=FLAGS.start_value,
                       num_train=FLAGS.num_train)

    if FLAGS.run_type == 'run_sa1':
        import sa_functions as sa
        sa.run_sa1(SA1_OUTPUTS_LIST,
                   FLAGS.start_value,
                   FLAGS.num_train,
                   INPUTS_NAMES,
                   SOBOL_VALUE,
                   FLAGS.log_unif_prior,
                   SAMPLES_COUNT)

    if FLAGS.run_type == 'plot_sa1':
        import plotting_functions as pf
        pf.plot_sa1(SA1_OUTPUTS_LIST,
                    FLAGS.start_value,
                    FLAGS.num_train,
                    FLAGS.log_unif_prior,
                    INPUT_PARAM_NAMES,
                    SA_OUTPUTS_LIST_NAMES)

    if FLAGS.run_type == 'run_sa2':
        import sa_functions as sa
        sa.run_sa2(FLAGS.sa_2_output,
                   FLAGS.start_value,
                   FLAGS.num_train,
                   INPUTS_NAMES,
                   SOBOL_VALUE,
                   FLAGS.log_unif_prior,
                   FIXED_EDP_VALUES,
                   SAMPLES_COUNT,
                   SA2_PARAMS)

    if FLAGS.run_type == 'plot_sa2':
        import plotting_functions as pf
        pf.plot_sa2(FLAGS.start_value,
                    FLAGS.num_train,
                    FLAGS.log_unif_prior,
                    SOBOL_VALUE,
                    SA2_PARAMS,
                    FIXED_EDP_VALUES)

if __name__ == "__main__":
    app.run(main)



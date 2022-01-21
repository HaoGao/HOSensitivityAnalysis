import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from directory_functions import setup_save_dir
from typing import Sequence, Dict

USE_TEX = False # whether to format plot text using latex or not

def plot_sa1(sa_outputs_list: Sequence[str],
             start_index: int,
             num_train: int,
             log_unif_prior: bool,
             input_param_names: Sequence[str],
             output_name_dict: Dict[str, str]) -> None:
    """
    Plots the results of SA1

    This function plots the total-effect sensitivity indices of the 8 HO material
    parameters, the RBM fibre parameters, and end-diastolic pressure (EDP)

    SA1 is performed by running "run_sa1" in main.py, and the resulting data is
    stored in "sa_data_save_dir" below
    """
    # directories to plot sa results, and where sa results data are stored
    plot_save_dir, sa_data_save_dir = setup_save_dir(start_index, num_train, log_unif_prior, sa_type= 'SA1')

    # x-axis indices to plot the results against
    ind = np.arange(len(input_param_names))
    width = .35 # distance between boxplots

    # update plotting settings
    plt.rc('text', usetex=USE_TEX)
    plt.rc('font', family='serif')
    plt.rcParams.update({'font.size': 20})
    plt.rcParams.update({"legend.markerscale": 0.04})

    for output_i in sa_outputs_list:

        # Load first-order sensitivity index samples
        S = np.load(f'{sa_data_save_dir}/{output_i}_S_samples.npy', allow_pickle=True)
        # load total-effect sensitivity index samples
        T = np.load(f'{sa_data_save_dir}/{output_i}_T_samples.npy', allow_pickle=True)

        # now create boxplot of sampled index values
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
        ax.yaxis.grid(True)

        # create bar plots up to median senstivity index values
        Smedian = np.median(S, axis=0)
        ax.bar(ind, Smedian, width, label=r'First-Order Index', bottom=0, zorder=10, edgecolor='black')

        Tmedian = np.median(T, axis=0)
        ax.bar(ind + width, Tmedian, width, label=r'Total-Effect Index', bottom=0, zorder=10, edgecolor='black')

        ax.set_ylim(0, 0.65)

        # overlay barplot with boxplot to show dispersion of sampled index values
        ax.boxplot(T, positions=ind+width, showfliers=False, patch_artist=False, zorder=11, widths=width, medianprops = dict(linewidth=0))
        ax.boxplot(S, positions=ind, showfliers=False, patch_artist=False, zorder=11, widths=width, medianprops = dict(linewidth=0))

        # for uniform SA, put output name on y-axis of plot
        if not log_unif_prior:
            label_value = output_name_dict[output_i]
            ax.set_ylabel(r'{}'.format(label_value), labelpad=20, fontsize = 28)

        # add input parameter names to x-axis of plot
        plt.xticks(ind + width / 2, input_param_names, fontsize=22)
        ax.legend(loc='best', framealpha=1.)

        # save plot
        plot_save_name = f'{plot_save_dir}/sa1_plot_{output_i}_{log_unif_prior}.pdf'
        plt.savefig(plot_save_name, bbox_inches='tight')


def plot_sa2(start_index: int,
             num_train: int,
             log_unif_prior: bool,
             sobol_value: int,
             fixed_pressure_params: Sequence[str],
             fixed_pressure_values: Sequence[float]) -> None:
    """
    Plots the results of SA2

    This function plots the total-effect sensitivity indices of the material
    parameters a, b, af and bf against end-diastolic pressure (EDP)

    SA2 is performed by running "run_sa2" in main.py, and the resulting data is
    stored in "sa_data_save_dir" below
    """

    # directories to plot sa results, and where sa results data are stored
    save_dir, sa_data_save_dir = setup_save_dir(start_index, num_train, log_unif_prior, sa_type= 'SA2')

    # settings for the plots
    linewidth = 3
    subplot_labels=[r'$a$',r'$b$',r'$a_{\mathrm{f}}$',r'$b_{\mathrm{f}}$']
    plt.rc('text', usetex=USE_TEX)
    plt.rc('font', family='serif')
    plt.rcParams.update({'font.size': 14})

    # read in mean / 97.5 Percentile / 2.5 Percentile of sampled total-effect sensitivty indices for Volume of each material parameter
    mean_sens_results_vol = pd.read_csv(f'{sa_data_save_dir}/Vol_meanSamples_{sobol_value}_{log_unif_prior}.txt', index_col=0)
    upper_sens_results_vol = pd.read_csv(f'{sa_data_save_dir}/Vol_upperSamples_{sobol_value}_{log_unif_prior}.txt', index_col=0)
    lower_sens_results_vol = pd.read_csv(f'{sa_data_save_dir}/Vol_lowerSamples_{sobol_value}_{log_unif_prior}.txt', index_col=0)

    # read in mean / 97.5 Percentile / 2.5 Percentile of sampled total-effect sensitivty indices for circumferential strains of each material parameter
    mean_sens_results_c11 = pd.read_csv(f'{sa_data_save_dir}/C11_meanSamples_{sobol_value}_{log_unif_prior}.txt', index_col=0)
    upper_sens_results_c11 = pd.read_csv(f'{sa_data_save_dir}/C11_upperSamples_{sobol_value}_{log_unif_prior}.txt', index_col=0)
    lower_sens_results_c11 = pd.read_csv(f'{sa_data_save_dir}/C11_lowerSamples_{sobol_value}_{log_unif_prior}.txt', index_col=0)

    # create figure
    fig, axs = plt.subplots(1,4, figsize=(15, 5), facecolor='w', edgecolor='k')
    axs = axs.ravel()
    fig.subplots_adjust(hspace = .15, wspace=.2,left=0.07,right=0.99,bottom=0.1,top=0.92)
    fig.text(0.5,0.005,'EDP (mmHg)',fontsize=20)

    # loop over each material parameter
    for par_index, plot_outputs_jk in enumerate(fixed_pressure_params):

        # set title of subplot to be material parameter name
        axs[par_index].set_title(subplot_labels[par_index],fontsize=20)

        # make plot of volume index values against pressure
        axs[par_index].plot(fixed_pressure_values, mean_sens_results_vol.loc[plot_outputs_jk],
                      color='forestgreen', label=r'LVV', linewidth=linewidth)
        axs[par_index].plot(fixed_pressure_values, upper_sens_results_vol.loc[plot_outputs_jk],
                      linestyle=':', color='forestgreen')
        axs[par_index].plot(fixed_pressure_values, lower_sens_results_vol.loc[plot_outputs_jk],
                      linestyle=':', color='forestgreen')

        # make plot of circumferential strain index values against pressure
        axs[par_index].plot(fixed_pressure_values, mean_sens_results_c11.loc[plot_outputs_jk],
                      color='steelblue', label=r'$\varepsilon_{cc}^*$', linewidth=linewidth)
        axs[par_index].plot(fixed_pressure_values, upper_sens_results_c11.loc[plot_outputs_jk],
                      linestyle=':', color='steelblue')
        axs[par_index].plot(fixed_pressure_values, lower_sens_results_c11.loc[plot_outputs_jk],
                      linestyle=':', color='steelblue')

        # set y-axis label only on left most plot
        if par_index == 0:
            axs[par_index].set_ylabel(r'Total-Effect Sensitivity Index', fontsize=20)

        axs[par_index].set_ylim([0,1])
        axs[par_index].grid(axis='y')
        axs[par_index].set_axisbelow(True)
        axs[par_index].legend()

    # save final figure
    save_name = f'{save_dir}/sa2_plot_{log_unif_prior}.pdf'
    plt.savefig(save_name, bbox_inches='tight')
    plt.close()



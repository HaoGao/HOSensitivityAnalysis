# Holzapfel-Ogden Model Sensitivity Analysis

This repository contains Python scripts to carry out a Sobol sensitivity analysis (SA) of the Holzapfel-Ogden (HO) model for passive left ventricle mechanics.

The analyses are performed using Gaussian process (GP) emulators, which allow for the required computations to be performed in reasonable time. The SA is performed using the library [SALib](https://github.com/SALib/SALib)

## Environment Setup

With [Anaconda](https://www.anaconda.com/download/):

```
conda env create -f environment.yml
conda activate ho-sa
```

## Running Sensitivity Analyses

Run Sensitivity Analysis 1 (SA1), with log-uniform prior:

```
python -m main --run_type="run_sa1" --log_unif_prior=True
```
The results can then be plotted with:
```
python -m main --run_type="plot_sa1" --log_unif_prior=True
```
The above commands re-create the second column of plots in Figure 3 of the paper

## Additional Comments

1. To run and plot SA1 with the uniform prior distribution, set ``--log_unif_prior=False`` in the above commands
2. Similarly, to run SA2, replace ``"run_sa1"`` and ``"plot_sa1"`` with ``"run_sa2"`` and ``"plot_sa2"``  respectively
3. These analyses, especially SA2, are computationally expensive both in terms of execution time and memory. To reduce the computational expense, reduce the values``SAMPLES_COUNT`` (line 51)  and / or ``SOBOL_VALUE`` (line 56) in the ``main.py`` script.
4. If you have Latex installed on your machine, you can set ``USE_TEX = True`` on line 7 of ``plotting_functions.py``. This will recreate the plots with the same formatting as seen in the paper.
5. Validation of the GP emulators against the out of sample data can be run by setting ``--run_type="emulator_validation"`` above

## Running Tests

Any scripts ending with ```_test.py``` test the Python script with the corresponding prefix. The tests for ```sa_functions.py``` can be run for example as follows:
 ```
pytest sa_functions_test.py -v
```

## Subdirectories:

### /training Data

Stores simulation data for training and validating the GP emulators

### /results
Stores the results of the SA


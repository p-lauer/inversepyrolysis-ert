# Inverse modelling pyrolization kinetics with ensemble learning methods - scripts

These scripts can be used to reproduce the study described in the article "Inverse modelling pyrolization kinetics with ensemble learning methods" [[1]]. The scripts here are intended to be used with default configuration, if not noted otherwise. To apply this model to your own data, you can adopt the scripts shown in ./test_model/.

## ./generate_db/

The scripts `generate_1c.py`, `generate_2c.py` and `generate_3c.py` are used to generate the training data for the model. There is an individual script for 1, 2 and 3 components. The reaction kinetic parameters and component fractions are sampled randomly and then the mass loss rate for TGA experiments with four different constant heating rates are calculated. The scripts are intended to be used on multiple CPUs. Example data sets are available for download [[2]].

In the following table, there are paramaters listed that can easily set by the user. Further parameters, as heating rates and sampling rates can also be modified in the scripts but may need more caution.

|Parameter|Description|Default value|
|---------|------------------------------------------------------|----------|
| i		  |number of elements that will be generated             | 6400000  |
| cores   |number of CPU cores used for generation                   | 128      |
| rrlimlow|lower boundary of peak reaction rate sampling (in /s) | 0.001    |
| rrlimup |upper boundary of peak reaction rate sampling (in /s) | 0.01     |
| rtlimlow|lower boundary of peak reaction rate sampling (in °C) | 100      |
| rtlimup |upper boundary of peak reaction rate sampling (in °C) | 500      |
| Tstart  |Start temperature of experiment (in °C)               |  20      |
| Tend    |End temperature of experiment (in °C)                 | 550      |

Standard output will be the mass loss rate for four TGA experiments with following configuration:

|Heating rate|Heating rate value|Time step $\Delta t$|Temperature step $\Delta T$|
|------|------------|---------|----------------|
|$\beta_1$     | 5 K/min	| 24 s	  | 2 K			   |
|$\beta_2$	   |10 K/min	| 12 s	  | 2 K			   |
|$\beta_3$	   |30 K/min    |  4 s    | 2 K			   |
|$\beta_4$     |40 K/min    |  3 s	  | 2 K			   |

The scripts will generate numbered files named `features*` and `labels*` individually for each used CPU core. In addition there is a `labels*` file that holds all generated labels. The naming convention can be seen in the table below and a description of the data structure in the files can be found under [Data structure](#data-structure) below.

|Filename | Description |
|---------|-------------|
|labels{$N_u$/1000}k_{$\beta_1$}_{$\beta_2$}_{$\beta_3$}_{$\beta_4$}_{$n$}_{$\Delta T$/s}.csv 				|generated labels  |
|features{$N_u$/1000}k_{$\beta_1$}_{$\beta_2$}_{$\beta_3$}_{$\beta_4$}_{$n$}_{$\Delta T$/s}_{$i_{core}$}.csv |generated labels, output per core |
|labels{$N_u$/1000}k_{$\beta_1$}_{$\beta_2$}_{$\beta_3$}_{$\beta_4$}_{$n$}_{$\Delta T$/s}_{$i_{core}$}.csv 	|generated labels, output per core  |

## ./build_models/

These scripts generate the individual sub models as described in the following table.

|Filename   |Output               | Description          |
|-----------|---------------------|----------------------|
|`sm1.py`   |`sm1.pickle`    | Sub model 1 (Classifier for estimation of number of components)|
|`sm3_1c.py`|`sm3_1c.pickle`| Sub model 3 for materials with 1 component (Regressor for estimation of reaction kinetic parameters)|
|`sm3_2c.py`|`sm3_2c.pickle`| Sub model 3 for materials with 2 components (Regressor for estimation of reaction kinetic parameters)|
|`sm3_3c.py`|`sm3_3c.pickle`| Sub model 3 for materials with 3 components (Regressor for estimation of reaction kinetic parameters)|

Default parameters are the ones used in the corresponding publication. A contemporary high performance system (128 CPU cores, 1 TB RAM) will be needed for this. However, the following hyper parameter values shall be taken as lower limits to produce models with slightly less accurate predictions but much less computational demand. Pre built models are available for download [[3]].

|Parameter | Sub model 1 (`sm1.py`) | Sub model 3, 1 Component (`sm3_1c.py`) | Sub model 3, 2 components (`sm3_2c.py`)|Sub model 3, 3 components (`sm3_3c.py`)|
|---|---|---|---|---|
|Number of estimators|500|1|50|50|
|Maximum tree depth|100|50|50|50|50|

## ./test_model/

Scripts to test the complete model

`calculate.py` in default configuration will calculate predictions from data not used during the training process for 150000 elements and output the prediction of the reaction kinetic parameters to `prediction.csv`

`evaluate.py` then evaluates the true (prescribed) and predicted reaction kinetic parameters and initial fractions by calculating R^2 scores.  It also calculates mass loss rates from the predictions and compares them to the initial mass loss rates used as input features. For comparison, the normalised RMSE is calculated and the distribution is plotted.

## <a name="data-structure"></a> Data structure

### `labels*.csv`

The mass loss rates are calculated from the reaction kinetic parameters can be found at the same row number of the corresponding `features*` file. The files do not have any header.

Columns description:

|Column|Symbol|Description|Unit|
|------|------|-----------|----|
| 1|$r_{1,p}$ |Peak reaction rate for component 1 			| s^-1 |
| 2|$r_{2,p}$ |Peak reaction rate for component 2 			| s^-1 |
| 3|$r_{3,p}$ |Peak reaction rate for component 3 			| s^-1 |
| 4|$T_{1,p}$ |Peak reaction temperature for component 1 		| °C   |
| 4|$T_{2,p}$  |Peak reaction temperature for component 2 		| °C   |
| 6|$T_{3,p}$  |Peak reaction temperature for component 3 		| °C   |
| 7|$Y_1$|Fraction of component 1						| 1	   |
| 8|$Y_2$|Fraction of component 2						| 1	   |
| 9|$Y_3$|Fraction of component 3						| 1	   |
|10|$A_1$|Pre-exponential factor for component 1			| s^-1 |
|11|$A_2$| Pre-exponential factor for component 2			| s^-1 |
|12|$A_3$| Pre-exponential factor for component 3			| s^-1 |
|13|$E_1$| Activation energy for component 1				| J/mol|
|14|$E_2$| Activation energy for component 2				| J/mol|
|15|$E_3$| Activation energy for component 3				| J/mol|
|16|$Y_1$| Fraction of component 1						| 1	   |
|17|$Y_2$| Fraction of component 2						| 1	   |
|18|$Y_3$| Fraction of component 3						| 1	   |

### `features*.csv`

The files do not have any header. There is a set of mass loss records for four TGA experiments with identical reaction kinetic parameters and four different heating rates per row. One row can be separated into the four experiments as shown in the following tabel. The mass loss rates are calculated from the reaction kinetic parameters in the same row number of the corresponding `labels*` file.

|Column|Description|Unit|
|------|-----------|----|
| 	1...266| Mass loss rate at $\beta_1$ 			| s^-1 |
| 267...532| Mass loss rate at $\beta_2$ 			| s^-1 | 
| 533...798| Mass loss rate at $\beta_3$ 			| s^-1 |
| 799...1064| Mass loss rate at $\beta_4$ 			| s^-1 |

Corresponding $T$ is 20...550 °C with $\Delta T=2K$. Then, $t$ is $\frac{T}{\beta}$.

### `sm1.pickle`

The file holds a single python element of class `sklearn.ensemble.ExtraTreesClassifier` [[4]]. It is pre trained to estimate the number of components with single reactions represented by input mass loss rate data. It can be loaded via pickle[[5]] into Python.

Using `sm1.predict(X)` expects 1064 features as input with following properties:

|Feature|Description|Unit|
|------|-----------|----|
| 	1...266| Mass loss rate at $\beta_1$ 			| s^-1 |
| 267...532| Mass loss rate at $\beta_2$ 			| s^-1 | 
| 533...798| Mass loss rate at $\beta_3$ 			| s^-1 |
| 799...1064| Mass loss rate at $\beta_4$ 			| s^-1 |
| 799...1064| Mass loss rate at $\beta_4$ 			| s^-1 |


Corresponding $T$ is 20...550 °C with $\Delta T=2K$. Then, $t$ is $\frac{T}{\beta}$.

Output is a single integer of 1,2 or 3 as the number of components in the material represented by the TGA mass loss rate profile.

### `sm3_1r.pickle`

The file holds a single python element of class `sklearn.ensemble.ExtraTreesRegressor` [[6]]. It is pre trained to estimate the reaction kinetic parameters of materials consisting of one component represented by input mass loss rate data. It can be loaded via pickle[[5]] into Python.

Using `sm3_1r.predict(X)` expects 1067 features as input with following properties:

|Feature|Description|Unit|
|------|-----------|----|
| 	1...266| Mass loss rate at $\beta_1$ 			| s^-1 |
| 267...532| Mass loss rate at $\beta_2$ 			| s^-1 | 
| 533...798| Mass loss rate at $\beta_3$ 			| s^-1 |
| 799...1064| Mass loss rate at $\beta_4$ 			| s^-1 |
| 1065      | Initial fraction of component 1 ($Y_1=1$)			|      |
| 1066      | Initial fraction of component 2 ($Y_2=0$)			|      |
| 1067      | Initial fraction of component 3 ($Y_3=0$)			|      |

Corresponding $T$ is 20...550 °C with $\Delta T=2K$. Then, $t$ is $\frac{T}{\beta}$.

Output is $log(A_1)$ and $E_1$.


### `sm3_2r.pickle`

The file holds a single python element of class `sklearn.ensemble.ExtraTreesRegressor` [[6]]. It is pre trained to estimate the reaction kinetic parameters of materials consisting of two components represented by input mass loss rate data. It can be loaded via pickle[[5]] into Python.

Using `sm3_2r.predict(X)` expects 1067 features as input with following properties:

|Feature|Description|Unit|
|------|-----------|----|
| 	1...266| Mass loss rate at $\beta_1$ 			| s^-1 |
| 267...532| Mass loss rate at $\beta_2$ 			| s^-1 | 
| 533...798| Mass loss rate at $\beta_3$ 			| s^-1 |
| 799...1064| Mass loss rate at $\beta_4$ 			| s^-1 |
| 1065      | Initial fraction of component 1 ($Y_1$)			|      |
| 1066      | Initial fraction of component 2 ($Y_2$)			|      |
| 1067      | Initial fraction of component 3 ($Y_3$)			|      |

Corresponding $T$ is 20...550 °C with $\Delta T=2K$. Then, $t$ is $\frac{T}{\beta}$.

Output is $log(A_1)$, $log(A_2)$, $E_1$ and $E_2$.

### `sm3_3r.pickle`

The file holds a single python element of class `sklearn.ensemble.ExtraTreesRegressor` [[6]]. It is pre trained to estimate the reaction kinetic parameters of materials consisting of three components represented by input mass loss rate data. It can be loaded via pickle[[5]] into Python.

Using `sm3_3r.predict(X)` expects 1067 features as input with following properties:

|Feature|Description|Unit|
|------|-----------|----|
| 	1...266| Mass loss rate at $\beta_1$ 			| s^-1 |
| 267...532| Mass loss rate at $\beta_2$ 			| s^-1 | 
| 533...798| Mass loss rate at $\beta_3$ 			| s^-1 |
| 799...1064| Mass loss rate at $\beta_4$ 			| s^-1 |
| 1065      | Initial fraction of component 1 ($Y_1$)			|      |
| 1066      | Initial fraction of component 2 ($Y_2$)			|      |
| 1067      | Initial fraction of component 3 ($Y_3$)			|      |

Corresponding $T$ is 20...550 °C with $\Delta T=2K$. Then, $t$ is $\frac{T}{\beta}$.

Output is $log(A_1)$, $log(A_2)$, $log(A_3)$, $E_1$, $E_2$ and $E_3$.

### `prediction.csv`

This is the columns description of the output from `calculate.py`. The file has no header.

|Column|Symbol|Description|Unit|
|------|------|-----------|----|
| 1|$log(A_1)$| Pre-exponential factor for component 1			| s^-1 |
| 2|$log(A_2)$| Pre-exponential factor for component 2			| s^-1 |
| 3|$log(A_3)$| Pre-exponential factor for component 3			| s^-1 |
| 4|$E_1$| Activation energy for component 1				| J/mol|
| 5|$E_2$| Activation energy for component 2				| J/mol|
| 6|$E_3$| Activation energy for component 3				| J/mol|
| 7|$Y_1$|Fraction of component 1						| 1	   |
| 8|$Y_2$|Fraction of component 2						| 1	   |
| 9|$Y_3$|Fraction of component 3						| 1	   |

[1]: https://doi.org/
[2]: https://doi.org/10.5281/zenodo.6337389
[3]: https://doi.org/10.5281/zenodo.6346476
[4]: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html
[5]: https://docs.python.org/3/library/pickle.html
[6]: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesRegressor.html#sklearn.ensemble.ExtraTreesRegressor
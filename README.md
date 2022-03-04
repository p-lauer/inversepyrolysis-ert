# Inverse modelling pyrolization kinetics with ensemble learning methods - scripts
## ./generate_db/

The scripts `generate_1c.py`, `generate_2c.py` and `generate_3c.py` are used to generate the training data for the model. There is an individual script for 1, 2 and three components. The reaction kinetic parameters and component fractions are sampled randomly and then the mass loss rate for TGA experiments with four different constant heating rates are calculated. The scripts are intended to be used on multiple CPUs. Example data sets are available for download[[1]](1).

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

## ./build/

Scripts to generate the individual sub models

## ./test/

Script to test the complete model

## <a name="data-structure"></a> Data structure

### `labels*.csv`

no Header

The mass loss rates are calculated from the reaction kinetic parameters can be found at the same row number of the corresponding `features*` file.

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
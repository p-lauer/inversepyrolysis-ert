# Inverse modelling pyrolization kinetics with ensemble learning methods - scripts
## ./generate_db/

Scripts to generate the data sets used to train the models

- Parameters to set
- Output files
- Multiprocessor
- Examples available at [1].

The output of the scripts are as following. Description of the data structure is below.

|Filename | Description |
|---------|-------------|
|labels{$N_u$/1000}k_{$\beta_1$}_{$\beta_2$}_{$\beta_3$}_{$\beta_4$}_{$n$}_{$\Delta T$/s}.csv 				|generated labels  |
|features{$N_u$/1000}k_{$\beta_1$}_{$\beta_2$}_{$\beta_3$}_{$\beta_4$}_{$n$}_{$\Delta T$/s}_{$i_{cores}$}.csv |generated labels, output per core |
|labels{$N_u$/1000}k_{$\beta_1$}_{$\beta_2$}_{$\beta_3$}_{$\beta_4$}_{$n$}_{$\Delta T$/s}_{$i_{cores}$}.csv 	|generated labels, output per core  |

## ./build/

Scripts to generate the individual sub models

## ./test/

Script to test the complete model

## File structure

### labels*.csv

no Header

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

### features*.csv

|Column|Description|Unit|
|------|-----------|----|
| 	1...266| Mass loss rate at $\beta_1$ 			| s^-1 |
| 267...532| Mass loss rate at $\beta_2$ 			| s^-1 | 
| 533...798| Mass loss rate at $\beta_3$ 			| s^-1 |
| 799...1064| Mass loss rate at $\beta_4$ 			| s^-1 |

Corresponding $T$ is 20...550 with $\Delta T=2K$
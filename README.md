# Inverse modelling pyrolization kinetics with ensemble learning methods - scripts
## ./generate_db/

Scripts to generate the data sets used to train the models

## ./build/

Scripts to generate the individual sub models

## ./test/

Script to test the complete model

## File structure

### labels*.csv

no Header

Columns description:

|Column|Description|Unit|
|------|-----------|----|
| 1| Peak reaction rate for component 1 			| s^-1 |
| 2| Peak reaction rate for component 2 			| s^-1 |
| 3| Peak reaction rate for component 3 			| s^-1 |
| 4| Peak reaction temperature for component 1 		| °C   |
| 4| Peak reaction temperature for component 2 		| °C   |
| 6| Peak reaction temperature for component 3 		| °C   |
| 7| Fraction of component 1						| 1	   |
| 8| Fraction of component 2						| 1	   |
| 9| Fraction of component 3						| 1	   |
|10| Pre-exponential factor for component 1			| s^-1 |
|11| Pre-exponential factor for component 2			| s^-1 |
|12| Pre-exponential factor for component 3			| s^-1 |
|13| Activation energy for component 1				| J/mol|
|14| Activation energy for component 2				| J/mol|
|15| Activation energy for component 3				| J/mol|
|16| Fraction of component 1						| 1	   |
|17| Fraction of component 2						| 1	   |
|18| Fraction of component 3						| 1	   |

### features*.csv

|Column|Description|Unit|
|------|-----------|----|
| 	1... 266| Mass loss rate at $\beta_1$ 			| s^-1 |
| 267... 532| Mass loss rate at $\beta_2$ 			| s^-1 | 
| 533... 798| Mass loss rate at $\beta_3$ 			| s^-1 |
| 799...1064| Mass loss rate at $\beta_4$ 			| s^-1 |

Corresponding $T$ is 20...550 with $\Delta T=2K$
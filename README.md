https://github.gatech.edu/ytalmor3/7641.A2
==========================================

# OMSCS 7641 - Machine Learning

This repository is for work submitted in Spring 2019 for the
Machine Learning (7641) class offered by GATech via Udacity.

## Getting Started

This project was built with `Python 3.7.0`.
The environment may be installed as such:
```
virtualenv .venv --python ~/local/path/to/3.7.0/bin/python
echo  'export PYTHONPATH="../:."' >> ./.venv/bin/activate
source ./.venv/bin/activate
python -m pip install --requirement requirements.txt
```

If running on a Mac, also run:
```
sed -i -e 's/: macosx/: TkAgg/g' ./.venv/lib/python3.7/site-packages/matplotlib/mpl-data/matplotlibrc
```

### Prerequisites

All pip prerequisites are listed in **requirements.txt**.
They may be installed with **pip** using:
```
pip install -r requirements.txt
```

### Environment
The environment may be activated with:
```
source ./.venv/bin/activate
```
It may be deactivated with: `deactivate`


## Abridged directory structure
```
├── algorithms
│   ├── genetic_algorithm.py
│   ├── mimic.py
│   ├── random_hill_climb.py
│   └── simulated_annealing.py
├── data
│   └── contraceptive.csv
├── neural_networks
│   ├── bar_chart.py
│   ├── neural_networks.py
│   └── nn_compare.py
├── plots
│   ├── complex_time
│   ├── fit_iter
│   │   └── *.png
│   ├── fit_time
│   │   └── *.png
│   └── neural_network
│       └── *.png
├── puzzles
│   ├── bar_chart.py
│   ├── k_coloring.py
│   ├── knapsack.py
│   ├── plotter.py
│   ├── puzzle_solver.py
│   ├── run_all.py
│   └── travelling_salesman.py
└── requirements.txt

```

## Instructions
Run `run_all.py` to run all the puzzle experiments.
Run `neural_networks.py` to run all neural network optimization experiments.
Run `bar_chart.py` to visualize.
All generated visualizations are saved under the `plots` directory.


## Project 1
A survey project exploring a number of randomized optimization algorithms on the contraceptives datasets.


## Authors
* **Yaniv Talmor**

# fsam

`fsam` is a Python module to perform feature selection in additive models. It is assumed
that the smooth components to be estimated are defined through a reduced-rank basis
(B−splines) and fitted via a penalized splines approach (P−splines). Our variable
selection approach is based on selecting the best subset of features of a given size,
taking into account that each of the features can enter in the model as linear,
non-linear, or both. This cardinality-constrained problem is stated as a mixed-integer
quadratic programming (MIQP) model. We develop a framework to compute tight bounds for
the regression coefficients to the case of additive models. A heuristic approach based
on the large neighborhood search metaheuristic and that exploits the exact formulation
of the problem is developed, thus yielding a _matheuristic_. Moreover, a method to build
a warm-start solution is also developed by combining additive models and group lasso.

Solving the optimization problems is done using [GUROBI](https://www.gurobi.com/)
optimization software.

## Project structure

The current version of the project is structured as follows:

* **fsam**: the main directory of the project, which consist of:
  * **fsam_fit**: contains the feature selection algorithm.
  * **penalized_group_lasso**: contains our warm-start approach.
  * **sop**: contains the methodology for estimating the smoothing parameters.
* **data**: a folder containing CSV files used in the real data numerical
  experiments.
* **examples**: a directory containing multiple numerical experiments.
* **img**: contains some images.
* **tests**: a folder including tests for the main methods of the project.

## Package dependencies

`fsam` mainly depends on the following packages:

* [cpsplines](https://github.com/ManuelNavarroGarcia/cpsplines).
* [gurobipy](https://www.gurobi.com). **License Required**
* [matplotlib](https://matplotlib.org/).
* [numpy](https://numpy.org/).
* [pandas](https://pandas.pydata.org/).
* [scikit-learn](https://scikit-learn.org/).
* [scipy](https://www.scipy.org/).
* [statsmodels](https://www.statsmodels.org/).
* [tqdm](https://tqdm.github.io/).
* [typer](https://typer.tiangolo.com/).

GUROBI requires a license to be used. For research or educational purposes, a free
yearly and renewable [academic license](https://www.gurobi.com/academia/academic-program-and-licenses/) is offered by the
company.

## Installation

1. To clone the repository on your own device, use

```{bash}
git clone https://github.com/ManuelNavarroGarcia/fsam.git
cd fsam
```

2. To install the dependencies, there are two options according to your
   installation preferences:

* Create and activate a virtual environment with `conda` (recommended)

```{bash}
conda env create -f env.yml
conda activate fsam
```

* Install the setuptools dependencies via `pip`

```{bash}
pip install -r requirements.txt
pip install -e .[dev]
```

3. If neccessary, add version requirements to existing dependencies or add new
   ones on `setup.py`. Then, update `requirements.txt` file using

```{bash}
pip-compile --extra dev > requirements.txt
```

and update the environment with `pip-sync`. Afterwards, the command

```{bash}
pip install -e .[dev]
```

needs to be executed.

## Testing

The repository contains a folder with unit tests to guarantee the main methods
meets their design and behave as intended. To launch the test suite, it is
enough to enter `pytest`. If only one test file wants to be run, the syntax is
given by

```{bash}
pytest tests/test_<file_name>.py
```

## Contributing

Contributions to the repository are welcomed! Regardless of whether it is a
small fix on the documentation or a notable feature to be included, I encourage
you to develop your ideas and make this project greater. Even suggestions about
the code structure are highly appreciated. Furthermore, users participating on
these submissions will figure as contributors on this main page of the
repository.

There are many ways you can contribute on this repository:

* [Discussions](https://github.com/ManuelNavarroGarcia/fsam/discussions).
  To ask questions you are wondering about or share ideas, you can enter an
  existing discussion or open a new one.

* [Issues](https://github.com/ManuelNavarroGarcia/fsam/issues). If you
  detect a bug or you want to propose an enhancement of the current version of
  the code, a issue with reproducible code and/or a detailed description is
  highly appreciated.

* [Pull Requests](https://github.com/ManuelNavarroGarcia/fsam/pulls). If
  you feel I am missing an important feature, either in the code or in the
  documentation, I encourage you to start a pull request developing this idea.
  Nevertheless, before starting any major new feature work, I suggest you to
  open an issue or start a discussion describing what you are planning to do.
  Recall that, before starting a pull request, all unit test must pass on your
  local repository.

## Contact Information and Citation

If you have encountered any problem or doubt while using `fsam`, please feel free to let
me know by sending me an email:

* Name: Manuel Navarro García (he/his)
* Email: <manuelnavarrogithub@gmail.com>

## Acknowledgements

Throughout the developing of this project I have received strong support from
various individuals. I would like to thank my PhD supervisors, Professor [Vanesa
Guerrero](https://github.com/vanesaguerrero) and Professor [María
Durbán](https://github.com/MariaDurban), whose insightful comments and
invaluable expertise has given way to many of the current functionalities of the
repository.

This publication is part of the project/grant PDC2022-133359-I00 funded by MCIN/AEI/10.13039/501100011033 and by the European Union “NextGenerationEU/PRTR”.

Esta publicación forma parte del proyecto PDC2022-133359-I00, financiado por MCIN/AEI/10.13039/501100011033 y por la Unión Europea “NextGenerationEU/PRTR”.

![Acknowledgment](./img/project_funding.png)
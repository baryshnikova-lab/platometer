INTRODUCTION
============

Platometer is a simple image-processing tool for quantifying colony sizes from arrayed growth experiments in yeast or bacteria. Typical applications include phenotypic screens of mutant collections and synthetic genetic array (SGA) experiments.

**WARNING. This package is still in development. Please use caution.**


GETTING STARTED
===============

Platometer requires Python 3 and a set of packages listed in `requirements.txt`. We recommend setting up a virtual environment and installing all the required packages via pip:

```
cd platometer/
virtualenv -p python3.6 platometer_env
source platometer_env/bin/activate
pip install -r requirements.txt
```

After the installation is complete, it is useful to run a "hello world" Platometer analysis using the Jupyter notebook at `examples/Usage_examples.ipynb`. 

To do so, from within the platometer_env environment install a new ipython kernel:

```
ipython kernel install --user --name=platometer
```

Then start jupyter, open `examples/Usage_examples.ipynb` and select the safepy kernel.

```
jupyter-notebook
```

HELP
====

Please direct all questions/comments to Anastasia Baryshnikova (<abaryshnikova@calicolabs.com>).

The main repository for this code is at <https://github.com/baryshnikova-lab/platometer>. Please subscribe to the repository to receive live updates about new code releases and bug reports.

# lug-nuts

A lightweight, (p)lug-and-play wrapper around the No U-Turn Sampler implementation found in `numpyro`. Just provide functions for your likelihood and its gradient (not necessarily written with JAX), a configuration file describing your model parameters, and let it run.

Key features: wraps non-JAX likelihoods inside a JAX `host_callback` so it can be used with code that expects JAX inputs (e.g., NumPyro samplers). A couple of built-in coordinate transformations to aid in sampling. Model configuration .json files to help organize parameter names and prior bounds. Ability to fix an arbitrary number of parameters in the given model and sample the remaining subset.

Note: This probably isn't the optimal way to work non-JAX likelihood functions into NumPyro. It definitely takes a hit in performance. But it works for now.

## Installation
Installation can be done simply via pip.
```
pip install https://www.github.com/gabefreedman/lug-nuts.git
```

## Model configuration files
Model setup is run through a configuration file which can either be a Python dict defined in script or read from a .json file. It contains information such as the parameter names, prior bounds, prior type, optional transforms, and whether to hold a parameter fixed or let it vary. A sample configuration is shown below, and other examples are included under `examples/configs`. The custom likelihood function must be supplied within the script or imported. For the example configuration files the `ll_func` key references a predefined set of likelihood functions included with the examples.
```
{
    "param_info": [
{
    "name": "A",
    "lower": 1e1,
    "upper": 1e5,
    "transform": "log",
    "prior": "uniform_log"
},
{
    "name": "omega",
    "lower": 1.0,
    "upper": 20.0,
    "transform": "identity",
    "prior": "uniform"
},
{
    "name": "phi",
    "lower": -3.1415,
    "upper": 3.1415,
    "transform": "identity",
    "prior": "uniform"
}
],
"free_mask": [true, false, true],
"fixed_vals": [null, 5.1, null],
"truths": [3.7e2, 5.1, 1.5708],
"ll_func": "(supplied within script)"
}
```

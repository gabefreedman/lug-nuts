#!/usr/bin/env python3
"""
NUTS inference for models with likelihood functions that may not be JAX-compatible.
"""

from typing import List, Dict, Any, Tuple, Callable, Sequence
import json

import numpy as np
import jax
import jax.numpy as jnp
import h5py
import numpyro
from numpyro.distributions import Normal
from numpyro.infer import MCMC, NUTS, init_to_value


# ----------------- Coordinate mappings -----------------
# All parameters are mapped from a constrained parameter space bounded by the set
# of (currently just uniform) priors to an unconstrained space used by the NUTS
# routine where all transformed parameters are sampled using normal distributions.
# An additional transformation can be invoked (``log`` vs ``identity``) in the config
# file to accomodate sampling any subset of variables in log space rather than linear.
# For syntax purposes, here ``z`` refers to a parameter vector in the unconstrained
# space and ``x`` refers to a vector in the physical bounded space.


def make_index_maps(free_mask: List[bool]) -> Tuple[List[int], Dict[int, int]]:
    """Build index mappings for free parameters from the full parameter vector
    (length ``n``) to a reduced parameter vector (length ``m``) of just the free parameters.

    Args:
        free_mask: Boolean list of length `n_params` where True indicates the
            parameter is sampled by the MCMC and False means the parameter is
            held fixed.

    Returns:
        A tuple ``(free_indices, index_in_free)`` where ``free_indices`` is a
        list of original indices that are free and ``index_in_free`` maps the
        original parameter index to its position in the reduced free-vector.
    """

    free_indices = [i for i, v in enumerate(free_mask) if v]
    index_in_free = {i: idx for idx, i in enumerate(free_indices)}
    return free_indices, index_in_free


def z_to_x_and_dx_dz(
    z_free: jnp.ndarray,
    param_info: List[Dict[str, Any]],
    free_mask: List[bool],
    fixed_values: List[Any],
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Map unconstrained variables to physical parameters and derivates for transformation.

    This function applies the per-parameter transform (either ``identity`` or ``log``)
    and computes the derivative dx/dz for all varying parameters.

    Args:
        z_free: JAX array of shape (m,) containing unconstrained free parameters.
        param_info: List of parameter metadata dicts.
        free_mask: Boolean list indicating which parameters are varying.
        fixed_values: List of length n_params containing values for fixed parameters.

    Returns:
        A tuple ``(x_full, dx_dz_free)`` where ``x_full`` is a JAX array of
        length n with physical parameter values and ``dx_dz_free`` is a JAX
        array of length m containing the derivatives dx_i/dz_i for the free
        parameters (in the same order as ``z_free``).
    """

    _, index_mapping = make_index_maps(free_mask)
    xs = []
    dx_dzs = []

    for i, param in enumerate(param_info):
        lower = float(param["lower"])
        upper = float(param["upper"])
        transform = param.get("transform", "identity")

        if not free_mask[i]:
            # if parameter is fixed, use provided value
            val = fixed_values[i]
            if val is None:
                raise ValueError(
                    f"Parameter {param['name']} is fixed but fixed_values not provided."
                )
            xs.append(jnp.array(val))
            # no dx/dz contribution for fixed param
        else:
            z_value = z_free[index_mapping[i]]
            s = jax.nn.sigmoid(z_value)
            if transform == "identity":
                x = lower + (upper - lower) * s
                dx_dz = (upper - lower) * s * (1.0 - s)
            elif transform == "log":
                if lower <= 0:
                    raise ValueError(
                        f"Parameter {param['name']} has 'log' transform but lower bound <= 0."
                    )
                log_lw = jnp.log10(lower)
                log_up = jnp.log10(upper)
                diff = log_up - log_lw
                u = log_lw + diff * s  # u in [log L, log U]
                x = 10**u
                dx_dz = x * jnp.log(10) * diff * s * (1.0 - s)
            else:
                raise ValueError(
                    f"Undefined transform '{transform}' for parameter {param['name']}."
                )
            xs.append(x)
            dx_dzs.append(dx_dz)

    x_vec = jnp.stack(xs)
    dx_dz_vec = jnp.stack(dx_dzs)
    return x_vec, dx_dz_vec


# ----------------- Wrapper for non-JAX external likelihood and gradient -----------------
# This is a workaround for model likelihood functions that are not readily available with JAX
# or cannot be easily rewritten to use JAX. It invokes a host pure_callback with a custom defined
# vector-Jacobian product (VJP) to allow it to mimic a regular JAX function that can be accessed
# with autodiff. This is essential for communicating with modern NUTS implementations such as in
# numpyro. The trade-off is, as one might expect, that the excessive host callbacks bring an
# increase in computation time.


def _make_host_call_loglik(
    x_vec: jnp.ndarray, ll_fn: Callable[..., float]
) -> jnp.ndarray:
    """Use JAX's pure_callback to invoke a host-side log-likelihood function.

    The likelihood function should accept the physical parameters as
    separate positional arguments (i.e. ll_fn(a, b, c...)).

    Args:
        x_vec: JAX array of shape (n,) containing physical parameters.
        ll_fn: Python callable returning a scalar float log-likelihood.

    Returns:
        The log-likelihood as a JAX scalar.
    """

    def _loglike(x_np):
        # x_np is a numpy array
        ll_val = ll_fn(*x_np)
        return np.array(ll_val, dtype=np.float64)

    # Need to define the result shape, a jax.ShapeDtypeStruct describing a scalar float64
    ret_shape = jax.ShapeDtypeStruct((), jnp.float64)
    ret = jax.pure_callback(_loglike, ret_shape, x_vec)
    return jnp.asarray(ret).reshape(())


def _make_host_call_grad(
    x_vec: jnp.ndarray, grad_fn: Callable[..., Sequence[float]]
) -> jnp.ndarray:
    """Use JAX's pure_callback to invoke a host-side log-likelihood gradient function.

    Args:
        x_full_jax: JAX array of shape (n_params,) containing physical parameters.
        grad_fn: Python callable returning a sequence of length n.

    Returns:
        A JAX array of shape (n_params,) containing the log-likelihood gradient.
    """

    n_params = x_vec.shape[0]

    def _grad(x_np):
        grd = grad_fn(*x_np)
        grd = np.asarray(grd, dtype=np.float64)
        return grd

    ret_shape = jax.ShapeDtypeStruct((n_params,), jnp.float64)
    ret = jax.pure_callback(_grad, ret_shape, x_vec)
    return jnp.asarray(ret)


def make_loglike_z(
    param_info: List[Dict[str, Any]],
    free_mask: List[bool],
    fixed_values: List[Any],
    ll_fn: Callable[..., float],
    grad_fn: Callable[..., Sequence[float]],
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Make a JAX-callable log-likelihood function defined on the unconstrained parameter space.
    The function comes with a custom VJP so that JAX can compute gradients.

    Args:
        param_info: List of parameter metadata dicts.
        free_mask: Boolean list indicating which parameters are varying.
        fixed_values: List of length n_params containing values for fixed parameters.
        ll_fn: User-provided log-likelihood function.
        grad_fn: User-provided log-likelihood gradient function.

    Returns:
        A version of the log-likelihood function that is differentiable through JAX.
    """

    @jax.custom_vjp
    def loglike_z(z_free: jnp.ndarray) -> jnp.ndarray:
        x_full, _ = z_to_x_and_dx_dz(z_free, param_info, free_mask, fixed_values)
        ll = _make_host_call_loglik(x_full, ll_fn)
        return ll

    def loglike_z_fwd(z_free):
        x_full, dx_dz_free = z_to_x_and_dx_dz(
            z_free, param_info, free_mask, fixed_values
        )
        ll = _make_host_call_loglik(x_full, ll_fn)
        return ll, (x_full, dx_dz_free)

    def loglike_z_bwd(res, g):
        x_vec, dx_dz = res
        grad_x = _make_host_call_grad(x_vec, grad_fn)
        free_inds, _ = make_index_maps(list(free_mask))
        grad_x = grad_x[jnp.array(free_inds, dtype=jnp.int32)]
        grad_z = g * (dx_dz * grad_x)
        return (grad_z,)

    # attach custom_vjp rules
    loglike_z.defvjp(loglike_z_fwd, loglike_z_bwd)

    return loglike_z


def log_prior_x(x_vec: jnp.ndarray, param_info: List[Dict[str, Any]]) -> jnp.ndarray:
    """Log-prior function in the physical parameter space.

    Supported per-parameter priors:
    - "uniform": uniform in x between [lower, upper]
    - "uniform_log": uniform in log10(x) between log(lower) and log(upper)

    Args:
        x_vec: Array of physical parameter values.
        param_info: List of parameter metadata dicts.

    Returns:
        Log-prior value.
    """

    logprs = []
    for i, param in enumerate(param_info):
        lower = float(param["lower"])
        upper = float(param["upper"])
        prior_type = param.get("prior", "uniform")
        x_i = x_vec[i]
        if prior_type == "uniform":
            logprs.append(-jnp.log10(upper - lower))
        elif prior_type == "uniform_log":
            logprs.append(
                -jnp.log(jnp.log(upper) - jnp.log(lower))
                - jnp.log(x_i)
                - jnp.log(jnp.log(10))
            )
        else:
            raise ValueError(
                f"Unsupported prior type '{prior_type}' for parameter {param['name']}."
            )
    return jnp.sum(jnp.stack(logprs))


# ----------------- Potential function for NUTS -----------------
def make_potential_fn(
    param_info: List[Dict[str, Any]],
    free_mask: List[bool],
    fixed_values: List[Any],
    ll_fn: Callable[..., float],
    grad_fn: Callable[..., Sequence[float]],
) -> Callable[[jnp.ndarray], jnp.ndarray]:
    """Make the JAX-callable potential function for the model on the unconstrained
    parameter space for use in NUTS.

    Args:
        param_info: List of parameter metadata dicts.
        free_mask: Boolean list indicating which parameters are varying.
        fixed_values: List of length n_params containing values for fixed parameters.
        ll_fn: User-provided log-likelihood function.
        grad_fn: User-provided log-likelihood gradient function.

    Returns:
        A JAX-differentiable potential function for NUTS.
    """
    loglike_of_z = make_loglike_z(param_info, free_mask, fixed_values, ll_fn, grad_fn)

    def potential(z_vec: jnp.ndarray) -> jnp.ndarray:
        x_full, dx_dz = z_to_x_and_dx_dz(z_vec, param_info, free_mask, fixed_values)
        logl = loglike_of_z(z_vec)

        logpr = log_prior_x(x_full, param_info)
        # compute Jacobian
        logjac = jnp.sum(jnp.log(jnp.abs(dx_dz)))

        log_posterior = logl + logpr + logjac
        return log_posterior

    return jax.jit(potential)


def make_numpyro_model(
    param_info: List[Dict[str, Any]],
    free_mask: List[bool],
    fixed_values: List[Any],
    ll_fn: Callable[..., float],
    grad_fn: Callable[..., Sequence[float]],
) -> Callable[[], None]:
    """Create a numpyro model for a given parameter space and likelihood/gradient functions.

    Args:
        param_info: List of parameter metadata dicts.
        free_mask: Boolean list indicating which parameters are varying.
        fixed_values: List of length n_params containing values for fixed parameters.
        ll_fn: User-provided log-likelihood function.
        grad_fn: User-provided log-likelihood gradient function.

    Returns:
        A callable numpyro model.
    """
    n_free = sum(free_mask)
    logdensity_fn = make_potential_fn(
        param_info, free_mask, fixed_values, ll_fn, grad_fn
    )

    def model():
        # Sample unconstrained free parameters
        z = numpyro.sample("z", Normal(jnp.zeros(n_free), 10.0))
        numpyro.factor("likelihood_and_jac", logdensity_fn(z))

    return model


def run_numpyro_nuts(
    param_info: List[Dict[str, Any]],
    free_mask: List[bool],
    fixed_values: List[Any],
    ll_fn: Callable[..., float],
    grad_fn: Callable[..., Sequence[float]],
    n_adapt: int = 1000,
    n_samples: int = 3000,
    n_chains: int = 1,
    seed: int = 1234,
    target_accept: float = 0.8,
    hdf5_out: str = "chains.h5",
) -> None:
    """Setup and run NUTS with numpyro and save results. Takes user-provided
    log-likelihood and log-likelihood gradient functions.

    Args:
        param_info: List of parameter metadata dicts.
        free_mask: Boolean list indicating which parameters are varying.
        fixed_values: List of length n_params containing values for fixed parameters.
        ll_fn: Host log-likelihood callable.
        grad_fn: Host gradient callable.
        num_warmup: Number of NUTS adaptation steps.
        num_samples: Number of samples in chain.
        num_chains: Number of independent chains.
        rng_seed: Random seed.
        target_accept: Target NUTS acceptance probability.
        hdf5_out: Output filename for HDF5 file.

    Returns:
        None. Samples are saved to HDF5 file.
    """

    free_indices, _ = make_index_maps(free_mask)
    m = len(free_indices)
    n_params = len(param_info)

    z0 = jnp.zeros((m,), dtype=jnp.float64)
    init_strategy = init_to_value(values={"z": z0})
    rng_key = jax.random.PRNGKey(seed)

    model_fn = make_numpyro_model(param_info, free_mask, fixed_values, ll_fn, grad_fn)

    # Build NUTS kernel
    kernel = NUTS(
        model_fn, init_strategy=init_strategy, target_accept_prob=target_accept
    )
    mcmc = MCMC(
        kernel,
        num_warmup=n_adapt,
        num_samples=n_samples,
        num_chains=n_chains,
        progress_bar=True,
    )

    # Run NUTS
    mcmc.run(rng_key)

    res = mcmc.get_samples(group_by_chain=True)
    for _, val in res.items():
        arr = np.asarray(val)
        if arr.ndim == 3 and arr.shape[-1] == m:
            samples_z = arr
            break
        # if m==1 the shape might be (n_chains, n_samples)
        if m == 1 and arr.ndim == 2:
            samples_z = np.expand_dims(arr, -1)
            break

    n_chains, n_samples, _ = samples_z.shape

    # Transform back to physical parameter space and save to HDF5
    with h5py.File(hdf5_out, "w") as f:
        f.attrs["n_params"] = n_params
        f.attrs["param_info"] = json.dumps(param_info)
        f.attrs["free_mask"] = json.dumps(free_mask)
        f.attrs["fixed_values"] = json.dumps(
            [None if v is None else float(v) for v in fixed_values]
        )
        ds = f.create_dataset(
            "chains", shape=(n_chains, n_samples, n_params), dtype=np.float64
        )
        for c in range(n_chains):
            for s in range(n_samples):
                z_vec = jnp.asarray(samples_z[c, s, :])
                x_vec, _ = z_to_x_and_dx_dz(z_vec, param_info, free_mask, fixed_values)
                ds[c, s, :] = np.asarray(x_vec)
    return

from dataclasses import dataclass
import jax.numpy as jnp


from jax import tree_util                     # add this

@tree_util.register_dataclass 
@dataclass
class MeanVarState:
    n:    jnp.ndarray          # scalar ()
    mean: jnp.ndarray          # (feat,)
    m2:   jnp.ndarray          # (feat,)  sum of squared diffs

def meanvar_init(shape):
    return MeanVarState(jnp.zeros((), jnp.int32),
                        jnp.zeros(shape, jnp.float32),
                        jnp.zeros(shape, jnp.float32))

def meanvar_update(state, x):
    x = jnp.where(jnp.isnan(x) | jnp.isinf(x), state.mean, x)
    x_mean = jnp.mean(x, axis=0)          # collapses batch dim if present

    n      = state.n + 1
    delta  = x_mean - state.mean
    mean   = state.mean + delta / n
    delta2 = x_mean - mean
    m2     = state.m2 + delta * delta2
    return MeanVarState(n, mean, m2)

def meanvar_normalize(state: MeanVarState, x: jnp.ndarray, eps=1e-8):
    var = state.m2 / jnp.maximum(1, state.n)
    std = jnp.sqrt(var + eps)
    return (x - state.mean) / std
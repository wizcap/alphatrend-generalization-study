"""Fixed5-4-1 policy inference, independent from optimizers and accounting."""
import numpy as np

INPUTS, HIDDEN, PARAMETERS = 5, 4, 29
CONNECTIONS = 24


def mlp(weights, x):
    """Population x observations tanh outputs, with29 bounded parameters."""
    w=np.atleast_2d(np.asarray(weights,float))
    x=np.asarray(x,float)
    if w.shape[1] != PARAMETERS or x.ndim != 2 or x.shape[1] != INPUTS:
        raise ValueError('Expected29parameters and5inputs')
    hidden=np.tanh(np.einsum('nd,pdh->pnh',x,w[:,:20].reshape(-1,5,4))+w[:,None,20:24])
    return np.tanh(np.einsum('pnh,ph->pn',hidden,w[:,24:28])+w[:,28,None])


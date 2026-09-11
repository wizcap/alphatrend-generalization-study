"""Dimension-specific linear and four-hidden-node CMA-ES policies."""
import cma
import numpy as np


def dimensions(kind, n):
    if n < 1:
        raise ValueError('Positive feature dimension required')
    if kind == 'linear':
        return n+1, n
    if kind == 'mlp':
        return 4*n+9, 4*n+4
    raise ValueError('Unknown fixed architecture')


def forward(weights, x, kind):
    w = np.atleast_2d(np.asarray(weights, float))
    n = x.shape[1]
    if w.shape[1] != dimensions(kind, n)[0]:
        raise ValueError('Policy weight dimension mismatch')
    if kind == 'linear':
        return np.tanh(np.einsum('pd,nd->pn', w[:, :n], x)+w[:, -1, None])
    hidden = np.tanh(np.einsum('nd,pdh->pnh', x, w[:, :4*n].reshape(-1, n, 4))+w[:, None, 4*n:4*n+4])
    return np.tanh(np.einsum('pnh,ph->pn', hidden, w[:, 4*n+4:4*n+8])+w[:, -1, None])


def fit(evaluate, n, seed, population, generations):
    optimizer = cma.CMAEvolutionStrategy(np.zeros(n), .5, {'seed': seed,
        'popsize': population, 'bounds': [-5., 5.], 'BoundaryHandler': cma.BoundTransform,
        'verbose': -9, 'verb_log': 0,
        'maxiter': generations})
    history, best, best_fitness = [], None, -np.inf
    for generation in range(generations):
        asked = optimizer.ask()
        fitness = np.asarray(evaluate(np.asarray(asked)), float)
        if fitness.shape != (population,) or not np.isfinite(fitness).all():
            raise ValueError('Invalid population fitness')
        index = int(np.argmax(fitness))
        if fitness[index] > best_fitness:
            best, best_fitness = np.array(asked[index], copy=True), float(fitness[index])
        if np.max(np.abs(asked)) > 5+1e-12:
            raise ValueError('CMA supplied out-of-bound evaluated weights')
        optimizer.tell(asked, (-fitness).tolist())
        history.append({'generation': generation+1, 'best_fitness': float(fitness.max()),
            'mean_fitness': float(fitness.mean()), 'sigma': float(optimizer.sigma),
            'candidate_evaluations': len(asked)})
    return best, best_fitness, history

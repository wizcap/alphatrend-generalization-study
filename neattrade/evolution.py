"""Fixed-budget optimizer adapters with no market or execution dependencies."""
import cma
import numpy as np

from .policies import PARAMETERS


def fit_cma(evaluate, seed, population, generations):
    """Fixed-budget ask/tell; never extend search based on market outcomes."""
    optimizer=cma.CMAEvolutionStrategy(np.zeros(PARAMETERS),.5,{
        'seed':seed,'popsize':population,'bounds':[-5.,5.],
        'verbose':-9,'verb_log':0,'maxiter':generations})
    records=[]
    for generation in range(generations):
        asked=optimizer.ask()
        fitness=np.asarray(evaluate(np.asarray(asked)),float)
        if fitness.shape != (population,) or not np.isfinite(fitness).all():
            raise ValueError('Invalid population fitness')
        optimizer.tell(asked,(-fitness).tolist())
        records.append({'generation':generation+1,'best_fitness':float(fitness.max()),
                        'mean_fitness':float(fitness.mean()),'sigma':float(optimizer.sigma),
                        'candidate_evaluations':len(asked)})
    return np.asarray(optimizer.result.xbest),float(-optimizer.result.fbest),records

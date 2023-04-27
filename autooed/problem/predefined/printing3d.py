'''
3D DLP printing problem suite.
'''

import numpy as np
from pymoo.factory import get_reference_directions
from pymoo.problems.util import load_pareto_front_from_file

from autooed.problem.problem import Problem

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier


class printing3d(Problem):

    config = {
        'type': 'integer',
        'n_var': 2,
        'n_obj': 2,
        'n_constr': 1,
        'var_lb': [0, 0],
        'var_ub': [100, 100]
    }

    def __init__(self):
        super().__init__()
        self.k = self.n_var - self.n_obj + 1
    
    def obj_func(self, x_):
        f = []

        for i in range(0, self.n_obj):
            _f = float (input (
            "ratios A-B {} sum {} Enter objective {}: ".
             format(np.round(x, 4), np.sum(np.round(x, 4)), i)))
            _f_ = -_f
            f.append(_f)
          

        f = np.array(f)
        return f

    def evaluate_objective(self, x_, alpha=1):
        f = []

        for i in range(0, self.n_obj):
            
            _f = float (input (
            "ratios A-B {} sum {} Enter objective {}: ".
             format(np.round(x_, 4), np.sum(np.round(x_, 4)), i)))
            _f = -_f
            f.append(_f)

        f = np.array(f)
        return f
    
    def evaluate_constraint(self, x_):
        x1, x2 = x_[0], x_[1]
        g1 = x1 + x2 - 100
        x_ = x_.reshape(1, -1)
        #print ('Printability accuracy on all data', RF_print.score(X_, Y))
        #print ('Tg accuracy on all data group 1 in range of [{}, {}] is: {}'.format(Tg_min, Tg_max, RF_Tg.score(X_, Tg_group)))
        return g1


class printing3d_dlp(printing3d):
    def _calc_pareto_front(self):
        ref_kwargs = dict(n_points=100) if self.n_obj == 2 else dict(n_partitions=15)
        ref_dirs = get_reference_directions('das-dennis', n_dim=self.n_obj, **ref_kwargs)
        return 0.5 * ref_dirs
   
    def evaluate_objective(self, x_, alpha=1):
        f = []
        objectives = ['Strength_Mpa', 'Toughness_MJ_m3']
        for i in range(0, self.n_obj):
            while True:
                try:
                    _f = float (input (
                            "ratios A-B {} sum {} Enter objective {}: ".
                            format(np.round(x_, 2), np.sum(np.round(x_, 2)), objectives[i])))
                except ValueError:
                    print ('the objective {} was not valid, try again'.format(objectives[i]))
                    continue
                else:
                    break
            _f = -_f
            f.append(_f)

        f = np.array(f)
        return f

    def evaluate_constraint(self, x_):
        x1, x2 = x_[0], x_[1]
        g1 = x1 + x2 - 100
        return g1


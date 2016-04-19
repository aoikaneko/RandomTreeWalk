# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 21:12:40 2016

@author: owner
"""

import time
import numbers
import warnings
import numpy as np
from collections import Sized
from sklearn.base import clone
from sklearn.utils import safe_indexing
from sklearn.metrics.scorer import check_scoring
from sklearn.cross_validation import KFold, FitFailedWarning, _index_param_value
from sklearn.utils.validation import _num_samples
from sklearn.grid_search import GridSearchCV, ParameterGrid, _CVScoreTuple
from sklearn.externals.joblib import Parallel, delayed, logger

def _check_cv(cv, n_samples):
    if cv is None:
        cv = 3
    if isinstance(cv, numbers.Integral):
        cv = KFold(n_samples, cv)
        
    return cv
    
def _safe_split(depthmaps, offset_points_projected, direction_vectors, true_joints, indices):
    depth_subset = safe_indexing(depthmaps, indices)
    offsets_subset = safe_indexing(offset_points_projected, indices)
    directions_subset = safe_indexing(direction_vectors, indices)
    truths_subset = safe_indexing(true_joints, indices)
    
    return depth_subset, offsets_subset, directions_subset, truths_subset
    
def _score(estimator, depth_test, truths_test, scorer):
    
    score = scorer(estimator, depth_test, truths_test)
    if not isinstance(score, numbers.Number):
        raise ValueError("scoring must return a number, got %s (%s) instead."
                     % (str(score), type(score)))
                    
    return score
    
def _fit_and_score(estimator, depthmaps, offset_points_projected, direction_vectors, true_joints, scorer, train, test, verbose, parameters, fit_params, return_train_score=False, return_parameters=False, error_score='raise'):
    
    if verbose > 1:
        if parameters is None:
            msg = "no parameters to be set"
        else:
            msg = '%s' % (', '.join('%s=%s' % (k, v) for k, v in parameters.items()))
        print("[CV] %s %s" % (msg, (64 - len(msg)) * '.'))
        
    # Adjust length of sample weights
    fit_params = fit_params if fit_params is not None else {}
    fit_params = dict([(k, _index_param_value(depthmaps, v, train))
                        for k, v in fit_params.items()])
                            
    if parameters is not None:
        estimator.set_params(**parameters)

    start_time = time.time()

    depth_train, offsets_train, directions_train, truths_train = _safe_split(depthmaps, offset_points_projected, direction_vectors, true_joints, train)
    depth_test, offsets_test, directions_test, truths_test = _safe_split(depthmaps, offset_points_projected, direction_vectors, true_joints, test)
    
    try:
        estimator.fit(depth_train, offsets_train, directions_train, **fit_params)
        
    except Exception as e:
        if error_score == 'raise':
            raise
        elif isinstance(error_score, numbers.Number):
            test_score = error_score
            if return_train_score:
                train_score = error_score
            warnings.warn("Classifier fit failed. The score on this train-test"
                          " partition for these parameters will be set to %f. "
                          "Details: \n%r" % (error_score, e), FitFailedWarning)
        else:
            raise ValueError("error_score must be the string 'raise' or a"
                             " numeric value. (Hint: if using 'raise', please"
                             " make sure that it has been spelled correctly.)"
                             )
    
    else:
        test_score = _score(estimator, depth_test, truths_test, scorer)
        if return_train_score:
            train_score = _score(estimator, depth_train, truths_train, scorer)
        
    scoring_time = time.time() - start_time
    
    if verbose > 2:
        msg += ", score=%f" % test_score
    if verbose > 1:
        end_msg = "%s -%s" % (msg, logger.short_format_time(scoring_time))
        print("[CV] %s %s" % ((64 - len(end_msg)) * '.', end_msg))

    ret = [train_score] if return_train_score else []
    ret.extend([test_score, _num_samples(depth_test), scoring_time])
    if return_parameters:
        ret.append(parameters)
    return ret

class GridSearchCVTree(GridSearchCV):
    
    def __init__(self, estimator, param_grid, scoring=None, fit_params=None,
                 n_jobs=1, iid=True, refit=True, cv=None, verbose=0,
                 pre_dispatch='2*n_jobs', error_score='raise'):
        
        super(GridSearchCVTree, self).__init__(
            estimator, param_grid, scoring, fit_params, n_jobs, iid,
            refit, cv, verbose, pre_dispatch, error_score)
    
    def _fit(self, depthmaps, offset_points_projected, direction_vectors, true_joints, parameter_iterable):

        cv = self.cv
        self.scorer_ = check_scoring(self.estimator, scoring=self.scoring)

        n_samples = _num_samples(depthmaps)
        
        if _num_samples(offset_points_projected) != n_samples:
            raise ValueError('offset_points_projected has a different number '
                                'of samples ({0}) than data (depthmaps: {1} samples)'.format(_num_samples(offset_points_projected), n_samples))
        
        if _num_samples(direction_vectors) != n_samples:
            raise ValueError('direction_vectors has a different number '
                                'of samples ({0}) than data (depthmaps: {1} samples)'.format(_num_samples(direction_vectors), n_samples))
        
        cv = _check_cv(cv, n_samples)
            
        if self.verbose > 0:
            if isinstance(parameter_iterable, Sized):
                n_candidates = len(parameter_iterable)
                print("Fitting {0} folds for each of {1} candidates, totalling"
                      " {2} fits".format(len(cv), n_candidates, n_candidates * len(cv)))
                      
        base_estimator = clone(self.estimator)

        pre_dispatch = self.pre_dispatch

        out = Parallel(
            n_jobs=self.n_jobs, verbose=self.verbose,
            pre_dispatch=pre_dispatch
        )(
            delayed(_fit_and_score)(clone(base_estimator), depthmaps, offset_points_projected,
                                    direction_vectors, true_joints, self.scorer_,
                                    train, test, self.verbose, parameters,
                                    self.fit_params, return_parameters=True,
                                    error_score=self.error_score)
                for parameters in parameter_iterable
                for train, test in cv)
        
        # Out is a list of triplet: score, estimator, n_test_samples
        n_fits = len(out)
        n_folds = len(cv)

        scores = list()
        grid_scores = list()
        for grid_start in range(0, n_fits, n_folds):
            n_test_samples = 0
            score = 0
            all_scores = []
            for this_score, this_n_test_samples, _, parameters in \
                    out[grid_start:grid_start + n_folds]:
                all_scores.append(this_score)
                if self.iid:
                    this_score *= this_n_test_samples
                    n_test_samples += this_n_test_samples
                score += this_score
            if self.iid:
                score /= float(n_test_samples)
            else:
                score /= float(n_folds)
            scores.append((score, parameters))
            
            grid_scores.append(_CVScoreTuple(
                parameters,
                score,
                np.array(all_scores)))
        # Store the computed scores
        self.grid_scores_ = grid_scores

        # Find the best parameters by comparing on the mean validation score:
        # note that `sorted` is deterministic in the way it breaks ties
        best = sorted(grid_scores, key=lambda x: x.mean_validation_score,
                      reverse=True)[0]
        self.best_params_ = best.parameters
        self.best_score_ = best.mean_validation_score

        if self.refit:
            # fit the best estimator using the entire dataset
            # clone first to work around broken estimators
            best_estimator = clone(base_estimator).set_params(
                **best.parameters)
            best_estimator.fit(depthmaps, offset_points_projected, direction_vectors, **self.fit_params)
            self.best_estimator_ = best_estimator
        return self

        
    def fit(self, depthmaps, offset_points_projected, direction_vectors, true_joints):

        return self._fit(depthmaps, offset_points_projected, direction_vectors, true_joints, ParameterGrid(self.param_grid))
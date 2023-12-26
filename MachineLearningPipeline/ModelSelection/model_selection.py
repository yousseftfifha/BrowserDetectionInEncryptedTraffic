import logging
from catboost import CatBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, confusion_matrix, classification_report)
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.neural_network import MLPClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.utils.class_weight import compute_class_weight


class ModelSelection:
    """
    This class will tune the model on given parameters using random search combined with the cross validation approach.
    """

    def __init__(self, hyperparam_configs, scoring, refit, nb_rand_search_iter, n_fold,
                 method='BayesSearchCV', random_state=42):

        self.method = method
        self.models_grid = hyperparam_configs
        self.scoring = scoring
        self.refit = refit
        self.nb_rand_search_iter = nb_rand_search_iter
        self.n_fold = n_fold
        self.random_state = random_state
        self.nb_jobs = 14
        self.dict_models = {
            'DecisionTreeClassifier': DecisionTreeClassifier(),
            'RandomForestClassifier': RandomForestClassifier(),
            'LGBMClassifier': LGBMClassifier(),
            'ExtraTreesClassifier': ExtraTreesClassifier(),
            'CatBoostClassifier': CatBoostClassifier(),
            'XGBClassifier': XGBClassifier(),
            'MLPClassifier': MLPClassifier()
        }

    def grid_to_space(self, grid):
        space_grid = {}
        for param, values in grid.items():
            print(values[0], values[-1])
            if isinstance(values[0], str) or isinstance(values[0], bool):
                space_grid[param] = Categorical(values)
            elif isinstance(values[0], int):
                space_grid[param] = Integer(values[0], values[-1])
            elif isinstance(values[0], float):
                space_grid[param] = Real(values[0], values[-1])
        print(space_grid)
        return space_grid

    def print_dataframe(self, filtered_cv_results):
        """Pretty print for filtered dataframe"""
        for mean_test_score, std_test_score, mean_fit_time, std_fit_time, params in zip(
                filtered_cv_results["mean_test_score"],
                filtered_cv_results["std_test_score"],
                filtered_cv_results["mean_fit_time"],
                filtered_cv_results["std_fit_time"],
                filtered_cv_results["params"],
        ):
            logging.info('******* cv_results *******')

            logging.info(
                "mean_test_score: {} (±{}), mean_fit_time: {} (±{}) for {}".format(mean_test_score,
                                                                                   std_test_score,
                                                                                   mean_fit_time,
                                                                                   std_fit_time,
                                                                                   params)
            )
            logging.info('*********************')

    def find_optimal_hyperparams(self, estimator_, grid, X_train, y_tain, X_val, y_val):

        nb_grid_pt = np.prod([len(params) for key, params in grid.items()])
        logging.info("estimator: {} number of grid points: {}, iter_number:{}".format(str(estimator_), nb_grid_pt,
                                                                                      self.nb_rand_search_iter))
        print("grid: ", grid)
        if self.method == "BayesSearchCV":
            search_space = self.grid_to_space(grid)
            class_labels = np.unique(y_tain)
            class_weights = compute_class_weight('balanced', classes=class_labels, y=y_tain)
            class_weights_dict = dict(zip(class_labels, class_weights))
            print(f"**************** Class Weights: {class_weights_dict}")
            if estimator_ == XGBClassifier:
                fit_params = {'scale_pos_weight': class_weights_dict}
            else:
                fit_params = {'class_weight': class_weights_dict}
            search_cv = BayesSearchCV(estimator_, search_space, n_iter=20, n_points=5, verbose=3,
                                      cv=self.n_fold, n_jobs=self.nb_jobs, scoring=self.scoring, fit_params=fit_params)

        else:
            if nb_grid_pt < self.nb_rand_search_iter:
                logging.info("-----Perform grid search ")

                search_cv = GridSearchCV(estimator_, grid, cv=self.n_fold, verbose=3, n_jobs=self.nb_jobs,
                                         scoring=self.scoring, return_train_score=True, refit=self.refit, )
            else:
                logging.info("------Perform randomized search ")

                search_cv = RandomizedSearchCV(estimator=estimator_, param_distributions=grid, scoring=self.scoring,
                                               n_iter=self.nb_rand_search_iter, cv=self.n_fold, n_jobs=self.nb_jobs,
                                               verbose=3, random_state=self.random_state, refit=self.refit)

        search_cv.fit(X_train, y_tain)
        val_score = search_cv.score(X_val, y_val)
        logging.info("validation score: {}".format(val_score))
        cv_results_ = pd.DataFrame(search_cv.cv_results_)
        logging.info("Best Accuracy: {}".format(search_cv.best_score_))
        self.print_dataframe(cv_results_)
        logging.info("Best Params for " + str(estimator_))
        logging.info(search_cv.best_params_)
        logging.info("Best Accuracy: {}".format(search_cv.best_score_))
        logging.info('--------------')
        return search_cv.best_estimator_, search_cv.best_score_, search_cv.best_params_, val_score

    def select_best_model(self, X_train, y_train, X_val, y_val, verbos):
        scores = []
        val_scores = []
        models = []
        model_best_params = []
        self.model_results = {}
        for model_name, params in self.models_grid.items():
            logging.info("*******------------model:{}".format(model_name))
            print("-------------grid:", params)

            model, best_score, best_params, val_score = self.find_optimal_hyperparams(self.dict_models[model_name],
                                                                                      params, X_train, y_train, X_val,
                                                                                      y_val)
            scores.append(best_score)
            models.append(model)
            model_best_params.append(best_params)
            val_scores.append(val_score)
            logging.info(
                '*****model_name: {} best_score {} best_params {}, validation score'.format(model_name, best_score,
                                                                                            best_params, val_score))

            self.print_model_performance(model, X_train, y_train, X_val, y_val, verbos)
        self.model_results = {'cv_score': scores, 'best_model': models, 'best_params': model_best_params,
                              'val_score': val_scores}
        # If the current model has a higher score on the validation set, update the best model and parameters
        self.model_results = pd.DataFrame.from_dict(self.model_results)
        logging.info('******* Best Model *******')
        logging.info('Best CV Score : {}'.format(max(scores)))
        logging.info('Best Valid Score : {}'.format(max(val_scores)))
        logging.info("Best model :{} ".format(str(models[val_scores.index(max(val_scores))])))
        return models[val_scores.index(max(val_scores))]

    def print_model_performance(self, estimator, X_train, y_train, X_test, y_test, verbos):
        estimator.fit(X_train, y_train)
        y_tr_pred = estimator.predict(X_train)
        y_pred = estimator.predict(X_test)
        if verbos:
            logging.info('--------------------------------------------------------------------')
            logging.info('Model used {}'.format(estimator))
            logging.info("Train results: {}".format(f1_score(y_train, y_tr_pred, average='micro')))
            logging.info(confusion_matrix(y_train, y_tr_pred))
            logging.info(classification_report(y_train, y_tr_pred))

            logging.info("Test  results {}".format(f1_score(y_test, y_pred, average='micro')))
            logging.info(confusion_matrix(y_test, y_pred))
            logging.info(classification_report(y_test, y_pred))
        logging.info("Accuracy:{}".format(accuracy_score(y_test, y_pred)))
        logging.info("F1 score micro:{}".format(f1_score(y_test, y_pred, average='micro')))
        logging.info("F1 score macro:{}".format(f1_score(y_test, y_pred, average='macro')))
        logging.info('--------------------------------------------------------------------')
        return accuracy_score(y_test, y_pred)

    def evaluate(self, model, X_train, X_test, y_train, y_test):

        model = model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        print(classification_report(y_test, y_pred))
        print("F1 (micro): {}".format(f1_score(y_test, y_pred, average='micro')))

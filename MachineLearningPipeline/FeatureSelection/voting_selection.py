import numpy as np
import pandas as pd
import time
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.feature_selection import (chi2, mutual_info_classif, SelectFromModel,RFE )
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score)
from sklearn.ensemble import ExtraTreesClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


class VotingSelector:
    """
    This class comprises methods on top of the init constructor that perform feature selection.
    """

    def __init__(self, model_configs, categorical_features, numeric_features):
        # Initialize the methods of feature selection
        self.selectors = {
            "lda_chi2_filter": self.lda_chi2_feature_selection,
            "mi_filter": self.mi_filter_selection,
            "embedded": self.embedded_feature_selection,
            # "rfe": self.recursive_feature_elimination,
        }
        # Initialize a dictionary of models to use
        self.model_config = model_configs
        self.categorical_features = categorical_features
        self.numeric_features = numeric_features
        self.initial_features = categorical_features + numeric_features
        self.models = {}
        for model_name, params in self.model_config.items():
            self.models[model_name] = globals()[model_name](**params)

    def lda_chi2_feature_selection(self, X, y, chi_square_thres=0.05):
        """
        This function performs feature selection on both numeric and categorical features using a combination of
        Linear Discriminant Analysis (LDA) for numeric features and the Chi-squared test for categorical features.
        :param X: The predictive variables (features).
        :param y: The target variable.
        :param chi_square_thres: The threshold value for feature selection using the Chi-squared test. Default value is 0.05
        :return: List of selected_features
        """
        selected_numeric_features = []
        selected_categ_features = []

        # Select numeric features using LDA
        if self.numeric_features:
            print(self.numeric_features)
            lda = LDA()
            lda.fit(X[self.numeric_features], y)
            median_sc = np.median(abs(lda.scalings_))
            selected_numeric_features = [feature for i, feature in enumerate(self.numeric_features) if
                                         (abs(lda.scalings_[i]) > median_sc).any()]

        # Select categorical features using Chi-squared test
        if self.categorical_features:
            chi_scores, p_values = chi2(X[self.categorical_features], y)
            selected_categ_features = [f for f, p in zip(self.categorical_features, p_values) if p < chi_square_thres]

        return selected_numeric_features + selected_categ_features

    @staticmethod
    def mi_filter_selection(X, y):
        """
        This function select the most important feature using mutual information.
        :param X: predictive dataset
        :param y: Target
        :return: list of selected_features
        """
        # Initialize mutual information for classification
        mi = mutual_info_classif(X, y)
        # select the most important features
        selected_features = [feature for feature, value in zip(X.columns, mi) if value > np.median(mi)]
        return selected_features

    @staticmethod
    def embedded_feature_selection( X, y, model):
        """Perform embedded method feature selection.

        Parameters:
        - X: array-like, shape (n_samples, n_features)
            Feature matrix.
        - y: array-like, shape (n_samples,)
            Target vector.
        - model: object
            The model to use for feature selection. The model must have a `feature_importances_` attribute.

        Returns:
        - selected_features: list of the selected features.
        """

        selected_features = []
        selector = SelectFromModel(model, threshold='mean', prefit=False)
        selector.fit(X, y)
        if hasattr(selector, 'feature_importances_'):
            # Create a selector object that will use the model to identify important features
            features_filter = selector.get_feature_names_out(X.columns.to_list())
            selected_features = [feature for feature, decision in zip(X.columns, features_filter) if decision]

        elif hasattr(selector, 'get_support'):
            # if not, use 'get_support' method to select features
            features_filter = selector.get_support()
            selected_features = [feature for feature, decision in zip(X.columns, features_filter) if decision]
        else:
            raise AttributeError('Model does not have feature importance or support attributes.')

        return selected_features

    @staticmethod
    def recursive_feature_elimination(X, y, model):
        """
        Perform recursive feature elimination on a given classification model.
        Parameters:
        model (object): Classification model to use for recursive feature elimination.
        X (array-like): Features to fit and transform.
        y (array-like): Target variable to fit and transform.
        n_features_to_select (int, optional): Number of features to select. If not provided, all features will be used.
        Returns:
        selected_features:  the selected features
        """

        try:
            # initialize RFE object with model and number of features to select
            rfe = RFE(model)

            # fit and transform the features
            rfe.fit_transform(X, y)

            # get the selected features
            selected_features = [X.columns[i] for i in range(X.shape[1]) if rfe.support_[i]]
            return selected_features

        except Exception as e:
            print(f"An error occurred: {e}")

    def majority_voting(self, X, y):
        """
        This function iterate over all selection methods, and for each feature, record whether
        it should be kept (1) or discarded (0) according to a certain method/model.

        :param X: predictive variables
        :param y: Target
        :return: votes: The result of the majority voting as a pandas DataFrame, where each row represents a feature and
        each column represents the votes from different methods/models for that feature
        """
        print("---- Calculate Votes ----")

        # Initialise scores for each feature to zero
        votes = []
        scores = {feature: 0 for feature in X.columns.to_list()}
        methods = []
        # Iterate each input selection technique
        for selector_name, selector_method in self.selectors.items():
            # Get the function from the techniques map, and run it getting the selected features
            if 'filter' not in selector_name:
                for model_name, model in self.models.items():
                    print("---- Method : ", selector_name, " Model: ", model_name, " ----")

                    selected_features = selector_method(X, y, model)
                    for feature in selected_features:
                        scores[feature] += 1
                    votes.append(
                        pd.DataFrame([int(feature in selected_features) for feature in X.columns]).T
                    )
                    methods.append(selector_name + "_" + model_name)

            else:
                print("---- Filter Method: ", selector_name, " ----")
                # select features based on the  selector method
                selected_features = selector_method(X, y)
                # append the result of voting of each feature
                votes.append(
                    pd.DataFrame([int(feature in selected_features) for feature in X.columns]).T
                )
                # For each of those features, increment their score by one
                for feature in selected_features: scores[feature] += 1
                methods.append(selector_name)

        votes = pd.concat(votes)
        votes.columns = X.columns

        votes.to_csv('log/votes.csv', sep=';')
        return votes

    @staticmethod
    def calculate_performance(model, x_train, y_train, x_test, y_test, average='micro'):
        """
        This function calculates the performance for a given model
        :param model: training model
        :param x_train: training subset
        :param y_train: training target
        :param x_test: test subset
        :param y_test: test target
        :return: performance
        """

        start = time.time()
        model.fit(x_train, y_train)
        stop = time.time()
        training = stop - start

        y_pred_tr = model.predict(x_train)
        start = time.time()
        y_pred = model.predict(x_test)
        stop = time.time()
        testing = stop - start
        f1_tr = f1_score(y_train, y_pred_tr, average=average)
        f1 = f1_score(y_test, y_pred, average=average)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average=average)
        rec = recall_score(y_test, y_pred, average=average)

        return f1_tr, f1, acc, prec, rec, training, testing

    def feature_evaluation(self, x_train, x_test, y_train, y_test, selected_features):
        """"
        This function calculates the performance for all models using initial features and the feature selected and
        returns a comparative dictionary.

        :param selected_features: list of selected features
        :param x_train: training subset
        :param y_train: training target
        :param x_test: test subset
        :param y_test: test target
        :return: A dictionary containing the performance metrics (F1 score, accuracy score, precision score,
        recall score, training time, and testing time) for each model on the selected features.
        """

        # Initialize lists to store the performance metrics for each model
        models = list(self.models.keys())

        # Calculate performance metrics for each model using dictionary comprehension
        performance_dict = {
            model_name: dict(
                zip(['f1_score_tr', 'f1_score', 'accuracy_score', 'precision_score', 'recall_score', 'training_time',
                     'testing_time'],
                    self.calculate_performance(model, x_train[selected_features], y_train, x_test[selected_features],
                                               y_test)))
            for model_name, model in self.models.items()
        }

        # Add model names to the dictionary
        for model_name in models:
            performance_dict[model_name]['model'] = model_name

        df = pd.DataFrame.from_dict(performance_dict, orient='index')

        df['nb_features'] = len(selected_features)

        df['score'] = 0.9 * np.mean(df['f1_score']) - 0.01 * (
                np.log((np.mean(df['training_time']) + 1)) + np.log((np.mean(df['testing_time']) + 1)))
        return df

    def get_best_threshold(self, log_dir, x_train, y_train, x_val, y_val):
        """
           Compute the best thresholds for the f1_scores_1 and training_1 metrics.

           Args:
           x_train: numpy array, training data features
           y_train: numpy array, training data labels
           x_val: numpy array, validation data features
           y_val: numpy array, validation data labels

           Returns: best_threshold: pandas dataframe,containing the details (threshold, selected features, number of
           selected features, and computed score) of the best threshold for feature selection based on the performance
           metrics and training/testing times.
           """

        # Compute majority voting on training data
        votes = self.majority_voting(x_train, y_train)

        # Initialize variables
        nb_votes = votes.shape[0]
        features_evals = []
        features = np.array(votes.columns)
        feature_nb_votes = np.sum(votes, axis=0)

        # Compute scores for different thresholds
        scores = {'threshold': [], 'features': [], 'score': [], 'nb_features': []}
        for threshold in range(nb_votes + 1):
            print(f'---- Feature Selection for Threshold: {threshold} ----')
            features_to_keep = features[feature_nb_votes >= threshold].tolist()
            print(f'Nb Selected features: {len(features_to_keep)}, Features: {features_to_keep}')
            if len(features_to_keep) == 0:
                continue
            feature_evaluation_df = self.feature_evaluation(x_train, x_val, y_train, y_val, features_to_keep)
            feature_evaluation_df['threshold'] = threshold

            scores['threshold'].append(threshold)
            scores['features'].append(features_to_keep)
            scores['nb_features'].append(len(features_to_keep))
            scores['score'].append(feature_evaluation_df['score'][0])
            features_evals.append(feature_evaluation_df)

        # Save scores and feature evaluations to csv files
        features_evals_df = pd.concat(features_evals)
        features_evals_df.to_csv(log_dir + '/threshold_eval.csv', sep=';')
        scores_df = pd.DataFrame(scores)
        scores_df.to_csv(log_dir + '/scores.csv', sep=';')

        # Find the best threshold and its details
        best_score = np.max(scores_df['score'].values)
        best_threshold = scores_df[scores_df['score'] == best_score]
        print(f'Best threshold details: {best_threshold}')

        return best_threshold

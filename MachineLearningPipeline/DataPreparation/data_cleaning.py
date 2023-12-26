from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import OneHotEncoder


class DataCleaning(BaseEstimator, TransformerMixin):
    def __init__(self, rare_thresh=0.01, quasi_constant_thresh=0.8, missing_thresh=0.1, encode_categ_cols=True):
        self.rare_thresh = rare_thresh
        self.missing_thresh = missing_thresh
        self.quasi_constant_thresh = quasi_constant_thresh
        self.categorical_counts = None
        self.numeric_cols = None
        self.bool_cols = None
        self.cat_cols = None
        self.constant_cols = []
        self.quasi_constant_cols = []
        self.cat_mapping = {}
        self.num_imputer = SimpleImputer(strategy='median')
        self.cat_imputer = SimpleImputer(strategy='most_frequent')
        self.encoder = None
        self.categ_col_encoding = encode_categ_cols
        self.cat_encode_cols = []
        self.all_col = None
        self.cols_to_drop = []

    def fit_transform(self, X, Y):
        # ------------fit-------------
        # Determine the numeric and categorical columns
        self.numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
        self.bool_cols = X.select_dtypes(include=bool).columns.tolist()
        self.cat_cols = X.select_dtypes(exclude=[np.number, bool]).columns.tolist()

        # Determine the mapping for rare categories
        for col in self.cat_cols:
            counts = X[col].value_counts(normalize=True)
            self.cat_mapping[col] = counts[counts > self.rare_thresh].index.tolist()

        # Determine the constant columns
        constant_cols = X.columns[X.nunique() == 1].tolist()

        # Determine the quasi constant columns for categorical features
        quasi_constant_cols = []
        for col in self.cat_cols + self.bool_cols:
            counts = X[col].value_counts(normalize=True)
            dominant_categories = counts[counts > self.quasi_constant_thresh].index.tolist()
            if len(dominant_categories) > 0:
                quasi_constant_cols.append(col)

        # Determine the duplicated columns
        duplicated_cols = X.columns[X.T.duplicated()].tolist()

        # Determine columns with high missing values
        num_missing = X.isnull().sum()
        high_missing_cols = num_missing[num_missing > self.missing_thresh * len(X)].index.tolist()

        # concatenate the lists of columns to drop
        self.cols_to_drop = set(duplicated_cols + constant_cols + quasi_constant_cols + high_missing_cols)

        # Update the list of numeric and categoric features

        self.numeric_cols = list(set(self.numeric_cols) - self.cols_to_drop)
        self.cat_cols = list(set(self.cat_cols) - self.cols_to_drop)
        self.bool_cols = list(set(self.bool_cols) - self.cols_to_drop)

        # Fit SimpleImputer on numeric columns
        if len(self.numeric_cols) > 0:
            self.num_imputer.fit(X[self.numeric_cols])

        # Fit SimpleImputer on categorical columns
        if len(self.cat_cols) > 0:
            self.cat_imputer.fit(X[self.cat_cols])

        # ---------Transform-------------------

        # Drop the duplicated rows
        X = X.drop_duplicates()
        Y = Y.loc[X.index]
        X.reset_index(drop=True, inplace=True)
        Y.reset_index(drop=True, inplace=True)

        if len(self.numeric_cols) > 0:
            X[self.numeric_cols] = self.num_imputer.transform(X[self.numeric_cols])
        if len(self.cat_cols) > 0:
            X[self.cat_cols] = self.cat_imputer.transform(X[self.cat_cols])
            # Fill missing values in boolean columns
        X[self.bool_cols] = X[self.bool_cols].fillna(False)
        X = X.drop(columns=list(self.cols_to_drop))

        # Replace rare categories with a new category
        for col in self.cat_cols:
            rare_cat = set(X[col].unique()) - set(self.cat_mapping[col])
            X[col] = X[col].replace(list(rare_cat), 'rare')

        # Label encoding of categorical features
        if not self.categ_col_encoding:
            X[self.cat_cols] = X[self.cat_cols].astype('category')
            self.all_col = self.cat_cols + self.numeric_cols + self.bool_cols
        else:
            X[self.cat_cols] = X[self.cat_cols].astype('str')
            self.encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
            self.encoder.fit(X[self.cat_cols])
            self.cat_encode_cols = self.encoder.get_feature_names_out().tolist()
            # Use pd.concat to join all columns at once
            encoded_cols = pd.DataFrame(data=self.encoder.transform(X[self.cat_cols]), columns=self.cat_encode_cols)
            X = pd.concat([X, encoded_cols], axis=1)
            self.all_col = self.cat_encode_cols + self.numeric_cols + self.bool_cols
        print("self.numeric_cols", self.numeric_cols)
        print("self.bool_cols", self.bool_cols)
        print("self.cat_cols", self.cat_cols)

        print("----------------", X.isna().sum())

        return X[self.all_col], Y

    def transform(self, X, Y=None):
        # Drop the constant columns
        X.drop(columns=list(self.cols_to_drop), inplace=True)
        # Drop the duplicated rows
        if X.shape[0] > 1:
            X.drop_duplicates(inplace=True)
            if Y is not None:
                Y = Y.loc[X.index]
                Y.reset_index(drop=True, inplace=True)
            X.reset_index(drop=True, inplace=True)

        # Replace rare categories with a new category
        for col in self.cat_cols:
            rare_cat = set(X[col].unique()) - set(self.cat_mapping[col])
            X[col] = X[col].replace(list(rare_cat), 'rare')
            # Fill missing values in boolean columns
        X[self.bool_cols] = X[self.bool_cols].fillna(False)

        if len(self.numeric_cols) > 0:
            X[self.numeric_cols] = self.num_imputer.transform(X[self.numeric_cols])
        if len(self.cat_cols) > 0:
            X[self.cat_cols] = self.cat_imputer.transform(X[self.cat_cols])

        if not self.categ_col_encoding:
            X[self.cat_cols] = X[self.cat_cols].astype('category')
        else:
            X[self.cat_cols] = X[self.cat_cols].astype('str')

            X = X.copy()
            encoded_cols = pd.DataFrame(data=self.encoder.transform(X[self.cat_cols]), columns=self.cat_encode_cols)
            X = pd.concat([X, encoded_cols], axis=1)

        return X[self.all_col], Y

# class DataCleaning(BaseEstimator, TransformerMixin):
#     def __init__(self, rare_thresh=0.01, quasi_constant_thresh=0.8, missing_thresh=0.1, encoding=True):
#         self.rare_thresh = rare_thresh
#         self.missing_thresh = missing_thresh
#         self.quasi_constant_thresh = quasi_constant_thresh
#
#         self.categorical_counts = None
#         self.numeric_cols = None
#         self.bool_cols = None
#         self.cat_cols = None
#         self.constant_cols = []
#         self.quasi_constant_cols = []
#
#         self.high_missing_cols = []
#         self.low_missing_cols = []
#         self.duplicated_cols = []
#         self.duplicated_rows = None
#         self.cat_mapping = {}
#         # Initialize SimpleImputer objects
#         self.num_imputer = SimpleImputer(strategy='median')
#         self.cat_imputer = SimpleImputer(strategy='most_frequent')
#         # self.scaler = StandardScaler()
#         self.encoder = None
#         self.categ_col_encoding = encoding
#         self.cat_encode_cols = None
#         self.all_col = None
#
#     def fit_transform(self, X, Y):
#
#         self.numeric_cols = X.select_dtypes(include=np.number).columns.tolist()
#         self.bool_cols = X.select_dtypes(include=bool).columns.tolist()
#         self.cat_cols = X.select_dtypes(exclude=[np.number, bool]).columns.tolist()
#
#         logging.info("Initial  numeric_cols {}:".format(self.numeric_cols))
#         logging.info("Initial cat_cols:{}:".format(self.cat_cols))
#         logging.info("Initial bool_cols:{}:".format(self.bool_cols))
#
#         # Determine the mapping for rare categories
#         for col in self.cat_cols:
#             counts = X[col].value_counts(normalize=True)
#             # print(col, "----> counts of categories : ", counts)
#             self.cat_mapping[col] = counts[counts > self.rare_thresh].index.tolist()
#
#         # Determine the constant columns
#         self.constant_cols = X.columns[X.nunique() == 1].tolist()
#         logging.info("Nb of constant columns:{}".format(len(self.constant_cols)))
#
#         # Determine the quasi constant columns for categorical features
#         for col in self.cat_cols + self.bool_cols:
#             counts = X[col].value_counts(normalize=True)
#             # print(col, " _", counts)
#             dominant_categories = counts[counts > self.quasi_constant_thresh].index.tolist()
#             if len(dominant_categories) > 0:
#                 self.quasi_constant_cols.append(col)
#
#         # self.quasi_constant_cols = X[self.numeric_cols].columns[X[self.numeric_cols].std() < self.constant_thresh]
#         logging.info("Nb of quasi-constant columns:{}".format(format(len(self.quasi_constant_cols))))
#
#         # Determine the duplicated columns
#         self.duplicated_cols = X.columns[X.T.duplicated()].tolist()
#         logging.info("Nb of duplicated columns:{}:".format(format(len(self.duplicated_cols))))
#
#         # Determine columns with high missing values
#         num_missing = X.isnull().sum()
#         self.high_missing_cols = num_missing[num_missing > self.missing_thresh * len(X)].index.tolist()
#         self.low_missing_cols = num_missing[num_missing <= self.missing_thresh * len(X)].index.tolist()
#         logging.info("high_missing_cols:{}:".format(len(self.high_missing_cols)))
#         logging.info("Nb of columns with low missing values:{}:".format(len(self.low_missing_cols)))
#
#         # concatenate the lists of columns to drop
#         self.cols_to_drop = set(
#             self.duplicated_cols + self.constant_cols + self.quasi_constant_cols + self.high_missing_cols)
#         # print("cols_to_drop:", self.cols_to_drop)
#         logging.info("Total nb of columns to drop:{}:".format(len(self.cols_to_drop)))
#
#         # Update the list of numeric and categoric features
#
#         self.numeric_cols = list(set(self.numeric_cols) - self.cols_to_drop)
#         self.cat_cols = list(set(self.cat_cols) - self.cols_to_drop)
#         self.bool_cols = list(set(self.bool_cols) - self.cols_to_drop)
#
#         logging.info("New numeric_cols:{}:".format(self.numeric_cols))
#         logging.info("New cat_cols:{}:".format(self.cat_cols))
#         logging.info("New bool_cols:{}:".format(self.bool_cols))
#
#         # Fit SimpleImputer on numeric columns
#         if len(self.numeric_cols) > 0:
#             self.num_imputer.fit(X[self.numeric_cols])
#
#         # Fit SimpleImputer on categorical columns
#         if len(self.cat_cols) > 0:
#             self.cat_imputer.fit(X[self.cat_cols])
#
#         # Determine the duplicated rows
#         self.duplicated_rows = X[X.duplicated(keep=False)]
#         # Drop the duplicated rows
#         X = X.drop_duplicates()
#         Y = Y.loc[X.index]
#
#         if len(self.numeric_cols) > 0:
#             X[self.numeric_cols] = self.num_imputer.transform(X[self.numeric_cols])
#         if len(self.cat_cols) > 0:
#             X[self.cat_cols] = self.cat_imputer.transform(X[self.cat_cols])
#         X[self.bool_cols].fillna(False)
#
#         X = X.drop(columns=list(self.cols_to_drop))
#
#         # Replace rare categories with a new category
#         for col in self.cat_cols:
#             rare_cat = set(X[col].unique()) - set(self.cat_mapping[col])
#             X[col] = X[col].replace(list(rare_cat), 'rare')
#
#         # Label encoding of categorical features
#         if not self.categ_col_encoding:
#             X[self.cat_cols] = X[self.cat_cols].astype('category')
#             self.all_col = self.cat_cols + self.numeric_cols + self.bool_cols
#         else:
#             X[self.cat_cols] = X[self.cat_cols].astype('str')
#             self.cat_encode_cols = []
#             # for c in self.cat_cols:
#             logging.info("column to encode:{} ".format(self.cat_cols, X[self.cat_cols].dtypes))
#             # logging.info("Unique vals:\n {}".format(X[c].value_counts()))
#             self.encoders = OneHotEncoder(sparse=False, handle_unknown='ignore')
#             self.encoders.fit(X[self.cat_cols])
#             # if 'rare' not in encoder.classes_.tolist() :
#             #     encoder.classes_=np.array(encoder.classes_.tolist() + ['rare'])
#             self.cat_encode_cols = self.encoders.get_feature_names_out().tolist()
#             X[self.cat_encode_cols] = pd.DataFrame(data=self.encoders.transform(X[self.cat_cols]),
#                                                    columns=self.cat_encode_cols)
#
#             logging.info("New one hot encoder columns : nb: {}, list: {}".format(len(self.cat_encode_cols),
#                                                                                  self.cat_encode_cols))
#             self.all_col = self.cat_encode_cols + self.numeric_cols + self.bool_cols
#
#         return X[self.all_col], Y
#
#     def transform(self, X, Y):
#         # Drop the constant columns
#         X = X.drop(columns=list(self.cols_to_drop))
#
#         # Drop the duplicated rows
#         X = X.drop_duplicates()
#         Y = Y.loc[X.index]
#
#         # Replace rare categories with a new category
#         for col in self.cat_cols:
#             rare_cat = set(X[col].unique()) - set(self.cat_mapping[col])
#             X[col] = X[col].replace(list(rare_cat), 'rare')
#         X[self.bool_cols].fillna(False)
#
#         if len(self.numeric_cols) > 0:
#             X[self.numeric_cols] = self.num_imputer.transform(X[self.numeric_cols])
#         if len(self.cat_cols) > 0:
#             X[self.cat_cols] = self.cat_imputer.transform(X[self.cat_cols])
#
#         if not self.categ_col_encoding:
#             X[self.cat_cols] = X[self.cat_cols].astype('category')
#         else:
#             X[self.cat_cols] = X[self.cat_cols].astype('str')
#
#             X[self.cat_encode_cols] = pd.DataFrame(data=self.encoders.transform(X[self.cat_cols]),
#                                                    columns=self.cat_encode_cols)
#
#         return X[self.all_col], Y

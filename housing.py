import csv
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import learning_curve
from sklearn.model_selection import GridSearchCV
from matplotlib.colors import ListedColormap
from sklearn import metrics
from sklearn.externals.six import StringIO
import pydotplus
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from sklearn.ensemble import VotingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA

from sklearn.linear_model import *

import xgboost as xgb
from xgboost.sklearn import XGBClassifier, XGBRegressor
#
# import keras
# from keras.datasets import mnist
# from keras.models import Sequential
# from keras.layers import Dense, Dropout
# from keras.optimizers import RMSprop
#
import sys
import re


def data_view():
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")

    # print train_df.head(3)
    # print train_df.info
    # print train_df.describe()
    #
    # print train_df.isnull
    # print test_df.isnull

    category_features = ['MSZoning', 'Street', 'Alley', 'LotShape',
                         'LotConfig', 'Neighborhood', 'Condition1',
                         'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
                         'Exterior2nd', 'MasVnrType', 'Foundation',
                         'Heating',
                         'Electrical', 'GarageType', 'GarageFinish', 'PavedDrive', 'MiscFeature',
                         'SaleType', 'SaleCondition']

    can_compare_features = ['LandContour', 'Utilities', 'LandSlope', 'ExterQual', 'ExterCond', 'BsmtQual', 'BsmtCond',
                            'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2', 'HeatingQC', 'CentralAir', 'KitchenQual',
                            'Functional', 'FireplaceQu', 'GarageQual', 'GarageCond',
                            'PoolQC', 'Fence']

    na_of_missing_category_features = ['MSZoning', 'Utilities', 'Exterior1st', 'Exterior2nd', 'MasVnrType',
                                       'MasVnrArea', 'KitchenQual', 'Functional', 'SaleType', 'Electrical',
                                       'BsmtFinType2', 'BsmtFinSF1']
    na_of_missing_value_features = ['LotFrontage', 'GarageYrBlt', 'GarageCars', 'GarageArea',
                                    'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath']
    na_of_notavailable_features = ['Alley', 'FireplaceQu', 'PoolQC', 'GarageType', 'GarageFinish', 'GarageQual',
                                   'GarageCond', 'Fence', 'MiscFeature', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
                                   'BsmtFinType1']

    # for these features, use NON to replace np.nan so we can convert it by labelEncoder
    train_df[na_of_notavailable_features] = train_df[na_of_notavailable_features].fillna("NON")
    test_df[na_of_notavailable_features] = test_df[na_of_notavailable_features].fillna("NON")

    # for missing category features (just missing several values), fill them with previous one
    train_df.loc[:, na_of_missing_category_features] = \
        train_df.loc[:, na_of_missing_category_features].fillna(method='pad')
    test_df.loc[:, na_of_missing_category_features] = \
        test_df.loc[:, na_of_missing_category_features].fillna(method='pad')

    # for value features, fill them with median
    imr_for_missing_value = preprocessing.Imputer(strategy="median")
    train_df.loc[:, na_of_missing_value_features] = \
        imr_for_missing_value.fit_transform(train_df.loc[:, na_of_missing_value_features])
    test_df.loc[:, na_of_missing_value_features] = \
        imr_for_missing_value.fit_transform(test_df.loc[:, na_of_missing_value_features])

    print train_df.isnull().sum()
    print "\n"
    print test_df.isnull().sum()

    # output abnormal cases
    # print train_df[train_df.loc[:, 'Electrical'].isnull()]
    # print train_df.loc[train_df.loc[:, 'Electrical'].isnull(), 'Electrical']
    # print test_df[test_df.loc[:, 'Electrical'].isnull()]
    # print test_df.loc[test_df.loc[:, 'Electrical'].isnull(), 'Electrical']

    for feature in set(category_features + can_compare_features):
        print feature
        lb = preprocessing.LabelEncoder()
        lb.fit(train_df.loc[:, feature])
        train_df.loc[:, feature] = lb.transform(train_df.loc[:, feature])
        test_df.loc[:, feature] = lb.transform(test_df.loc[:, feature])

    all_features = train_df.columns
    good_feature = list(set(all_features) - {'Id', 'SalePrice'})
    train_df_selected = train_df.loc[:, good_feature]
    test_df_selected = test_df.loc[:, good_feature]

    train_x = train_df_selected.values
    test_x = test_df_selected.values
    train_y = train_df.SalePrice.values.ravel()

    # sns.pairplot(train_df.loc[:, ['SalePrice', 'LotFrontage', 'GarageYrBlt', 'GarageCars', 'GarageArea']])
    # plt.show()

    # dummy seems not help
    # category_features_names = map_category_to_index(train_df_selected, category_features)
    # print train_df_selected.head(3)
    # print test_df_selected.head(3)
    # print category_features_names
    # ohe = preprocessing.OneHotEncoder(categorical_features=category_features_names)
    # ohe.fit(train_x)
    # train_x = ohe.transform(train_x).toarray()
    # test_x = ohe.transform(test_x).toarray()

    if False:
        # use pca to transfer data
        pca = PCA(n_components=5)
        pca.fit(train_x)
        print pca.explained_variance_
        print pca.explained_variance_ratio_
        train_x = pca.transform(train_x)
        test_x = pca.transform(test_x)

    scaler = preprocessing.StandardScaler()
    scaler.fit(train_x)
    train_x = scaler.transform(train_x)
    test_x = scaler.transform(test_x)

    # print train_x[0:3]
    # print train_y
    return train_x, train_y, test_x, train_df, test_df


def map_category_to_index(data_df, category_names):
    index_list = []
    for name in category_names:
        index_list.append(list(data_df.columns.values).index(name))
    return index_list


def generate_result(clf, test_x, test_df, filename):
    predict_y = clf.predict(test_x)
    test_result_df = pd.DataFrame({'Id': test_df.Id, 'SalePrice': predict_y})
    test_result_df.to_csv("%s.csv" % filename, index=False)


def gridsearch_randomforestregressor(train_x, train_y, test_x, test_df):
    print "==========%s==========" % sys._getframe().f_code.co_name
    tuned_parameters = {
        'n_estimators': range(100, 1000, 100),
        'max_depth': range(2, 10, 2),
        'min_samples_split': range(2, 10, 2)
    }

    clf = GridSearchCV(RandomForestRegressor(), tuned_parameters, cv=10, scoring='r2', n_jobs=-1, verbose=1)
    clf.fit(train_x, train_y)
    print clf.best_estimator_
    print clf.best_params_
    print clf.best_score_
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    generate_result(clf, test_x, test_df, "gridsearch_randomforestregressor")


def gridsearch_gbm(train_x, train_y, test_x, test_df):
    print "==========%s==========" % sys._getframe().f_code.co_name
    tuned_parameters = {
                        'n_estimators': range(100, 1000, 100), # 400
                        'learning_rate': np.linspace(0.01, 0.2, 20), # 0.11
                        # 'max_features': ['auto', 'sqrt', 'log2'],  # 'auto' is better
                        'max_depth': range(2, 10, 2)  # 4
                        }

    clf = GridSearchCV(GradientBoostingRegressor(max_features='auto'),
                       tuned_parameters, cv=10, scoring='r2', n_jobs=-1, verbose=1)
    clf.fit(train_x, train_y)
    print clf.best_estimator_
    print clf.best_params_
    print clf.best_score_
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    generate_result(clf, test_x, test_df, sys._getframe().f_code.co_name)


def gridsearch_svr(train_x, train_y, test_x, test_df):
    print "==========%s==========" % sys._getframe().f_code.co_name
    c_gamma_range = [0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000]
    tuned_parameters = {
        'C': c_gamma_range,
        'kernel': ['linear', 'rbf'],
        'gamma': c_gamma_range
    }

    clf = GridSearchCV(SVR(), tuned_parameters, cv=10, scoring='r2', n_jobs=-1, verbose=1)
    clf.fit(train_x, train_y)
    print clf.best_estimator_
    print clf.best_params_
    print clf.best_score_
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    generate_result(clf, test_x, test_df, sys._getframe().f_code.co_name)


def dry_run_for_all_models(classifiers, train_x, train_y, test_x, test_df):
    print "==========%s==========" % sys._getframe().f_code.co_name
    train_build_x, train_val_x, train_build_y, train_val_y = train_test_split(train_x, train_y,
                                                                              test_size=0.2, random_state=0)
    for clf in classifiers:
        clf.fit(train_build_x, train_build_y)
        train_build_y_pred = clf.predict(train_build_x)
        train_val_y_pred = clf.predict(train_val_x)

        print "[%s] R^2 train_build: %f, train_val: %f" % (clf.__class__.__name__,
                                                           metrics.r2_score(train_build_y, train_build_y_pred),
                                                           metrics.r2_score(train_val_y, train_val_y_pred))
        generate_result(clf, test_x, test_df, clf.__class__.__name__)


def main():
    global_classifiers = {LinearRegression(),
                          Ridge(alpha=1.0),
                          Lasso(alpha=1.0),
                          ElasticNet(alpha=1.0, l1_ratio=0.5),
                          RANSACRegressor(LinearRegression(),
                                          max_trials=1000,
                                          min_samples=50,
                                          residual_metric=lambda x: np.sum(np.abs(x), axis=1),
                                          residual_threshold=5,
                                          random_state=0),
                          RandomForestRegressor(n_estimators=1000,
                                                min_samples_split=2,
                                                min_samples_leaf=1,
                                                max_depth=8,
                                                criterion='mse',
                                                random_state=1),
                          SVR(kernel='linear', gamma=0.0001, C=1000),
                          GradientBoostingRegressor(n_estimators=400, max_features='auto',
                                                    learning_rate=0.11, max_depth=4),
                          # XGBRegressor(learning_rate=0.01,
                          #              n_estimators=3500,
                          #              max_depth=3,
                          #              min_child_weight=1,
                          #              gamma=0.2,
                          #              subsample=0.7,
                          #              colsample_bytree=0.6,
                          #              objective='binary:logistic',
                          #              nthread=4,
                          #              reg_alpha=0,
                          #              reg_lambda=1,
                          #              scale_pos_weight=1,
                          #              seed=27)
                          }

    train_x, train_y, test_x, train_df, test_df = data_view()
    exec_flag = ['not used', 1, 0, 0, 0, 0, 0, 0, 0]
    # part I Traditional model comparison
    if exec_flag[1]:
        dry_run_for_all_models(global_classifiers, train_x, train_y, test_x, test_df)

    # part II GridSearch for RandomForestRegressor
    if exec_flag[2]:
        gridsearch_randomforestregressor(train_x, train_y, test_x, test_df)

    # part III GridSearch for SVR
    if exec_flag[3]:
        gridsearch_svr(train_x, train_y, test_x, test_df)

    # part IV GridSearch for GBM
    if exec_flag[4]:
        gridsearch_gbm(train_x, train_y, test_x, test_df)

        # # part V Voting
        # if exec_flag[5]:
        #     voting_try(classifiers, train_x, train_y, test_x, test_df)


if __name__ == "__main__":
    pd.set_option('display.height', 1000)
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    main()

import pandas as pd
import numpy as np
import sklearn
from sklearn.cross_validation import KFold
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import linear_model
from sklearn import svm
import pickle

from models import *
from process_data import *
# import matplotlib.pyplot as plt
# %matplotlib inline
 
def azureml_main(train_data = None, test_data = None):
    features = [x for x in train_data.columns if x not in ['geo','segment','subgroup' ]]
    test_data = test_data[features]

    t = encode_categorical_feature('religion')
    t.fit(train_data)
    test_data = t.transform(test_data)
    X_test  = process_data_X(test_data)

    model_name = 'sum_of_models'
    model_name = 'rf'
    if 1:
        train_data = t.transform(train_data)
        X_train    = process_data_X(train_data)
        y_train    = process_data_y(train_data)
        if model_name == 'rf':
            m = RandomForestClassifier(n_estimators=1000, max_depth=13)
        elif model_name == 'gbm':
            m = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=1)
        elif model_name == 'sum_of_models':
            m = sum_model()
        elif model_name == 'vote_common':
            m = vote_model()
        elif model_name == 'max_proba_models':
            m = max_proba_model()
        elif model_name == 'combined_model':
            m = combined_model()
        m.fit(X_train, y_train)
    else:
        m = pickle.load(open("./Script Bundle/model_7.1_sum_of_models.pkl",'rb'))

    y_pred = m.predict(X_test)
    
    test_data['Geo_Pred']      = y_pred/100
    test_data['Segment_Pred']  = y_pred/10%10
    test_data['Subgroup_Pred'] = y_pred%10
    
    return test_data[['patientID','Geo_Pred','Segment_Pred','Subgroup_Pred']]

if __name__ == '__main__':
    data = pd.read_csv('../../datasets/WomenHealth_Training.csv')

    scores = []
    for train_ind, valid_ind in KFold(data.shape[0], n_folds=4): 
        train_data = data.loc[train_ind,:]
        test_data  = data.loc[valid_ind,:]
        y_test     = process_data_y(test_data)
        data_pred  = azureml_main(train_data, test_data)
        y_pred     = process_data_y(data_pred, False)
        scores.append(sklearn.metrics.accuracy_score(y_pred, y_test))
        print scores[-1]


        

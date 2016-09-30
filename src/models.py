import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import linear_model
from sklearn import svm

def gen_models():
    def gen_models_rf(parameters):
        models = []
        for n in parameters['n_estimators']:
            for m in parameters['max_depth']:
                models.append(RandomForestClassifier(n_estimators=n, max_depth=m, random_state=0))
        return models
    def gen_models_gbm(parameters):
        models = []
        for n in parameters['n_estimators']:
            for m in parameters['max_depth']:
                for l in parameters['learning_rate']:
                    models.append(GradientBoostingClassifier(n_estimators=n, max_depth=m, random_state=0))
        return models
    def gen_models_log(parameters):
        models = []
        for p in parameters['penalty']:
            for c in parameters['C']:
                models.append(linear_model.LogisticRegression(penalty=p, C=c))
        return models

    def gen_models_svm(parameters):
        models = []
        for c in parameters['C']:
            for k in parameters['kernel']:
                if 'gamma' in parameters:
                    for g in parameters['gamma']:
                        models.append(svm.SVC(C=c, kernel=k, gamma=g, probability=True))
                else:
                    models.append(svm.SVC(C=c, kernel=k, probability=True))

        return models

    param_gbm = { 'n_estimators':[50, 100,200],
                  'learning_rate':[0.1,1.0],
                  'max_depth':[1,2,3]}
    param_rf  = {'n_estimators':[100,200,500,800,1000,],
                  'max_depth':[1,2,4,6,8,10,13]}
    param_svc = [
                  {'C':[0.1, 1, 10, 100], 'kernel': ['linear']},
                  {'C':[0.1, 1, 10, 100], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']},
                ]
    param_log = {'penalty':('l1','l2'), 
                  'C':[0.1, 1, 10, 100]}
    models  = gen_models_gbm(param_gbm)
    models += gen_models_rf(param_rf)
    models += gen_models_svm(param_svc[0])
    models += gen_models_svm(param_svc[1])
    models += gen_models_log(param_log)
    selection_ind = [45, 38, 52, 31, 24, 16, 17, 30, 14, 15, 37]
    models = [models[i] for i in selection_ind]
    return models


class model_collection:
    def __init__(self, models):
        self.models = models[:]
    def fit(self, X_train, y_train):
        self.classes = np.unique(y_train)
        for i in range(len(self.models)):
            self.models[i].fit(X_train,y_train)
    def predict_proba(self, X_test):
        self.y_proba = []
        for m in self.models:
            self.y_proba.append(m.predict_proba(X_test))
        return self.y_proba
    def predict(self, X_test):
        self.y_pred = []
        for m in self.models:
            self.y_pred.append(m.predict(X_test))
        return self.y_pred
    

class sum_model:
    def __init__(self):
        self.models = gen_models()
    def fit(self, X_train, y_train):
        self.classes = np.unique(y_train)
        for i in range(len(self.models)):
            self.models[i].fit(X_train,y_train)
    def predict(self, X_test):
        y_pred      = np.array([m.predict(X_test) for m in self.models]).T
        y_proba     = [m.predict_proba(X_test) for m in self.models]
        y_proba_max = np.array([[row.max() for row in yy] for yy in y_proba]).T
        # The sum of probabilities
        y_proba_sum = np.zeros_like(y_proba[0])
        for yy in y_proba:
            y_proba_sum += yy
        y_pred_sum = [self.classes[row.argmax()] for row in y_proba_sum]
        
        prediction = np.array(y_pred_sum)
        return prediction
        

class combined_model:
    def __init__(self):
        self.models = gen_models()
        self.num_models = len(self.models)
    def fit(self, X_train, y_train):
        self.classes = np.unique(y_train)
        for i in range(len(self.models)):
            self.models[i].fit(X_train,y_train)
    def predict(self, X_test):
        y_pred      = np.array([m.predict(X_test) for m in self.models]).T
        y_proba     = [m.predict_proba(X_test) for m in self.models]
        y_proba_max = np.array([[row.max() for row in yy] for yy in y_proba]).T
        # predictions with better confidency
        p_cut = 0.8
        y_pred_increament = np.zeros(X_test.shape[0])
        for k in range(y_pred.shape[1]):
            for ind in range(X_test.shape[0]):
                if y_pred_increament[ind]==0 and y_proba_max[ind,k]>p_cut:
                    y_pred_increament[ind] = y_pred[ind,k]
        # remaining preditions
        ind_remaining = [i for i in range(X_test.shape[0]) if y_pred_increament[i]==0]
        best_m_for_remaining = 0
        y_pred_increament[ind_remaining] = y_pred[ind_remaining,best_m_for_remaining]
        return y_pred_increament

def vote_common(row):
    d={}
    for v in row:
        d[v] = d.get(v,0)+1
    vote = sorted(d,key=d.get)
    return vote[0]

class vote_model:
    def __init__(self):
        self.models = gen_models()
        self.num_models = len(self.models)
    def fit(self, X_train, y_train):
        self.classes = np.unique(y_train)
        for i in range(len(self.models)):
            self.models[i].fit(X_train,y_train)
    def predict(self, X_test):
        y_pred      = np.array([m.predict(X_test) for m in self.models]).T
        y_pred_vote = np.apply_along_axis(vote_common, 1, y_pred)
        return y_pred_vote


class max_proba_model:
    def __init__(self):
        self.models = gen_models()
        self.num_models = len(self.models)
    def fit(self, X_train, y_train):
        self.classes = np.unique(y_train)
        for i in range(len(self.models)):
            self.models[i].fit(X_train,y_train)
    def predict(self, X_test):
        y_pred      = np.array([m.predict(X_test) for m in self.models]).T
        y_proba     = [m.predict_proba(X_test) for m in self.models]
        y_proba_max = np.array([[row.max() for row in yy] for yy in y_proba]).T
        pred_choice = np.apply_along_axis(np.argmax, 1,y_proba_max)
        pred_by_choice = np.array([y_pred[i,pred_choice[i]] for i in range(len(y_test))])
        return pred_by_choice


class combined_model:
    def __init__(self):
        self.models = gen_models()
        self.num_models = len(self.models)
    def fit(self, X_train, y_train):
        self.classes = np.unique(y_train)
        for i in range(len(self.models)):
            self.models[i].fit(X_train,y_train)
    def predict(self, X_test):
        y_pred      = np.array([m.predict(X_test) for m in self.models]).T
        y_proba     = [m.predict_proba(X_test) for m in self.models]
        y_proba_max = np.array([[row.max() for row in yy] for yy in y_proba]).T
        # predictions with better confidency
        p_cut = 0.8
        y_pred_increament = np.zeros(X_test.shape[0])
        for k in range(y_pred.shape[1]):
            for ind in range(X_test.shape[0]):
                if y_pred_increament[ind]==0 and y_proba_max[ind,k]>p_cut:
                    y_pred_increament[ind] = y_pred[ind,k]
        # remaining preditions
        ind_remaining = [i for i in range(X_test.shape[0]) if y_pred_increament[i]==0]
        best_m_for_remaining = 0
        y_pred_increament[ind_remaining] = y_pred[ind_remaining,best_m_for_remaining]
        return y_pred_increament



import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

class encode_categorical_feature:
    def __init__(self,feature):
        self.f_set = []
        self.feature = feature
    def fit(self,data):
        if self.feature in data.columns:
            self.f_set = sorted(list(data[self.feature].unique()))
            f_values   = data[self.feature].apply(lambda x: [self.f_set.index(x)])
            self.encoder = OneHotEncoder().fit(list(f_values))
    def transform(self,data):
        if not self.f_set:
            return data
        f_values  = data[self.feature].apply(lambda x: [self.f_set.index(x)])
        f_encoded = self.encoder.transform(list(f_values)).toarray()
        f_df      = pd.DataFrame(f_encoded, columns=[self.feature+'_'+str(x) for x in self.f_set])
        new_data  = pd.concat([data.reset_index(drop=True),f_df], axis=1)
        new_data.drop(self.feature, axis=1, inplace=True)
        return new_data


def process_data_X(data):
    data.fillna(0, inplace=True)
    features = [x for x in data.columns if x not in 
                    ['patientID', 'geo','segment','subgroup','combined_label', 'INTNR' ]]
    X = data[features]
    # select features
    selection1 = [  'DISTRICT', u'tribe', u'REGION_PROVINCE', u'babydoc', u'foodinsecurity', 
                    u'religion_Buddhist', u'india', u'hindu', u'religion_Hindu', 
                    u'religion_Russian/Easter', u'educ', u'Debut', u'literacy', u'christian', 
                    u'hivknow', u'ModCon', u'age', u'thrasher', u'usecondom', 
                    u'religion_Other Christia', u'religion_Muslim', u'muslim', u'lowlit', 
                    u'multpart', u'motorcycle', u'CHILDREN', u'LaborDeliv']
    selection2 = [  u'christian', u'hindu', u'REGION_PROVINCE', u'DISTRICT', u'electricity', 
                    u'age', u'tribe', u'foodinsecurity', u'EVER_HAD_SEX', u'EVER_BEEN_PREGNANT', 
                    u'CHILDREN', u'india', u'married', u'multpart', u'educ', u'literacy', 
                    u'LaborDeliv', u'babydoc', u'Debut', u'ModCon', u'usecondom', u'hivknow', 
                    u'religion_Buddhist', u'religion_Hindu', u'religion_Russian/Easter']
    selection3 = [  u'DISTRICT', u'tribe', u'REGION_PROVINCE', u'babydoc', u'india', u'educ', 
                    u'Debut', u'literacy', u'hivknow', u'ModCon', u'age', u'usecondom', 
                    u'multpart', u'CHILDREN', u'LaborDeliv', u'married']
    selected_features = selection1+selection2+selection3
    X = X[selected_features]
    return X.values

    

def process_data_y(data, isTrain=True):
    if isTrain:
        data['combined_label'] = 100*data['geo'] + 10*data['segment'] + data['subgroup']
    else:
        data['combined_label'] = 100*data['Geo_Pred'] + 10*data['Segment_Pred'] + data['Subgroup_Pred']
    return data['combined_label'].values


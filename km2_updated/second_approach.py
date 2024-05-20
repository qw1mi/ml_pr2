#2nd approach
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
from first_approach import OutlierReplacer
data = pd.read_csv('data/data.csv')

def map_categorical_features(data):
    data_encoded= data.copy()
    month_to_num = {
        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'June': 6, 'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
    }
    visitor_to_num = {
        'Returning_Visitor': 2, 'New_Visitor': 1, 'Other': 0
    }

    data_encoded['Month'] = data_encoded['Month'].map(month_to_num)
    data_encoded['VisitorType'] = data_encoded['VisitorType'].map(visitor_to_num)

    label_encoder = LabelEncoder()
    data_encoded['Weekend'] = label_encoder.fit_transform(data_encoded['Weekend'])
    data_encoded['Revenue'] = label_encoder.fit_transform(data_encoded['Revenue'])

    return data_encoded

columns_to_keep = ['Administrative', 'Administrative_Duration', 'ProductRelated', 'ProductRelated_Duration', 'ExitRates',
                   'Month', 'OperatingSystems', 'Browser', 'Region', 'TrafficType', 'VisitorType', 'Weekend', 'Revenue']

preprocessor = ColumnTransformer(
    transformers=[
        ('scaler', StandardScaler(), columns_to_keep),
    ],
    remainder='drop'
)

pipeline = Pipeline([
        ('mapper', FunctionTransformer(map_categorical_features)),
        ('preprocessor', preprocessor),
        ('outlier_replacer', OutlierReplacer(factor=3.0))
    ])

pipeline.fit(data) 
processed_data = pipeline.transform(data)  
columns = columns_to_keep
processed_df = pd.DataFrame(processed_data, columns=columns)  
processed_df.head()
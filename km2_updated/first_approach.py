#1st approach
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer

class OutlierReplacer(BaseEstimator, TransformerMixin):
    def __init__(self, factor=3.0):
        self.factor = factor

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for i in range(X.shape[1]):
            col = X[:, i]
            q1, q3 = np.percentile(col, [25, 75])
            IQR = q3 - q1
            l_bound = q1 - self.factor * IQR
            u_bound = q3 + self.factor * IQR

            if u_bound == 0:
                continue

            col[col < l_bound] = l_bound
            col[col > u_bound] = u_bound

        return X

def preprocess_data(data):
    data_encoded = data.copy()
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

    preprocessor = ColumnTransformer(
        transformers=[
            ('scaler', StandardScaler(), data_encoded.columns),
        ],
        remainder='passthrough'
    )

    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('outlier_replacer', OutlierReplacer(factor=3.0)),
    ])

    data_encoded = pipeline.fit_transform(data_encoded)

    return data_encoded

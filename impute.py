import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
class DataFrameImputer(TransformerMixin):
    """Imputer to impute missing values  
    """
    def fit(self, X,constant=None):
        """Impute missing values.
        If columns is of dtype object, it will write the most common occuring object,
        else it will select the mean value of the column

        :param X: The Dataframe to impute missing values
        :type X: pandas.Dataframe
        :param constant: Constant value for non numeric columns, defaults to None
        :type constant: object, optional
        :return: DataFrameImputer Object with data fitted
        :rtype: DataFrameImputer
        """
        self.fill = pd.Series([X[c].value_counts().index[0] if constant is None and X[c].dtype == np.dtype('O') else constant if X[c].dtype == np.dtype('O')  else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X):
        """Fills the missing values according to the column type

        :param X: Initial Dataframe
        :type X: pandas.Dataframe
        :return: Dataframe with imputed missing values
        :rtype: pandas.Dataframe
        """
        return X.fillna(self.fill)


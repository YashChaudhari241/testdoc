import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin
class DataFrameImputer(TransformerMixin):
    """
        Imputer to impute missing values such that 
    """
    def fit(self, X,constant=None):
        """Impute missing values.
        If columns is of dtype object, it will write the most common occuring object,
        else it will select the mean value of the column

        Args:
            X (pandas.Dataframe): The Dataframe to impute missing values
            constant (object): Constant value for non numeric columns

        Returns:
            DataFrameImputer: DataFrameImputer Object with data fitted
        """
        self.fill = pd.Series([X[c].value_counts().index[0] if constant is None and X[c].dtype == np.dtype('O') else constant if X[c].dtype == np.dtype('O')  else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X):
        """Fills the missing values according to the column type

        Args:
            X (pandas.Dataframe): Initial Dataframe

        Returns:
            pandas.Dataframe: Dataframe with imputed missing values
        """
        return X.fillna(self.fill)





import pandas as pd
from sklearn.preprocessing import StandardScaler


# Define your functions here
def drop_columns(df, columns_to_drop):
    """
    Drops specified columns from the DataFrame.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    columns_to_drop (list): List of columns to drop.
    
    Returns:
    pd.DataFrame: DataFrame with specified columns removed.
    """
    return df.drop(columns=columns_to_drop)

def standardize_columns(df, columns_to_standardize):
    """
    Standardizes specified numerical columns using StandardScaler.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    columns_to_standardize (list): List of numerical columns to standardize.
    
    Returns:
    pd.DataFrame: DataFrame with standardized columns.
    """
    scaler = StandardScaler()
    df[columns_to_standardize] = scaler.fit_transform(df[columns_to_standardize])
    return df

def check_missing_values(df):
    """
    Checks for missing values in the DataFrame.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    
    Returns:
    pd.Series: Series containing the count of missing values for each column.
    """
    return df.isnull().sum()

def preprocess_data(df):
    """
    A pipeline function that applies a series of preprocessing steps.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    
    Returns:
    pd.DataFrame: Preprocessed DataFrame.
    """
    # Example pipeline
    df = drop_columns(df, ['patient_id'])
    numerical_cols = ['age', 'bmi', 'physical_activity', 'diet_quality', 
                      'sleep_quality', 'lung_function_fev1', 'lung_function_fvc']
    df = standardize_columns(df, numerical_cols)
    missing_values = check_missing_values(df)
    print("Missing values per column:\n", missing_values)
    return df

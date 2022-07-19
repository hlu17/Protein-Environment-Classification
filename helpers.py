
import sqlite3
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

DATABASE_NAME = 'metagenome_classification.db'

#-------- DATA MANIPULATION
def get_protein_proportions(df):
    # Each column has counts of "hits" now, but not consistent across observations. 
    # Get proportions each protein appears.
    df = df.div(df.sum(axis=1), axis=0)
    return df


def drop_empty_columns(df):
    """
    Some Pfams don't have hits in any sample - drop them.
    :param df: pandas dataframe
    :ptype: df
    :return: dataframe with empty columns removed
    :rtype: df
    """
    df.replace(0, np.nan, inplace=True)
    df.dropna(how='all', axis=1, inplace=True)
    df.replace(np.nan, 0, inplace=True)
    return df


#-------- DATA SETUP
def official_data_split():
    labels = pd.read_csv('data//metadata_by_jgi_spid.tsv', sep='\t', header=0)
    features = pd.read_csv('data//features.csv',header=None)
    columns = pd.read_csv('data//feature_column_names.tsv', sep='\t', header=None)
    
    # Add columns headers to the features data
    first_column = columns.iloc[:, 0].values.tolist()
    samples_ids = features.iloc[:, 0].values.tolist()
    column_names = ['sample_id', *first_column]
    
    features.columns = column_names
    
    # Filter the labels to align with features data
    labels = labels[labels['jgi_spid'].isin(samples_ids)]
    labels.shape

    # assign features and labels
    X = features.loc[:, 'PF00001.19':'PF17225.1']
    Y = labels[['EMPO_1', 'EMPO_2', 'EMPO_3']]
    
    # Split the training and test dataset by 70/30 ratio, stratifies by class labels
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=1, stratify=Y)
    
    return X_train.reset_index(drop=True), X_test.reset_index(drop=True), Y_train.reset_index(drop=True), Y_test.reset_index(drop=True)
    
#-------- QUERYING DATABASE
def get_training_observations():
    print(f"Getting all training observations from '{DATABASE_NAME}'...")
    conn = sqlite3.connect('..//data//' + DATABASE_NAME)
    
    x_train_transposed = pd.read_sql_query("SELECT * FROM x_train", con = conn)
    
    conn.commit()
    conn.close()
    
    x_train_transposed.set_index('index', inplace=True)
    x_train = x_train_transposed.T
    x_train_normalized = get_protein_proportions(x_train)
    
    return x_train_normalized

def get_training_labels():
    print(f"Getting all training labels from '{DATABASE_NAME}'...")
    conn = sqlite3.connect('..//data//' + DATABASE_NAME)
    
    y_train_transposed = pd.read_sql_query("SELECT * FROM y_train", con = conn)
    
    conn.commit()
    conn.close()
    
    y_train_transposed.set_index('index', inplace=True)
    y_train = y_train_transposed.T
    
    return y_train

def get_test_observations():
    print(f"Getting all test observations from '{DATABASE_NAME}'...")
    conn = sqlite3.connect('..//data//' + DATABASE_NAME)
    
    x_test_transposed = pd.read_sql_query("SELECT * FROM x_test", con = conn)
    
    conn.commit()
    conn.close()
    
    x_test_transposed.set_index('index', inplace=True)
    x_test = x_test_transposed.T
    x_test_normalized = get_protein_proportions(x_test)
    
    return x_test_normalized

def get_test_labels():
    print(f"Getting all test labels from '{DATABASE_NAME}'...")
    conn = sqlite3.connect('..//data//' + DATABASE_NAME)
    
    y_test_transposed = pd.read_sql_query("SELECT * FROM y_test", con = conn)
    
    conn.commit()
    conn.close()
    
    y_test_transposed.set_index('index', inplace=True)
    y_test = y_test_transposed.T
    
    return y_test


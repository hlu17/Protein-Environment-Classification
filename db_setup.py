"""Sets up sqlite db with official training and test sets.
Run if 'data/metagenome_classification.db' does not exist or is corrupted.
"""

import sqlite3
from helpers import official_data_split, DATABASE_NAME

# make database
conn = sqlite3.connect('data//' + DATABASE_NAME)
c = conn.cursor()

# run script to make training and test sets
X_train, X_test, Y_train, Y_test = official_data_split()

# add train/test tests to db
X_train.T.to_sql('x_train', conn, if_exists='replace', index=True)
Y_train.T.to_sql('y_train', conn, if_exists='replace', index=True)
X_test.T.to_sql('x_test', conn, if_exists='replace', index=True)
Y_test.T.to_sql('y_test', conn, if_exists='replace', index=True)

conn.commit()
conn.close()

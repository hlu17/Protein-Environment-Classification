"""
Creates a pandas dataframe from the metadata-all.db file 
https://www.sqlitetutorial.net/sqlite-python/update/
"""
import sqlite3
from sqlite3 import Error
import os
import pandas as pd
    
class Database:
    def __init__(self):
        self.database = os.path.abspath("metadata-all.db")
        self.conn = self.create_connection()                              
        self.df = self.query_all()
        
    def create_connection(self):
        """ create a database connection to a SQLite database """
        self.conn = None
        try:
            self.conn = sqlite3.connect(self.database)
            return self.conn
        except Error as e:
            print(e)
        return self.conn

    def query_all(self):
        """ query the entrie table and return as a pandas DataFrame """
        query = """ 
                SELECT *
                FROM tbl1
                
                """
        df = pd.read_sql_query(query, self.conn)
        return df


if __name__ == '__main__':
    db = Database()
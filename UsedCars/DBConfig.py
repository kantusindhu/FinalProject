# author: kantus
# Date: 10/06/2023
# Details: This Python file is created to connect to Database and used the regular syntax to connect the database.

import mysql.connector
class DBConnection:
    @staticmethod
    def getConnection():
        database = mysql.connector.connect(host="localhost", user="root", passwd="root", port='3306', db='used_cars')
        return database
if __name__=="__main__":
    print(DBConnection.getConnection())
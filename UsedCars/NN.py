# author: kantus
# Date: 10/27/2023
# Details:
# 1). Used the syntax for Nueral Network Algorithm using Decision Tree Classifier. https://www.datacamp.com/tutorial/decision-tree-classification-python how to calculate the accuracy and same 
#     and used different sources to pull how to calculate the Precision, Recall, F1 Score, Mean Absolute Error, Mean Squared Error 
#     and R-Squared Error.

from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn import metrics
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score,confusion_matrix, mean_squared_error, mean_absolute_error
from DBConfig import DBConnection
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
import pandas as pd
def nn_evaluation(X_train, X_test, y_train, y_test):
    db = DBConnection.getConnection()
    cursor = db.cursor()

    dtc_clf = DecisionTreeClassifier()

    dtc_clf.fit(X_train, y_train)

    predicted = dtc_clf.predict(X_test)

    accuracy = accuracy_score(y_test, predicted)*100

    precision = precision_score(y_test, predicted, average="macro")*100

    recall = recall_score(y_test, predicted, average="macro")*100

    fscore = f1_score(y_test, predicted, average="macro")*100

    values = ("ANN", str(accuracy), str(precision), str(recall), str(fscore))
    sql = "insert into evaluations values(%s,%s,%s,%s,%s)"
    cursor.execute(sql, values)
    db.commit()

    

    print("NN=",accuracy,precision,recall,fscore)
    return accuracy,precision,recall,fscore


def get_errors(Z_train, Z_test, m_train, m_test):
    dtc_clf_mfe = DecisionTreeRegressor()

    dtc_clf_mfe.fit(Z_train, m_train)

    y_pred = dtc_clf_mfe.predict(Z_test)
    mae = metrics.mean_absolute_error(m_test, y_pred)
    mse = metrics.mean_squared_error(m_test, y_pred)
    rse = metrics.r2_score(m_test, y_pred)
    print('Mean Absolute Error      : ', mae)
    print('Mean Squared  Error      : ', mse)
    print('R Squared Error          : ', rse)
    return mae,mse,rse


def main():
    df = pd.read_csv("preprocessed_dataset.csv")
    y_train = df['Name']
    del df['Name']
    del df['Sno']
    del df['Location']
    X = df
    y = y_train

    from sklearn.model_selection import train_test_split

    # Split train test: 70 % - 30 %
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=52)
    d=nn_evaluation(X, X_test, y, y_test)
    return d

def errors():
    df = pd.read_csv("preprocessed_dataset.csv")
    m_train = df['Price']
    del df['Name']
    del df['Sno']
    del df['Location']
    del df['Price']
    Z = df
    m = m_train


    from sklearn.model_selection import train_test_split

    Z_train, Z_test, m_train, m_test = train_test_split(Z, m, test_size=0.3, random_state=52)
    c=get_errors(Z_train, Z_test, m_train, m_test)
    return c
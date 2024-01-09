# author: kantus
# Date: 11/03/2023
# Details:
# 1). For SVM Model, I have used the LinearSVC to calcuate the Accuracy, Precision, Recall and F1 Score.
# 2). Source: https://www.simplilearn.com/tutorials/scikit-learn-tutorial/sklearn-svm-support-vector-machines
# 3). For calculating the Mean Absolute Error, Mean Squared Error and R-Squared Error are calaulted using the DecisionTreeRegressor.

from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, precision_score, accuracy_score, recall_score,confusion_matrix
from DBConfig import DBConnection
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import pandas as pd


def svm_evaluation(X_train, X_test, y_train, y_test):
    db = DBConnection.getConnection()
    cursor = db.cursor()

    svm_clf = LinearSVC()

    svm_clf.fit(X_train, y_train)

    predicted = svm_clf.predict(X_test)

    accuracy = accuracy_score(y_test, predicted)*100

    precision = precision_score(y_test, predicted, average="micro")*100

    recall = recall_score(y_test, predicted, average="micro")*100

    fscore = f1_score(y_test, predicted, average="micro")*100

    
    values = ("SVM", str(accuracy), str(precision), str(recall), str(fscore))
    sql = "insert into evaluations values(%s,%s,%s,%s,%s)"
    cursor.execute(sql, values)
    db.commit()

    print("SVM=", accuracy, precision, recall, fscore)
    return accuracy, precision, recall, fscore


def get_errors(Z_train, Z_test, m_train, m_test):
    svm_clf_mfe = DecisionTreeRegressor()

    svm_clf_mfe.fit(Z_train, m_train)

    y_pred = svm_clf_mfe.predict(Z_test)

    mae = metrics.mean_absolute_error(m_test, y_pred)
    mse = metrics.mean_squared_error(m_test, y_pred)
    rse = metrics.r2_score(m_test, y_pred)
    print('Mean Absolute Error      : ', mae)
    print('Mean Squared  Error      : ', mse)
    print('R Squared Error          : ', rse)
    return mae, mse, rse



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
    d=svm_evaluation(X_train, X_test, y_train, y_test)
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


from flask import Flask, render_template, request,flash
from flask import Response
from matplotlib.figure import Figure
import csv
import cv2
from flask import session
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import numpy as np
import sys
import os
import io
import pandas as pd
from sklearn.model_selection import train_test_split
import random
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt2
import matplotlib.pyplot as plt3
import matplotlib.pyplot as plt4
from DBConfig import DBConnection
from sklearn.ensemble import RandomForestClassifier

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import pickle

app = Flask(__name__)
app.secret_key = "abc"

dict={}
accuracy_list=[]
accuracy_list.clear()
precision_list=[]
precision_list.clear()
recall_list=[]
recall_list.clear()
f1score_list=[]
f1score_list.clear()

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/user")
def user():
    return render_template("user.html")


@app.route("/admin")
def admin():
    return render_template("admin.html")

@app.route("/newuser")
def newuser():
    return render_template("register.html")

@app.route("/prediction")
def prediction():
    return render_template("prediction.html")

@app.route("/adminlogin_check",methods =["GET", "POST"])
def adminlogin():

        uid = request.form.get("unm")
        pwd = request.form.get("pwd")
        if uid=="admin" and pwd=="admin":

            return render_template("admin_home.html")
        else:
            return render_template("admin.html",msg="Invalid Credentials")
@app.route("/adminhime")
def adminhome():
    return render_template("admin_home.html")
    
@app.route("/preprocessing")
def preprocessing():
    return render_template("data_preprocessing.html")



@app.route("/uploaddataset")
def uploaddataset():
    return render_template("upload.html")

@app.route("/uploaddatasetaction", methods =["GET", "POST"])
def uploaddatasetaction():
    f=open('s.txt','w')
    if True:
        query = "delete from dataset"
        db = DBConnection.getConnection()
        cursor = db.cursor()
        cursor.execute(query)
        db.commit()
        cursor = db.cursor()
            
        try:
            fname = 'preprocessed_dataset.csv'
            import csv
            sql = "insert into dataset values(%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)"

            with open(fname, newline='') as csvfile:
                reader = csv.reader(csvfile, delimiter=',')
                next(reader)
                for row in reader:
                    f.write(str(row))


                    values = (row[0],row[1],row[2],row[3],row[4],row[5],row[6],row[7],row[8],row[9],row[10],row[11],row[12])
                    cursor.execute(sql,values)
                db.commit()

        
        except Exception as e:
            f.write(str(e))
            
    return render_template("upload.html", msg="Dataset Uploaded Completed..!")

@app.route("/perevaluations")
def perevaluations():
    #accuracy_graph()
    #precision_graph()
    #recall_graph()
    #f1score_graph()

    return render_template("metrics.html")




@app.route("/data_preprocessing" ,methods =["GET", "POST"] )
def data_preprocessing():
    fname = request.form.get("file")
    df = pd.read_csv(fname)
    df1 = df.dropna()
    df1.to_csv("preprocessed_dataset.csv", index=False)
    return render_template("data_preprocessing.html",msg="Data Preprocessing Completed..!")


@app.route("/evaluations" )
def evaluations():

    lr_list=[]
    dt_list = []
    knn_list = []
    svm_list = []
    nn_list = []
    metrics=[]

    from KNN import main
    accuracy_knn, precision_knn, recall_knn, fscore_knn=main()
    knn_list.append("KNN")
    knn_list.append(accuracy_knn)
    knn_list.append(precision_knn)
    knn_list.append(recall_knn)
    knn_list.append(fscore_knn)

    from NN import main
    accuracy_nn, precision_nn, recall_nn, fscore_nn=main()
    
    nn_list.append("NN")
    nn_list.append(accuracy_nn)
    nn_list.append(precision_nn)
    nn_list.append(recall_nn)
    nn_list.append(fscore_nn)

    from LR import main
    accuracy_lr, precision_lr, recall_lr, fscore_lr=main()

    lr_list.append("RF")
    lr_list.append(accuracy_lr)
    lr_list.append(precision_lr)
    lr_list.append(recall_lr)
    lr_list.append(fscore_lr)

    from SVM import main
    accuracy_svm, precision_svm, recall_svm, fscore_svm =main()

    svm_list.append("SVM")
    svm_list.append(accuracy_svm)
    svm_list.append(precision_svm)
    svm_list.append(recall_svm)
    svm_list.append(fscore_svm)

    





    metrics.clear()
    metrics.append(knn_list)
    metrics.append(nn_list)
    metrics.append(lr_list)
    metrics.append(svm_list)
    

    return render_template("evaluations.html", evaluations=metrics)


def accuracy_graph():
    db = DBConnection.getConnection()
    cursor = db.cursor()
    accuracy_list.clear()

    cursor.execute("select accuracy from evaluations")
    acdata=cursor.fetchall()

    for record in acdata:
        accuracy_list.append(float(record[0]))

    height = accuracy_list
    print("height=",height)
    bars = ('KNN','NN','LR','SVM')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height, color=['red', 'green', 'blue', 'orange','yellow'])
    plt.xticks(y_pos, bars)
    plt.xlabel('Algorithms')
    plt.ylabel('Accuracy')
    plt.title('Analysis on Accuracies')
    plt.savefig('static/accuracy.png')
    plt.clf()
    #plt.savefig('accuracy.png')

    return ""


def precision_graph():
    db = DBConnection.getConnection()
    cursor = db.cursor()

    cursor.execute("select precesion from evaluations")
    pdata = cursor.fetchall()

    precision_list.clear()
    for record in pdata:
        precision_list.append(float(record[0]))

    height = precision_list
    print("pheight=",height)
    bars = ('KNN','NN','LR','SVM')
    y_pos = np.arange(len(bars))
    plt2.bar(y_pos, height, color=['green', 'brown', 'violet', 'blue','red'])
    plt2.xticks(y_pos, bars)
    plt2.xlabel('Algorithms')
    plt2.ylabel('Precision')
    plt2.title('Analysis on Precisions')
    plt2.savefig('static/precision.png')
    plt2.clf()
    return ""

def recall_graph():
    db = DBConnection.getConnection()
    cursor = db.cursor()
    recall_list.clear()
    cursor.execute("select recall from evaluations")
    recdata = cursor.fetchall()

    for record in recdata:
        recall_list.append(float(record[0]))

    height = recall_list

    bars = ('KNN','NN','LR','SVM')
    y_pos = np.arange(len(bars))
    plt3.bar(y_pos, height, color=['orange', 'cyan', 'gray', 'violet','green'])
    plt3.xticks(y_pos, bars)
    plt3.xlabel('Algorithms')
    plt3.ylabel('Recall')
    plt3.title('Analysis on Recall')
    plt3.savefig('static/recall.png')
    plt3.clf()
    return ""


def f1score_graph():
    db = DBConnection.getConnection()
    cursor = db.cursor()
    f1score_list.clear()

    cursor.execute("select f1score from evaluations")
    fsdata = cursor.fetchall()

    for record in fsdata:
        f1score_list.append(float(record[0]))

    height = f1score_list
    print("fheight=",height)
    bars = ('KNN','NN','LR','SVM')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height, color=['gray', 'green', 'orange', 'brown','blue'])
    plt.xticks(y_pos, bars)
    plt.xlabel('Algorithms')
    plt.ylabel('F1-Score')
    plt.title('Analysis on F1-Score')
    plt4.savefig('static/f1score.png')
    plt4.clf()
    return ""


@app.route("/user_register",methods =["GET", "POST"])
def user_register():
    try:
        sts=""
        name = request.form.get('name')
        pwd = request.form.get('pwd')
        mno = request.form.get('mno')
        email = request.form.get('email')
        database = DBConnection.getConnection()
        cursor = database.cursor()
        sql = "select count(*) from register where userid='" + email + "'"
        cursor.execute(sql)
        res = cursor.fetchone()[0]
        if res > 0:
            sts = 0
        else:
            sql = "insert into register values(%s,%s,%s,%s,%s)"
            values = (name,email, pwd,email,mno)
            cursor.execute(sql, values)
            database.commit()
            sts = 1

        if sts==1:
            return render_template("user.html", msg="Registered Successfully..! Login Here.")


        else:
            return render_template("register.html", msg="User name already exists..!")



    except Exception as e:
        print(e)

    return ""


@app.route("/userlogin_check",methods =["GET", "POST"])
def userlogin_check():

        uid = request.form.get("unm")
        pwd = request.form.get("pwd")

        database = DBConnection.getConnection()
        cursor = database.cursor()
        sql = "select count(*) from register where userid='" + uid + "' and passwrd='" + pwd + "'"
        cursor.execute(sql)
        res = cursor.fetchone()[0]
        if res > 0:
            session['uid'] = uid

            return render_template("user_home.html")
        else:

            return render_template("user.html", msg2="Invalid Credentials")

        return ""

@app.route("/userhome")
def userhome():
    return render_template("user_home.html")

@app.route("/predictionaction", methods =["GET", "POST"])
def predictionaction():
    year = request.form.get("year")
    km = request.form.get("km")
    fuel = request.form.get("fuel")
    tra= request.form.get("tra")
    owner = request.form.get("owner")
    mil = request.form.get("mil")
    cc = request.form.get("cc")
    bhp = request.form.get("bhp")
    seats = request.form.get("seats")





    df = pd.read_csv("preprocessed_dataset.csv")

    y_train = df['Name']
    del df['Name']
    del df['Sno']
    del df['Price']
    del df['Location']
    X = df
    y = y_train

    print(df.columns)



    X_test=[[float(year),float(km),float(fuel),float(tra),float(owner),float(mil),float(cc),float(bhp),float(seats)]]
    print(X_test)

    model=''



    if os.path.exists('model.pkl'):
        with open('model.pkl', 'rb') as f:
            model = pickle.load(f)
    else:
        from sklearn.neural_network import MLPClassifier
        model = MLPClassifier()
        model.fit(X, y)
        with open('model.pkl', 'wb') as f:
            pickle.dump(model, f)


    predicted = model.predict(X_test)
    result = predicted[0]
    print("res=", result)
    database = DBConnection.getConnection()
    cursor = database.cursor()
    
    sql = "select * from dataset where car_name like '%" + result + "%' "
    cursor.execute(sql)
    res = cursor.fetchall()
    ares=res
    name=''
    cost=0
    for r1 in res:
        name=r1[1]
        cost=float(r1[12])


    cost1=cost-0.1
    cost2=cost+0.1

    cursor = database.cursor()
    
    #sql = "select * from dataset where Price  BETWEEN " + str(cost1) + " AND " + str(cost2) + " "
    sql = "select * from dataset where Price="+str(cost)+""
    print(sql)
    cursor.execute(sql)
    res = cursor.fetchall()
    metrics=[]


    name=[]
    year=[]
    driven=[]
    fuel=[]
    seats=[]
    mil=[]
    eng=[]

    cost=[]

    for r1 in res:
        name.append(r1[1])
        year.append(r1[3])
        driven.append(r1[4])
    

        if r1[5]=='1': 
            fuel.append('Diesel')

        elif r1[5]=='2':
            fuel.append('Petrol')

        elif r1[5]=='3':
            fuel.append('CNG')
        else:
            fuel.append('LPG')
    

        
        mil.append(r1[8])
        eng.append(r1[9])
        seats.append(r1[11])
        cost.append(r1[12])
        
    metrics.clear()
    metrics.append(name)
    metrics.append(year)
    metrics.append(driven)
    metrics.append(fuel)
    metrics.append(mil)
    metrics.append(eng)
    metrics.append(seats)
    metrics.append(cost)
    

    return render_template("predictionres.html",data1=ares,data=res)


if __name__ == '__main__':
    app.run(host="localhost", port=1234, debug=True)

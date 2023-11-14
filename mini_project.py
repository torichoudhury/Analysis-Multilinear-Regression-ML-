#importing modules
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

"""Data Preprocessing 
1) Import the dataset (using pandas library).
2) Check for missing values in the dataset and handle them appropriately.

"""
def preprocess(dataset,path):
    #importing dataset from csv 
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values
    #taking care of missing values 
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer.fit(X[:, 1:3])
    X[:, 1:3] = imputer.transform(X[:, 1:3])
    
    #writing a new csv 
    processed= pd.DataFrame(data=np.column_stack((X, y)), columns=dataset.columns)
    processed.to_csv(path, index=False)
    




#visualising data in form of graphs(user preferanced)
def visualising(xaxis,yaxis,data):
    xlist=[]
    ylist=[]
    for index, row in data.iterrows():
        xlist.append(row[xaxis])
        ylist.append(row[yaxis])
    plt.plot(xlist, ylist)
    plt.title(" ")
    plt.show()

def regressionmodel(dataset):
    X = dataset.iloc[:, 1:-1].values
    y = dataset.iloc[:, -1].values
    #splitting data into training set and test set
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    #training model on the training set 
    
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)
    #predicting the Test set results
    y_pred = regressor.predict(X_test)
    np.set_printoptions(precision=2)
    print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))


#main function 
def main():
    data = pd.read_csv("Sample(2023)h.csv")
    desdata = "Processed.csv"
    while True:
        print("\nChoose the number to perform an action\n1-Data Pre-processing.\n2-Visualize Graphs.\n3-Predicting data\n4-Exit.")
        choose=int(input("Enter your choice: "))
        #try:
        if choose==1:
            preprocess(data,desdata)
        elif choose==2:
            xaxis=input("Enter the name of the first axis variable :")
            yaxis=input("Enter the name of the second axis variable :")
            visualising(xaxis,yaxis,pd.read_csv(desdata))
        elif choose==3:
            regressionmodel(pd.read_csv(desdata))
        elif choose==4:
            exit()
        else:
            print("Please Enter a Valid Number")
            continue
        #except ValueError:
            #eak
main()
            
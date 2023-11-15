"""The following block of code works on 4 primary functions 
1. Preprocess(dataset,path): This function is to check for any missing values and replacing it with the mean of the column
2. Visualising(xaxis,yaxis,data): It plots the graph based on the the name of the column provided
3. Regressionmodel(dataset): This is a Machine Learning model for predicting Active Power by learning the patterns from existing dataset
    Multi-Linear Regression model is used. 
    For demonstration Dataset of Independent variable is taken from the existing dataset by splitting it.
4. main(): All the above functions can be callede by user under this function 

Note: The sample csv 'Sample(2023)h.csv' has missing values in Vrms and Irms column given to demonstrate Preprocess() function,
and hence the others functions won't work until Preprocess() function is run atleast once.

Thank you.
"""

#importing modules
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


#to preprocess data by substituting the missing values 
def preprocess(dataset,path):
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    #taking care of missing values 
    imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
    imputer.fit(X[:, 1:3])
    X[:, 1:3] = imputer.transform(X[:, 1:3])
    
    #writing a new csv 
    processed= pd.DataFrame(data=np.column_stack((X, y)), columns=dataset.columns)
    processed.to_csv(path, index=False)
    

#visualising data in form of graphs
def visualising(xaxis,yaxis,data):
    xlist=[]
    ylist=[]
    for index, row in data.iterrows():
        xlist.append(row[xaxis])
        ylist.append(row[yaxis])
    #plotting graphs 
    plt.plot(xlist, ylist)
    plt.title(" ")
    plt.xlabel(xaxis)
    plt.xticks(rotation=45)
    plt.ylabel(yaxis)
    plt.grid(True)
    plt.legend()
    plt.show()

def regressionmodel(dataset):
    print("\nThis regression model can predict the value of 'Active Power' by analysing the 'Vrms','Irms and 'Frequency'")
    print("\nTo demonstrate the model, the dataset is split into Training set and Test set")
    print("The model is trained on the training set and used on the test set and the output is compared with actual values")
    X = dataset.iloc[:, 1:-1].values
    y = dataset.iloc[:, -1].values

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    print("\nTest set of Independent variables(Vrms,Irms and Frequency respectively)","\n",X_test,"\n","Actual values of Active power corresponding to Test set \n",y_test)

    # Linear Regression Model Training
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    # Prediction
    y_pred = regressor.predict(X_test)
    print("\nPredicted values of Active power corresponding to the test set\n",y_pred,"\n")
    
    try:
        wish=int(input("Enter any number to visualize the comparison on a graph "))
        # Plotting the predicted vs. actual values
        plt.scatter(y_test, y_pred, color='green')
        plt.title('Active Power')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.show()
    except ValueError:
        print("Returning to main menu")

    
#main function 
def main():
    data = pd.read_csv("Sample(2023)h.csv")
    desdata = "Processed.csv"
    
    while True:
        print("\nChoose the number to perform an action\n1-Data Pre-processing.\n2-Visualize Graphs.\n3-Predicting data\n4-Exit.")
        print("Please note that 2 and 3 does not work until 1 is used atleast once. \n ")

        try:
            choose=int(input("Enter your choice: "))
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
        except ValueError:
            print("Please Enter a Valid input and try again ")
            continue

main()
            

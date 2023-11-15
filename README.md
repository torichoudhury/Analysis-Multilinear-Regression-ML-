

## Introduction
Welcome to the Power Consumption Analysis tool! This Python script provides functionalities for data preprocessing, visualization, and regression analysis on a dataset related to power consumption. Below, you'll find information on the primary functions and how to use them.

## Table of Contents
- [Functions](#functions)
    - [1. Preprocess](#1-preprocess)
    - [2. Visualizing](#2-visualizing)
    - [3. Regression Model](#3-regression-model)
    - [4. Main](#4-main)
- [Note](#note)
- [Dependencies](#dependencies)
- [Usage](#usage)

## Functions

### 1. Preprocess
```python
def preprocess(dataset, path):
    """
    This function preprocesses the dataset by replacing missing values with the mean of the column.
    
    Parameters:
    - dataset: Pandas DataFrame, the original dataset
    - path: str, path to save the processed dataset
    """
    # Implementation details are in the code.
```

### 2. Visualizing
```python
def visualizing(xaxis, yaxis, data):
    """
    This function visualizes data in the form of graphs based on the provided column names.
    
    Parameters:
    - xaxis: str, the name of the first axis variable
    - yaxis: str, the name of the second axis variable
    - data: Pandas DataFrame, the dataset for visualization
    """
    # Implementation details are in the code.
```

### 3. Regression Model
```python
def regressionmodel(dataset):
    """
    This function implements a Multi-Linear Regression model for predicting Active Power based on existing data.
    
    Parameters:
    - dataset: Pandas DataFrame, the dataset for regression analysis
    """
    # Implementation details are in the code.
```

### 4. Main
```python
def main():
    """
    The main function that provides a menu for the user to choose different actions:
    1. Data Pre-processing
    2. Visualize Graphs
    3. Predicting Data
    4. Exit
    """
    # Implementation details are in the code.
```

## Note
The sample CSV file 'Sample(2023)h.csv' provided in this repository has missing values in the 'Vrms' and 'Irms' columns to demonstrate the `Preprocess` function. Other functions won't work until the `Preprocess` function is executed at least once.

## Dependencies
Ensure you have the following Python modules installed:
- `numpy`
- `matplotlib`
- `pandas`
- `scikit-learn`


Follow the on-screen menu to execute different functionalities. Make sure to run the data preprocessing before using other functions.

Thank you for using the Power Consumption Analysis tool! If you encounter any issues or have suggestions, feel free to open an issue on GitHub.

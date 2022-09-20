import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LinearRegression  
from sklearn import linear_model
from sklearn import metrics  
import pickle

with open('marks', 'rb') as f:
    model = pickle.load(f)

a = model.predict([[1.5]])

print(a)
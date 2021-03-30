from django.http import HttpResponse
from django.shortcuts import render
from django.contrib.staticfiles.storage import staticfiles_storage

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def home(request):
    return render(request, 'index.html', {"predicted": ""})

def predict(request):

    md = str(request.GET['md'])
    if md == 'Alpha':
        md = 0
    elif md == 'Delta':
        md = 1
    elif md == 'Sigma':
        md = 2
    else:
        md = 3

    yr = int(request.GET['yr'])

    fl = str(request.GET['fl'])
    if fl == 'Diesel':
        fl = 0
    else:
        fl = 1
   
    km = int(request.GET['km'])

    rawdata = staticfiles_storage.path('scross_updated.csv')
    df = pd.read_csv(rawdata)

    x = df[["Model", "Year", "Fuel", "Kilometer"]]
    y = df["Price in Rs"]

    x_train,x_test,y_train,y_test = train_test_split(x, y, train_size = 0.8 , random_state = 27)

    lr = LinearRegression()
    lr.fit(x_train, y_train)

    ip = np.array([[md, yr, fl, km]])

    y_pred = lr.predict(ip)

    res = float(y_pred)
    #r = format(res, ".2f")

    re = '{:,.2f}'.format(res)
    
    #model.predict([[5.7,2.6,3.5,1.0]])

    return render(request, 'index.html', {'predicted': re, 'md': md, 'yr': yr, 'fl': fl, 'km': km})
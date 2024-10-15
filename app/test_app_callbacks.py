#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 10 00:29:55 2024

@author: qb
"""
import pandas as pd
import numpy as np


import dash
from dash import html
# from app import dash
from contextvars import copy_context
from dash._callback_context import context_value
from dash._utils import AttributeDict

from app import process_form, table_processing

#writting two test cases to check the output 
#taking sub sample of data to check the output

df = pd.read_csv('data/Cars.csv')

df['mileage']=df['mileage'].apply(lambda x : str(x).split()[0]).astype('float')
df['engine'] = df['engine'].apply(lambda x: str(x).split()[0]).astype('float')
df['max_power'] = df['max_power'].apply(lambda x: str(x).split()[0])
df.drop(df[df['max_power'] == 'bhp'].index, inplace=True)
df['max_power'] = df['max_power'].astype('float')
df.drop(columns = 'torque', inplace=True)
df['name'] = df['name'].apply(lambda x:x.split()[0])
cols = df.columns.values
df['owner'].loc[df[(df['owner'] == 'Fourth & Above Owner') | (df['owner'] == 'Third Owner')].index] = 'others'
df.drop(df[df['owner']=='Test Drive Car'].index, inplace=True)
df_ = df.copy(deep=True)
df.drop(columns = {'selling_price'}, inplace=True)
df.drop(df[(df['fuel'] == 'LPG') | (df['fuel'] == 'CNG')].index, inplace=True)
max_models = df.groupby('name')['year'].max()
max_seat = df['seats'].max()
cols = df.columns


#make sure that non of the prediction is na/na
#checking for multiple inputs of predictions

def random_sample(size):
   idx = np.random.uniform(0,df.shape[0], size)
   idx = np.apply_along_axis(np.round, axis=0, arr = idx)
   return np.array(idx, dtype='int')

def test_out_callback_example():
    def run_callback(trigger, n_clicks):
        context_value.set(AttributeDict(**{"triggered_inputs": [trigger]}))
        #try size 1
        idx = random_sample(1)
        #try size more than 1
        data = df.loc[idx]
        em=[('em'+col, 'children') for col in cols]
        
        return table_processing(data, [],n_clicks,em, testing=True)
    
    ctx = copy_context()
    trigger = {"prop_id": "submit", 
               "prop_id": "clean"}
    n_clicks = 1
    output = ctx.run(run_callback, trigger, n_clicks)
    print(output)
    #some prediction is returnred by the model
    # assert pred.shape[0] > 1
    # assert prob.shape[0] > 1
    # trigger = {"prop_id": "clean"}
    # n_clicks = 1
    # output = ctx.run(run_callback, trigger, n_clicks)
    # assert output == 1
    



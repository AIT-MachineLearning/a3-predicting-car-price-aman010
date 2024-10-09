#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 22 18:10:37 2024

@author: qb
"""
import numpy as np 
import mlflow
from mlflow import MlflowClient
import time
import pickle
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_score, recall_score, accuracy_score, f1_score


class LogisticRegression:
    kfold = StratifiedKFold(n_splits=5)
    
    def __init__(self, k, n, method,  num_epochs ,batch_size,l2=None ,cv=kfold,alpha = 0.001, max_iter=5000):
        self.k = k
        self.n = n
        self.cv = cv
        self.alpha = alpha
        self.max_iter = max_iter
        self.method = method
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.l2 = l2
        
        
    def fit(self, X, dummies,Y):
        
        self.W = np.random.rand(self.n, self.k)
        self.losses  = []
        self.weights = []
        self.train_loss = []
        self.valid_loss = []
        self.loss_across_batches = {'train_loss':[], 'val_loss':[]}
        for fold, (train_idx, val_idx) in enumerate(self.cv.split(X, Y)):
            X_cross_train = X[train_idx]
            y_cross_train = dummies[train_idx]
            y_train = Y[train_idx]
            X_cross_val   = X[val_idx]
            y_cross_val   = dummies[val_idx]
            y_val = Y[val_idx]
            
                
            client = MlflowClient()
            name = "Logistic_regression_st125490"
            src_name = "Logistic_regression-staging"
        

            mlflow.set_experiment(name)
            
            with mlflow.start_run(run_name=f"Fold-{fold}", tags={"group":"Fold:{}_Method:{}".format(fold, self.method), "project":f"{name}"},nested=True):

                run = mlflow.active_run()
                src_uri = f"runs:/{run.info.run_id}/logit_model"
                
                if self.l2:
                    params = {"method": self.method, "l2":self.l2,"lr":self.alpha}
                else:
                    params = {"method":self.method, "lr":self.alpha}
                    
    
                mlflow.log_params(params=params)
                
                client.log_artifact(run.info.run_id, '/home/qb/ML_Assignment_3/mlflow')

                
                try:
                    model = client.get_registered_model(src_name)
                except mlflow.exceptions.MlflowException:
                    client.create_registered_model(src_name)
                    mv_src = client.create_model_version(src_name, src_uri, run.info.run_id)
                        
                for epoch in range(self.num_epochs):
                    if self.method == "batch":
                        start_time = time.time()
                        for i in range(self.max_iter):
                            loss, grad =  self.gradient(X, Y)
                            self.losses.append(loss)
                            self.W = self.W - self.alpha * grad
                            if i % 500 == 0:
                                print(f"Loss at iteration {i}", loss)
                
                    elif self.method == "minibatch":
                        for i in range(0, X_cross_train.shape[0], self.batch_size):
                            #ix = np.random.randint(0, X.shape[0]) #<----with replacement
                            batch_X = X_cross_train[i:i+self.batch_size, :]
                            batch_Y = y_cross_train[i:i+self.batch_size]
                            #is gradient updating for same thea values
                            loss, grad = self.gradient(batch_X, batch_Y)
                            self.losses.append(loss)
                            self.W = self.W - self.alpha * grad
                          
                            if epoch == self.num_epochs-1:
                                self.weights.append(self.W)
                                
                                h= self.predict(batch_X, prob=True)
                                loss_=self.log_liklehood(batch_X, batch_Y, h)
                                self.train_loss.append((loss_, fold))
                                
                                
                                h = self.predict(X_cross_val, prob=True)
                                loss_ = self.log_liklehood(X_cross_val, y_cross_val, h)
                                self.valid_loss.append((loss_, fold))                            
                                # val_x = X_corss_val[i:i+self.batch_size]
                                # val_y = y_cross_val[i:i+self.batch_size]
                                #h = self.h_theta(batch_Y, self.W)
                                #loss_ = -np.sum(batch_Y*np.log(h))/m
                            # if i % 500 == 0:
                            #     print(f"Loss at iteratioin", loss)
                                
                        
                     
                        
                        train_Y_pred=self.predict(X_cross_train, prob=True)
                        class_train_pred = self.predict(X_cross_train ,prob=False)
                        train_loss = self.log_liklehood(X_cross_train, y_cross_train, train_Y_pred)
                        
                        valid_Y_pred = self.predict(X_cross_val, prob=True)
                        class_pred = self.predict(X_cross_val, prob=False)
                        val_loss = self.log_liklehood(X_cross_train, y_cross_val, valid_Y_pred)
                        
                        mlflow.log_metric(key="train_loss", value=train_loss, step=epoch)
                        mlflow.log_metric(key="val_loss", value=val_loss, step=epoch)
    
                    
    
                        #compute and log the precision, recall, accuracy for the model
    
                    elif self.method == "sto":
                        start_time = time.time()
                        list_of_used_ix = []
                        for i in range(self.max_iter):
                            idx = np.random.randint(X.shape[0])
                            while i in list_of_used_ix:
                                idx = np.random.randint(X.shape[0])
                                X_train = X[idx, :].reshape(1, -1)
                                Y_train = Y[idx]
                                loss, grad = self.gradient(X_train, Y_train)
                                self.losses.append(loss)
                                self.W = self.W - self.alpha * grad
                            
                            list_of_used_ix.append(i)
                            if len(list_of_used_ix) == X.shape[0]:
                                list_of_used_ix = []
                            if i % 500 == 0:
                                print(f"Loss at iteration {i}", loss)
                            print(f"time taken: {time.time() - start_time}")
                        
                #perform prediction across all the thetas
                #avearge across the mini prediction
                
                # classes=np.argmax(self.h_theta(X_cross_train, updated_weights), axis=1)
                #prediction for the validation loss
                if epoch == self.num_epochs-1:
                    #iterate accross all the min batches weights
                    fold_loss_train = []
                    m = X_cross_train.shape[0]
                    for weight in self.weights:
                        h=self.h_theta(X_cross_train, weight)
                        loss_ = -np.sum(y_cross_train*np.log(h))/m
                        fold_loss_train.append(loss_)
                    print('train loss', np.mean(fold_loss_train))
                    self.loss_across_batches['train_loss'].append(np.mean(fold_loss_train))
                    
                    fold_loss_valid = []
                    for weight in self.weights:
                        h =self.h_theta(X_cross_val, weight)
                        loss_ = -np.sum(y_cross_val*np.log(h))/m
                        fold_loss_valid.append(loss_)
                    print('valid loss', np.mean(fold_loss_valid))
                    self.loss_across_batches['val_loss'].append(np.mean(fold_loss_valid))
                        #avarge the prediction across the min batches
                        
                    
                y_hat_val=self.predict(X_cross_val)
                
                train_precision =precision_score(y_train, class_train_pred, average='macro')
                valid_precision=precision_score(y_val, class_pred, average='macro')

                train_recall = recall_score(y_train, class_train_pred, average='macro')
                valid_recall = recall_score(y_val, class_pred, average='macro')
                
                #accuracy
                train_acc = accuracy_score(y_train, class_train_pred)
                valid_acc = accuracy_score(y_val, class_pred)
                
                
                #f1 score                            
                train_f1=f1_score(y_train, class_train_pred, average='macro')   
                valid_f1 = f1_score(y_val, class_pred, average='macro')
                    
                mlflow.log_metric(key="train_precision", value=train_precision, )
                mlflow.log_metric(key="validation_precision", value=valid_precision)
                mlflow.log_metric(key="train_recall", value=train_recall)
                mlflow.log_metric(key="valid_recall", value=valid_recall)
                mlflow.log_metric(key="train_acc", value=train_acc)
                mlflow.log_metric(key="valid_acc", value=valid_acc)
                mlflow.log_metric(key="f1-score-train", value=train_f1)
                mlflow.log_metric(key='valid_f1', value=valid_f1)
                # val_loss_new = self.log_liklehood(y_hat_val, y_cross_val)
                # print('validation_loss:', val_loss_new)
        mlflow.sklearn.log_model(model, artifact_path='model')
    # #######
        mlflow.end_run()

    
    def gradient(self, X, Y):
        m = X.shape[0]
        h = self.h_theta(X, self.W) 
        error = h - Y.reshape(m, -1)
        if self.l2:
            loss = -np.sum(Y*np.log(h)) / m  + self.l2 * np.sum(self.W)
            grad = self.softmax_grad(X, error) + self.l2 * self.W
        else:
            loss = -np.sum(Y*np.log(h))/m
            grad = self.softmax_grad(X, error)
        return loss, grad

    def log_liklehood(self,X ,Y, h):
        m=X.shape[0]
        loss_ = -np.sum(Y*np.log(h))/m
        return loss_

    def get_run_id(self):
        return 
        
    def softmax(self, theta_t_x):   
        return np.exp(theta_t_x) / np.sum(np.exp(theta_t_x), axis=1, keepdims=True)

    def softmax_grad(self, X, error):
        return  X.T @ error

    def h_theta(self, X, W):
        '''
        Input:
            X shape: (m, n)
            w shape: (n, k)
        Returns:
            yhat shape: (m, k)
        '''
        return self.softmax(X @ W) #e^-x*W/sum(e-x*W)
    
    def precision(self, tp, fn):
        return tp/(tp+fn)
    
    def recall(self, tp, fp):
        return tp/(tp+fp)
    
    def accuracy(self, tp, fp, tn, fn):
        return (tp + tn)/(tp + tn + fp + fn)
    
    def get_train_val_prediction(self):
        return self.loss_across_batches
    

    def predict(self, X_test, prob=False):
        if prob == True:
            pred_prb=self.h_theta(X_test, self.W)
            return pred_prb
        else:
            pred_cl =  np.argmax(self.h_theta(X_test, self.W), axis=1)
            return pred_cl
    
    

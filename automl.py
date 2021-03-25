# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 11:06:34 2021

@author: ama99
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import RFECV
from sklearn.model_selection import cross_val_score, train_test_split, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib
from sklearn.metrics import log_loss, roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
from keras.layers import Dense, Dropout
from keras.models import Sequential
import lightgbm as lgb

# tensorflowの警告抑制
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

class BinaryClassifier:
    """
    ２値分類用のautoml
    """
    scores = {}
    pred_probas = {}
    models = {}
    best_ = {}
    train_columns = None
    
    def __init__(self, name=''):
        pass
    
    def fit(self, tr_x, tr_y, categorical_features = None, scoring = 'roc_auc'):
        self.categorical_features = categorical_features
        self.scoring = scoring
        # preprocessing-1: one-hot encoding
        if self.categorical_features:
            X_ohe = pd.get_dummies(tr_x, dummy_na=True, columns = self.categorical_features)
            X_ohe = X_ohe.dropna(axis=1, how='all')
            self.train_columns = X_ohe.columns.values
        else:
            X_ohe = tr_x.copy()
            self.train_columns = X_ohe.columns.values
        # preprocessing-2: null imputation
        self.imp = SimpleImputer()
        self.imp.fit(X_ohe)
        X_ohe = pd.DataFrame(self.imp.transform(X_ohe), columns = self.train_columns)
        
        # preprocessing-3: feature selection
        selector = RFECV(estimator=RandomForestClassifier(n_estimators=100,random_state=0), step=0.05)
        selector.fit(X_ohe, tr_y)
        self.selected = self.train_columns[selector.support_]
        X_ohe_selected = selector.transform(X_ohe)
        X_ohe_selected = pd.DataFrame(X_ohe_selected, columns = self.selected )
        
        skmodels = {
                "GBC":GradientBoostingClassifier(random_state=1),
                "LGBM":lgb.LGBMClassifier(objective='binary',random_state=1),
                "RFC":RandomForestClassifier(random_state=1),
                "LR":LogisticRegression(C=1.0)
                }
        
        for ml_name, ml_method in skmodels.items():
            print(ml_name)
            clf = Pipeline([('scl',StandardScaler()),('est',ml_method)])
            clf.fit(X_ohe_selected, tr_y)
            self.models[ml_name] = clf
            joblib.dump(clf, './model/'+ ml_name + '.pkl')
#            self.pred_probas[ml_name] = clf.predict_proba(test_x)
            self.scores[ml_name] = np.average(cross_val_score(clf, X_ohe_selected, tr_y, scoring=scoring, cv=5))
        
        self._nn(X_ohe_selected, tr_y, scoring = scoring)
        
        print(self.scores)
        df = pd.DataFrame.from_dict(self.scores, orient='index', columns = ["score"]).sort_values("score", ascending=False)
        self.best_ = {"name": df.index[0], "model": self.models[df.index[0]] , "score": df.iat[0,0]}
        
    def _nn(self, tr_x, tr_y, scoring = 'roc_auc'):
        
        def _nnmodel(X_train, y_train):
            # -----------------------------------
            # ニューラルネットの実装
            # -----------------------------------
            
            # データのスケーリング
            nn_scaler = StandardScaler()
            X_train = nn_scaler.fit_transform(X_train)
            
            # ニューラルネットモデルの構築
            model = Sequential()
            model.add(Dense(256, activation='relu', input_shape=(X_train.shape[1],)))
            model.add(Dropout(0.2))
            model.add(Dense(256, activation='relu'))
            model.add(Dropout(0.2))
            model.add(Dense(1, activation='sigmoid'))
            
            model.compile(loss='binary_crossentropy',
                          optimizer='adam', metrics=["accuracy"])
            
            # 学習の実行
            # -----------------------------------
            # アーリーストッピング
            # -----------------------------------
            from keras.callbacks import EarlyStopping
            
            # アーリーストッピングの観察するroundを20とする
            # restore_best_weightsを設定することで、最適なエポックでのモデルを使用する
            batch_size = 128
            epochs = 100
            early_stopping = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
            
            history = model.fit(X_train, y_train,
                                batch_size=batch_size, epochs=epochs,
                                verbose=1, validation_split=0.2, callbacks=[early_stopping])
            return model, nn_scaler
        

        #cvでAUC計算
        nn_score=[]
        # define X-fold cross validation
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)
        for train, test in kfold.split(tr_x, tr_y):
            X_train = tr_x.iloc[train]
            X_test = tr_x.iloc[test]
            y_train = tr_y.iloc[train]
            y_test = tr_y.iloc[test]
            
            model, nn_scaler = _nnmodel(X_train, y_train)
            X_test = nn_scaler.transform(X_test)
            if scoring == 'roc_auc':
                pred = model.predict_proba(X_test)
                nn_score.append(roc_auc_score(y_test, pred))
            elif scoring == 'accuracy':
                pred = model.predict(X_test)
                nn_score.append(accuracy_score(y_test, pred))
            elif scoring == 'precision':
                pred = model.predict(X_test)
                nn_score.append(precision_score(y_test, pred))
            elif scoring == 'f1':
                pred = model.predict(X_test)
                nn_score.append(f1_score(y_test, pred))
            elif scoring == 'recall':
                pred = model.predict(X_test)
                nn_score.append(recall_score(y_test, pred))
            elif scoring == 'log_loss':
                pred = model.predict(X_test)
                nn_score.append(log_loss(y_test, pred))
        self.scores["NN"] = np.average(nn_score)
        
        #全データを使用して学習
        model, nn_scaler = _nnmodel(tr_x, tr_y)
        self.models["NN"] = [nn_scaler, model]
        joblib.dump([nn_scaler, model], './model/NN.pkl')
        
    def predict_proba(self, test_x, how = "best"):
        if self.categorical_features:
            Xs_ohe = pd.get_dummies(test_x, dummy_na=True, columns = self.categorical_features)
            
        else:
            Xs_ohe = test_x.copy()
        cols_m = pd.DataFrame(None, columns = self.train_columns, dtype=float)
        
        # consistent with columns set
        Xs_exp = pd.concat([cols_m, Xs_ohe])
        Xs_exp.loc[:,list(set(self.train_columns)-set(Xs_ohe.columns.values))] = \
            Xs_exp.loc[:,list(set(self.train_columns)-set(Xs_ohe.columns.values))].fillna(0, axis=1)
        Xs_exp = Xs_exp.drop(list(set(Xs_ohe.columns.values)-set(self.train_columns)), axis=1)
        
        
        # re-order the score data columns
        Xs_exp = Xs_exp.reindex(self.train_columns, axis=1)
        Xs_exp = pd.DataFrame(self.imp.transform(Xs_exp), columns = self.train_columns)
        Xs_exp_selected = Xs_exp.loc[:, self.selected]
        
        if how == "best":
            if self.best_["name"] == "NN":
                Xs_exp_selected = self.best_["model"][0].transform(Xs_exp_selected)
                score = pd.DataFrame(self.best_["model"][1].predict_proba(Xs_exp_selected)[:,1], columns=['pred_score'])
            else:
                score = pd.DataFrame(self.best_["model"].predict_proba(Xs_exp_selected)[:,1], columns=['pred_score'])
        elif how == "NN":
            Xs_exp_selected = self.models["NN"][0].transform(Xs_exp_selected)
            score = pd.DataFrame(self.models["NN"][1].predict_proba(Xs_exp_selected), columns=['pred_score'])
        elif how == "all":
            score = []
            for ml_name, ml_model in self.models.items():
                if ml_name == "NN":
                    Xs_exp_selected = self.models[ml_name][0].transform(Xs_exp_selected)
                    score.append(pd.DataFrame(self.models[ml_name][1].predict_proba(Xs_exp_selected), columns=[ml_name]))
                else:
                    score.append(pd.DataFrame(self.models[ml_name].predict_proba(Xs_exp_selected)[:,1], columns=[ml_name]))
            score = pd.concat(score, axis=1)
            score = pd.concat([score, pd.DataFrame(score.mean(axis=1), columns=['all'])], axis=1)
        else :
            score = pd.DataFrame(self.models[how].predict_proba(Xs_exp_selected)[:,1], columns=['pred_score'])
        
        return score
     
    def predict(self, test_x, threshold = 0.5, how = "best"):
        score = self.predict_proba(test_x, how = how)
        if how == "all":
            preds = (score["all"] >= threshold).astype(int)
            preds.rename("pred_score")
        else:
            preds = (score["pred_score"] >= threshold).astype(int)
        return preds
     
    
        
if __name__ == '__main__':
    # SET PARAMETERS
    file_model = 'dm_for_model'
    file_score = 'dm_for_fwd'
    categorical_features = ['mode_category']   
    
    df = pd.read_csv('./data/'+ file_model + '.csv', header=0)
    ID = df.iloc[:,0] 
    y = df.iloc[:,-1]
    X = df.iloc[:,1:-1]
    
    scoring = 'roc_auc'
    
    model = BinaryClassifier()
    model.fit(X, y, categorical_features = categorical_features, scoring = scoring)
    


    dfs = pd.read_csv('./data/'+ file_score + '.csv', header=0)
    IDs = dfs.iloc[:,[0]] 
    Xs = dfs.iloc[:,1:-1]
    how = "all"
    score = model.predict_proba(Xs, how = how)
    pred = model.predict(Xs, how = how)
    
    IDs.join(score).to_csv('./data/'+  how + '_' + scoring + '_with_predproba.csv', index=False)


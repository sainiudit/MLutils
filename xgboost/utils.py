import operator
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
import pandas as pd
import numpy as np
from sklearn import preprocessing

def ceate_feature_map(features,featuremap_File):
    outfile = open(featuremap_File, 'w')
    i = 0
    for feat in features:
        outfile.write('{0}\t{1}\tq\n'.format(i, feat))
        i = i + 1
    outfile.close()

def plot_feature(bst,featuremap_File):
    importance = bst.get_fscore(fmap=featuremap_File)
    importance = sorted(importance.items(), key=operator.itemgetter(1))
    df = pd.DataFrame(importance, columns=['feature', 'fscore'])
    df['fscore'] = df['fscore'] / df['fscore'].sum()
    plt.figure()
    df.plot()
    df.plot(kind='barh', x='feature', y='fscore', legend=False, figsize=(6, 10))
    plt.title('XGBoost Feature Importance')
    plt.xlabel('relative importance')
    return df.sort_values(by='fscore',ascending=False)

def train_Val(params,train,target_labels,boosting_rounds=40,print_every_n=10):
    X_train, X_test, y_train, y_test = train_test_split(train, target_labels, test_size=0.30, random_state=42)
    dtrain = xgb.DMatrix(X_train, label =y_train)
    dtest = xgb.DMatrix(X_test, label =y_test)
    evallist  = [(dtest,'eval'), (dtrain,'train')]
    #,verbose_eval=print_every_n
    bst =xgb.train(params,dtrain,num_boost_round=boosting_rounds,evals=evallist,early_stopping_rounds=50,feval=rmspe_xg, verbose_eval=False)
    return bst,X_test,y_test

def rmspe(y, yhat):
    return np.sqrt(np.mean((yhat/y-1) ** 2))

def rmspe_xg(yhat, y):
    y = np.expm1(y.get_label())
    yhat = np.expm1(yhat)
    return "rmspe", rmspe(y,yhat)

def encodeLabels(df,attributes):
    for col in attributes:
        le = preprocessing.LabelEncoder()
        le.fit(df[col])
        df[col]=le.transform(df[col])
    return df
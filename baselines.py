import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from flair.embeddings import TransformerDocumentEmbeddings
from flair.data import Sentence
import xgboost as xgb
from sklearn.model_selection import cross_validate
from sklearn.model_selection import GridSearchCV
import nltk
from utils import numerical_df
from sklearn.metrics import f1_score as f1, precision_score as ps, recall_score as rs
from debater_python_api.api.debater_api import DebaterApi
from sklearn.model_selection import KFold

import json
import io
from sklearn.dummy import DummyClassifier

possible_tasks = ['argumentative', 'claim', 'evidence','procon'] 


def dummy_class_baseline(df, tasks = possible_tasks, fold = 10, strategy="stratified"):
    
    if isinstance(tasks, str):
        tasks = [tasks]
    
    if not isinstance(tasks, list):
        return ValueError('Tasks is not a string or a list.')
    
    if not all(item in possible_tasks for item in tasks):
        return ValueError(f'Tasks can only be or contain the following elements {possible_tasks} but found {tasks}')
    
    res = []
    
    for task in tasks:
        kf = KFold(n_splits=fold)
        data = np.array(df[task].values).astype(int) 
        if task == 'procon':
            data = ((data[data != 0] + 1)/2).astype(int)
        tres = [] 
        
        for train_index, test_index in kf.split(data):
            
            dummy_clf = DummyClassifier(strategy=strategy)
            X_train, X_test = data[train_index], data[test_index]  
            dummy_clf.fit(X_train, X_train)

            labels = dummy_clf.predict(X_test)
        
            tres.append((f1(X_test, labels, average='micro'), ps(X_test, labels, average='micro'), rs(X_test, labels, average='micro')))
        res.append(np.concatenate([[task], np.mean(tres, axis=0)]))
    res = pd.DataFrame(res)
    res.columns = ['Task', 'F1', 'Precision', 'Recall']
    return res

def random_class_baseline(df, tasks = possible_tasks, fold = 10):
    res = dummy_class_baseline(df, tasks, fold, strategy = 'stratified')
    res = res.style.set_caption(f'Random class results with {fold} fold split using weighted approach')
    return res

def majority_class_baseline(df, tasks = possible_tasks, fold = 10):
    
    res = dummy_class_baseline(df, tasks, fold, strategy = 'prior')
    res = res.style.set_caption(f'Majority class results with {fold} fold split')
    return res

param_grid = { # Could be made an argument
    #'gamma': [0,0.1,0.2,0.4,0.8,1.0],
    #'learning_rate': [0.01, 0.03, 0.06],
    'max_depth': [1,3,5,6,7,8,9,10],
    'n_estimators': [1,2,5,7,10,15,20,25,30,40,60,80,100],
    #'reg_alpha': [0,0.1,0.2,0.4,0.8,1.0],
    #'reg_lambda': [0,0.1,0.2,0.4,0.8,1.0],
    'objective': ['binary:logistic'],
    'eval_metric':['auc'], 
    'tree_method':["gpu_hist"]
}

def xgboost_baseline(df, model_name='bert-base-cased', tasks = possible_tasks, use_topic = True):
    
    tweet_embeddings = []
    
    # Make sure task is correctly formatted
    if not isinstance(tasks, str) and not isinstance(tasks, list):
        raise ValueError("task must be list or str")
    
    if type(tasks) == str:
        tasks = [tasks]

    if not all(elem in possible_tasks for elem in tasks):
        raise ValueError("task must only contain any of the following strings: ", possible_tasks, ', but found:', tasks)
        
        
        
    print('Loading language model')
    embedding = TransformerDocumentEmbeddings(model_name)
    # Get embeddings
    print('Generating the embeddings')
    for topic, tweet in zip(df.topic, df.tweet):
        sent = Sentence(topic, tweet) if use_topic else Sentence(tweet)
        tweet_embeddings.append(embedding.embed(sent)[0].get_embedding().cpu().detach().numpy())
    
    tweets_data = np.array(tweet_embeddings)
    
    res = [] # Columns
    
    for task in tasks:
        print('Generating results for', task)
        label = df[task].to_numpy()
        data = tweets_data.copy()
        if task == 'procon':
            mask = label != 0
            data = data[mask]
            label = label[mask]
        
    
        clf0 = GridSearchCV(
            estimator=xgb.XGBRFClassifier(n_estimators=1, max_depth=1, objective='binary:logistic', eval_metric='auc', tree_method="gpu_hist"), 
            scoring='f1', 
            param_grid=param_grid, 
            n_jobs=10, 
            verbose=1, 
            cv=3
        )
        
        clf0.fit(data, label)
        params = pd.DataFrame(clf0.cv_results_)
        best = params[params.rank_test_score == 1].iloc[0]
        model = xgb.XGBRFClassifier(n_estimators=best['param_n_estimators'], max_depth=best['param_max_depth'], objective='binary:logistic', eval_metric='auc', tree_method="gpu_hist")
        
        cv_results = cross_validate(model, data, label, scoring=('f1', 'precision', 'recall'), cv=10)
        
        res.append((task, cv_results['test_f1'].mean(), cv_results['test_precision'].mean(),cv_results['test_recall'].mean()))
    
    res = pd.DataFrame(res)
    res.columns = ('Tasks','F1', 'Precision', 'Recall')
    res = res.style.set_caption('Results from fitting XGBoost model on ' + model_name + ' embeddings')
    return res
        
    
def ibm_baseline(df, tasks = possible_tasks):
    if not isinstance(tasks, str) and not isinstance(tasks, list):
        raise ValueError("task must be list or str")
    
    if type(tasks) == str:
        tasks = [tasks]

    if not all(elem in possible_tasks for elem in tasks):
        raise ValueError("task must only contain any of the following strings: ", possible_tasks, ', but found:', tasks)
    

    credentials_path = './credentials.json'

    with io.open(credentials_path) as f_in:
        credentials = json.load(f_in)
    
    api_key = credentials['debater_api_key']
    debater_api = DebaterApi(apikey=api_key)
    clients = {
        "claim": debater_api.get_claim_detection_client(),
        "evidence": debater_api.get_evidence_detection_client(),
        "procon": debater_api.get_pro_con_client(),
        "argumentative": debater_api.get_argument_quality_client(),
    }

    sentence_topic_dicts = [{'sentence' : row.tweet, 'topic' : row.topic } for row in df.iloc]

    res = []
    
    for task in tasks:
        print('Gathering results for', task)
        data = sentence_topic_dicts
        label = df[task].to_numpy()
        
        client = clients[task]
        
        if task == 'procon':
            mask = label != 0
            data = [d for d, m in zip(data, mask) if m]
            label = label[mask]
            scores = [1 if s > 0 else -1 for s in client.run(data)]
        else:
            scores = np.round(client.run(data))
        
        res.append((task, f1(label, scores), ps(label, scores), rs(label, scores)))
        
    res = pd.DataFrame(res)
    res.columns = ('Tasks','F1', 'Precision', 'Recall')
    res = res.style.set_caption('Results from using imbs project debater api for 0 shot evalutaion')
    return res
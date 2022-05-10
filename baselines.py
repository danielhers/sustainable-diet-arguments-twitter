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
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import TrainingArguments, Trainer
from datasets import Dataset, load_metric
from transformers import pipeline

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util

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



def bm25_baseline(df, tasks = possible_tasks, bm25_cutoff = 0.1):

    if isinstance(tasks, str):
        tasks = [tasks]
    
    if not isinstance(tasks, list):
        return ValueError('Tasks is not a string or a list.')
    
    if not all(item in possible_tasks for item in tasks):
        return ValueError(f'Tasks can only be or contain the following elements {possible_tasks} but found {tasks}')
    
    res = []
    
    
    
    
    
    topics = df.topic.drop_duplicates().tolist()
    for topic in topics:
        corpus = df[df.topic == topic].tweet.tolist()
        tokenized_corpus = [doc.split(" ") for doc in corpus]
        bm25 = BM25Okapi(tokenized_corpus)
        tokenized_query = topic.split(" ")
        df.loc[df.topic == topic, 'score'] = bm25.get_scores(tokenized_query) >= bm25_cutoff

    for task in tasks:

            
        labels = df[task]
        preds = np.round(df['score'].astype(int))
        res.append((task, f1(preds, labels, average='micro'), ps(preds, labels, average='micro'), rs(preds, labels, average='micro')))
    res = pd.DataFrame(res)
    res.columns = ['Task', 'F1', 'Precision', 'Recall']
    return res


def qa_model_score(df, topics, model = 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1', dot_score = True):    
    df = df.copy()
    df['score'] = [0]*len(df) 
    
    #Load the model
    print('Loading model')
    model = SentenceTransformer(model)

    docs = df.tweet.values
    doc_emb = model.encode(docs)

    score_dfs = [] 
    for topic in topics:
        
        print('Scoring tweet topic pair', f'"{topic}"')
        df['score'] = [0]*len(df) 
        df['topic'] = [topic]*len(df)
    
        #Encode query and documents
        query_emb = model.encode(topic)
        
        if dot_score:
            #Compute dot score between query and all document embeddings
            scores = util.dot_score(query_emb, doc_emb)[0].cpu().tolist()
        else:
            scores = cosine_similarity(query_emb, doc_emb)[0].tolist()
            
        #Combine docs & scores
        #doc_score_pairs = list(zip(docs, scores))
        df['score'] = scores
        score_dfs.append(df.copy())

        #Sort by decreasing score
        #doc_score_pairs = sorted(doc_score_pairs, key=lambda x: x[1], reverse=True)
    return score_dfs
        


def qa_model_baseline(df, tasks = possible_tasks, model = 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1', model_cutoff = 0.5):

    if isinstance(tasks, str):
        tasks = [tasks]
    
    if not isinstance(tasks, list):
        return ValueError('Tasks is not a string or a list.')
    
    if not all(item in possible_tasks for item in tasks):
        return ValueError(f'Tasks can only be or contain the following elements {possible_tasks} but found {tasks}')
    
    res = []
    
    
    print('Calculating scores')
    topics = df.topic.drop_duplicates().tolist()
    for topic in topics:
        score_dfs = qa_model_score(df[df.topic == topic], [topic], model)[0]
        df.loc[df.topic == topic, 'score'] = score_dfs.score >= model_cutoff
    
    for task in tasks:
        labels = df[task]
        preds = np.round(df['score'].astype(int))
        res.append((task, f1(preds, labels, average='micro'), ps(preds, labels, average='micro'), rs(preds, labels, average='micro')))
    res = pd.DataFrame(res)
    res.columns = ['Task', 'F1', 'Precision', 'Recall']
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
        sent = Sentence(f'{topic}[SEP]{tweet}') if use_topic else Sentence(tweet)
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




def bert_baseline(df, fold = 3, model_name='bert-base-cased', tasks = possible_tasks, use_topic = True, seed = 42, batch_size = 5, epochs = 1):
    
    
    # Make sure task is correctly formatted
    if not isinstance(tasks, str) and not isinstance(tasks, list):
        raise ValueError("task must be list or str")
    
    if type(tasks) == str:
        tasks = [tasks]

    if not all(elem in possible_tasks for elem in tasks):
        raise ValueError("task must only contain any of the following strings: ", possible_tasks, ', but found:', tasks)
        
    metric = load_metric('accuracy')
    
    print('Loading tokenizer')
    tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    def tokenize_function(examples):
        if use_topic: 
            tweet = f"{examples['topic']} [SEP] {examples['tweet']}"
        else:
            tweet = examples['tweet']
        
        return tokenizer(tweet, padding="max_length", truncation=True, add_special_tokens=True)

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return metric.compute(predictions=predictions, references=labels)
    
        
    
            
    
    res = [] # Columns
    
    for task in tasks:
        kf = KFold(n_splits=fold)
        tres = []
        fold_num = 0
        for train_index, test_index in kf.split(range(len(df))):
            print('Splitting out data, for fold', fold_num + 1)
            fold_num += 1
            df_train = df.loc[train_index]
            df_test = df.loc[test_index]
            
            print('Loading language model')
            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

            print('Setting up dataset for', task)
            df_train_renamed = df_train.rename(columns={task: "labels"})
            df_train_renamed['labels'] = df_train_renamed['labels'].astype(int)  
            df_test_renamed = df_test.rename(columns={task: "labels"})
            df_test_renamed['labels'] = df_test_renamed['labels'].astype(int)

            train_data = Dataset.from_pandas(df_train_renamed)
            test_data  = Dataset.from_pandas(df_test_renamed)


            train_processed = train_data.map(tokenize_function).map(lambda x: x, batched=True, batch_size = batch_size).remove_columns(['__index_level_0__'])
            test_processed = test_data.map(tokenize_function).map(lambda x: x, batched=True, batch_size = batch_size).remove_columns(['__index_level_0__'])

            
            train_full = train_processed.shuffle(seed=seed).remove_columns( df_train_renamed.loc[:, df_train_renamed.columns != 'labels'].columns)
            test_full  = test_processed.shuffle(seed=seed).remove_columns( df_test_renamed.loc[:, df_test_renamed.columns != 'labels'].columns)

            print('Setting up training')

            train_args = TrainingArguments(
                output_dir = model_name + '_' + task,
                per_device_train_batch_size = batch_size,
                num_train_epochs = epochs,
                evaluation_strategy='epoch'
            )

            trainer = Trainer(
                model = model,
                args = train_args,
                train_dataset = train_full,
                eval_dataset = test_full,
                compute_metrics = compute_metrics,
            )

            print('Training')
            trainer.train()

            print('Inferring')
            predictions = trainer.predict(test_full)
            preds = predictions.predictions.argmax(-1)
            del model
            del train_full
            del test_full
            torch.cuda.empty_cache()
            
            label = df_test[task].to_numpy()
            tres.append((f1(label, preds, average='micro'), ps(label, preds, average='micro'), rs(label, preds, average='micro')))
        res.append(np.concatenate([[task], np.mean(tres, axis=0)]))
        
    
    res = pd.DataFrame(res)
    res.columns = ('Tasks','F1', 'Precision', 'Recall')
    res = res.style.set_caption('Results finetuning bert-model ' + model_name + ' on ' + task + ' dataset and applying to test set')
    return res
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
import transformers
from transformers import TrainingArguments, Trainer
from datasets import Dataset, load_metric
from transformers import pipeline

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, util

import json
import io
from sklearn.dummy import DummyClassifier

possible_tasks = ['argumentative', 'claim', 'evidence','procon'] 


def calc_scores(preds_set, labels_set, task = None):
    
    average_sets = []
    
    for average in ['binary', 'macro', 'micro']:
        averages = []
        for preds, labels in zip(preds_set, labels_set):
            averages.append((f1(preds, labels, average=average, zero_division = 0), 
                             ps(preds, labels, average=average, zero_division = 0), 
                             rs(preds, labels, average=average, zero_division = 0)))

        average_sets.append((average, *np.round(np.mean(averages, axis=0), 2)))
    
    if task == None:
        return np.array(average_sets)
    else:
        return np.hstack([[[task]]*3, np.array(average_sets)])


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
        df_use = df.copy()
        data = np.array(df[task].values)#.round().astype(int)
        if task == 'procon':
            mask = data != 0
            data = [d for d, m in zip(data, mask) if m]
        elif task != 'argumentative':
            mask = df_use.argumentative > .5
            data = [d for d, m in zip(data, mask) if m]
        else:
            df_use = df_use.drop_duplicates(subset=['tweet'])
            data = df_use.argumentative.to_numpy()
        
        data = np.array(data).round().astype(int)
        tres = [] 
        tlabels = []
        for train_index, test_index in kf.split(data):
            
            dummy_clf = DummyClassifier(strategy=strategy)
            X_train, X_test = data[train_index], data[test_index]  
            dummy_clf.fit(X_train, X_train)

            labels = dummy_clf.predict(X_test)
            
            tres.append(labels)
            tlabels.append(X_test)
            
        res.append(calc_scores(tres, tlabels, task))
    res = pd.DataFrame(np.concatenate(res))
    res.columns = ['Task', 'Averaging', 'F1', 'Precision', 'Recall']
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
        res.append(calc_scores([preds], [labels], task))
    res = pd.DataFrame(np.concatenate(res))
    res.columns =  ['Task', 'Averaging', 'F1', 'Precision', 'Recall']
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
        res.append(calc_scores([preds], [labels], task))
    res = pd.DataFrame(np.concatenate(res))
    res.columns =  ['Task', 'Averaging', 'F1', 'Precision', 'Recall']
    return res

param_grid = { # Could be made an argument
    #'gamma': [0,0.1,0.2,0.4,0.8,1.0],
    'learning_rate': [0.01, 0.03, 0.06],
    'max_depth': [1,3,5,6,7,8,9,10],
    'n_estimators': [1,2,5,7,10,15,20,25,30,40,60,80,100],
    #'reg_alpha': [0,0.1,0.2,0.4,0.8,1.0],
    #'reg_lambda': [0,0.1,0.2,0.4,0.8,1.0],
    'objective': ['binary:logistic'],
    #'eval_metric':['auc'], 
    'tree_method':["gpu_hist"]
}

def xgboost_baseline(df, model_name='bert-base-cased', fold = 3, tasks = possible_tasks, use_topic = True, average = ''): # empty average is binary
    
    tweet_embeddings = []
    tweet_embeddings_argumentative = []
    
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
        #tweet = f"{topic}[SEP]{tweet}"
        tweet_emb = embedding.embed(Sentence(tweet))[0].get_embedding().cpu().detach().numpy()
        topic_emb = embedding.embed(Sentence(topic))[0].get_embedding().cpu().detach().numpy()
        #con = np.concatenate((tweet_emb,topic_emb))
        tweet_embeddings.append(tweet_emb + topic_emb)

    if 'argumentative' in tasks:
        for tweet in df.tweet.drop_duplicates():
            sent = Sentence(tweet)
            tweet_embeddings_argumentative.append(embedding.embed(sent)[0].get_embedding().cpu().detach().numpy())
    
    tweets_data = np.array(tweet_embeddings)
    
    res = [] # Columns
    
    for task in tasks:
        print('Generating results for', task)
        df_use = df.copy()
        data = tweets_data.copy()
        label = df_use[task].to_numpy()
        
        if task == 'procon':
            mask = label != 0
            data = [d for d, m in zip(data, mask) if m]
            label = label[mask]
        elif task != 'argumentative':
            mask = df_use.argumentative > .5
            data = [d for d, m in zip(data, mask) if m]
            label = label[mask]
        else:
            df_use = df_use.drop_duplicates(subset=['tweet'])
            label = df_use.argumentative.to_numpy()
            data = np.array(tweet_embeddings_argumentative)
        
        scorings = ('f1', 'precision', 'recall',
                    'f1_macro', 'precision_macro', 'recall_macro',
                    'f1_micro', 'precision_micro', 'recall_micro')
        
        clf0 = GridSearchCV(
            estimator=xgb.XGBRFClassifier(n_estimators=1, max_depth=1, objective='binary:logistic', tree_method="gpu_hist"), 
            scoring='f1_macro', 
            param_grid=param_grid, 
            n_jobs=20, 
            verbose=0, 
            cv=3
        )
        
        clf0.fit(data, label)
        params = pd.DataFrame(clf0.cv_results_)
        best = params[params.rank_test_score == 1].iloc[0]
        print('Best parameters:')
        print(best['params'])
        model = xgb.XGBRFClassifier(n_estimators=best['param_n_estimators'], max_depth=best['param_max_depth'], objective='binary:logistic', eval_metric='auc', tree_method="gpu_hist")
        
        end = f'_{average}' if average != '' else ''
        
        
        
        cv_results = cross_validate(model, data, label, scoring=scorings, cv=fold)
        
        average_sets = []
        for aver in scorings:
            average_sets.append(np.round(cv_results[f'test_{aver}'].mean(), 2))
        
        average_sets = np.hstack([[['binary'], ['macro'], ['micro']], np.array(average_sets).reshape((3,3))])
        res.append(np.hstack([[[task]]*3, average_sets]))
        
    res = pd.DataFrame(np.concatenate(res))
    res.columns =  ['Task', 'Averaging', 'F1', 'Precision', 'Recall']
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
    
    argumentative_mask = df.argumentative == 1

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
            data = [d for d, m in zip(data, argumentative_mask) if m]
            label = label[argumentative_mask]
            scores = np.round(client.run(data))
        
        res.append(calc_scores([scores], [label], task))
    res = pd.DataFrame(np.concatenate(res))
    res.columns =  ['Task', 'Averaging', 'F1', 'Precision', 'Recall']
    res = res.style.set_caption('Results from using imbs project debater api for 0 shot evalutaion')
    return res



from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import EarlyStoppingCallback
# To control logging level for various modules used in the application:
import logging
import re


def set_global_logging_level(level=logging.ERROR, prefices=[""]):
    """
    Override logging levels of different modules based on their name as a prefix.
    It needs to be invoked after the modules have been loaded so that their loggers have been initialized.

    Args:
        - level: desired level. e.g. logging.INFO. Optional. Default is logging.ERROR
        - prefices: list of one or more str prefices to match (e.g. ["transformers", "torch"]). Optional.
          Default is `[""]` to match all active loggers.
          The match is a case-sensitive `module_name.startswith(prefix)`
    """
    prefix_re = re.compile(fr'^(?:{ "|".join(prefices) })')
    for name in logging.root.manager.loggerDict:
        if re.match(prefix_re, name):
            logging.getLogger(name).setLevel(level)

set_global_logging_level(logging.ERROR, ["nlp", "torch", "tensorflow", "tensorboard", "wandb", "matplotlib", 'sklearn'])
logging.disable(logging.WARNING) # disable INFO and DEBUG logging everywhere
   
            
def bert_baseline(df, fold = 3, model_name='bert-base-cased', tasks = possible_tasks, seed = 42, batch_size = 5, epochs = 1, learning_rate = 5e-5, average='binary'):
    
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
    use_topic = False
    def tokenize_function(examples):
        if use_topic: 
            return tokenizer(examples['topic'], examples['tweet'], padding="max_length", max_length=68, truncation=True, add_special_tokens=True)
        else:
            return tokenizer(examples['tweet'], padding="max_length", max_length=68, truncation=True, add_special_tokens=True)
    
    
            
    
    res = [] # Columns
    
    for task in tasks:
        
        def compute_metrics(pred):
            #logits, labels = eval_pred
            labels = pred.label_ids
            preds = pred.predictions.copy()
            if task == 'procon':
                labels[labels > 0] = 1
                labels[labels < 0] = -1
                preds[preds > 0] = 1
                preds[preds < 0] = -1
            else:
                labels = np.round(labels)
                preds = np.round(preds)#.argmax(-1)
                
            
            precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', labels=np.unique(labels))
            acc = accuracy_score(labels, preds)
            return {
                #'accuracy': acc,
                '#######': '###########',
                'f1': f1,
                #'precision': precision,
                #'recall': recall,
                '########': '############',
                'avg_label': np.mean(labels),
                'avg_preds': np.mean(preds),
                'avg_prediction': np.array_str(np.mean(pred.predictions, axis=0)),
            }
            #predictions = np.argmax(logits, axis=-1)
            #return metric.compute(predictions=predictions, references=labels)
        
        kf = KFold(n_splits=fold)
        tres = []
        tlabels = []
        fold_num = 0
        use_topic = task != 'argumentative'
        df_use = df.copy()
        if task == 'procon':
            mask = df_use.procon != 0
            df_use = df_use[mask]
            #df_use.procon = (df_use.procon+1)//2
        elif task != 'argumentative':
            mask = df_use.argumentative > 0.5
            df_use = df_use[mask]
        else:
            df_use = df_use.drop_duplicates(subset=['tweet'])
            
        for train_index, test_index in kf.split(range(len(df_use))):
            print('Splitting out data, for fold', fold_num + 1)
            fold_num += 1
            df_train = df_use.iloc[train_index]
            df_valid, df_test = np.array_split(df_use.iloc[test_index], 2)
            
            print('Loading language model')
            model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1)

            print('Setting up dataset for', task)
            df_train_renamed = df_train.rename(columns={task: "labels"})
            df_train_renamed['labels'] = df_train_renamed['labels']#.astype(int)  
            df_valid_renamed = df_valid.rename(columns={task: "labels"})
            df_valid_renamed['labels'] = df_valid_renamed['labels']#.astype(int)
            df_test_renamed = df_test.rename(columns={task: "labels"})
            df_test_renamed['labels'] = df_test_renamed['labels']#.astype(int)

            train_data = Dataset.from_pandas(df_train_renamed)
            valid_data  = Dataset.from_pandas(df_valid_renamed)
            test_data  = Dataset.from_pandas(df_test_renamed)


            train_processed = train_data.map(tokenize_function).map(lambda x: x, batched=True, batch_size = batch_size).remove_columns(['__index_level_0__'])
            valid_processed = valid_data.map(tokenize_function).map(lambda x: x, batched=True, batch_size = batch_size).remove_columns(['__index_level_0__'])
            test_processed = test_data.map(tokenize_function).map(lambda x: x, batched=True, batch_size = batch_size).remove_columns(['__index_level_0__'])

            
            train_full = train_processed.shuffle(seed=seed).remove_columns( df_train_renamed.loc[:, df_train_renamed.columns != 'labels'].columns)
            valid_full  = valid_processed.shuffle(seed=seed).remove_columns( df_test_renamed.loc[:, df_test_renamed.columns != 'labels'].columns)
            test_full  = test_processed.shuffle(seed=seed).remove_columns( df_test_renamed.loc[:, df_test_renamed.columns != 'labels'].columns)

            print('Setting up training')

            train_args = TrainingArguments(
                output_dir = model_name + '_' + task,
                evaluation_strategy ='steps',
                num_train_epochs = epochs,
                eval_steps = 20, 
                save_total_limit = 10,
                learning_rate=learning_rate,
                per_device_train_batch_size = batch_size,
                per_device_eval_batch_size = batch_size,
                weight_decay=0.05,
                push_to_hub=False,
                metric_for_best_model = 'f1',
                load_best_model_at_end=True,
                optim="adamw_torch"
            )

            trainer = Trainer(
                model = model,
                args = train_args,
                train_dataset = train_full,
                eval_dataset = valid_full,
                compute_metrics = compute_metrics,
                callbacks = [EarlyStoppingCallback(early_stopping_patience=5)],
            )
            

            print('Training')
            trainer.train()

            print('Inferring')
            predictions = trainer.predict(test_full)
            preds = predictions.predictions
            label = df_test[task].to_numpy()
            
            if task == 'procon':
                label[label > 0] = 1
                label[label < 0] = -1
                preds[preds > 0] = 1
                preds[preds < 0] = -1
            else:
                preds = np.round(predictions.predictions)#.argmax(-1)
                label = np.round(label)
            tres.append(preds)
            tlabels.append(label)
            
            del model
            del train_full
            del test_full
            torch.cuda.empty_cache()
            
            
        res.append(calc_scores(tres, tlabels, task))
    res = pd.DataFrame(np.concatenate(res))
    res.columns = ['Task', 'Averaging', 'F1', 'Precision', 'Recall']
    res = res.style.set_caption('Results finetuning bert-model ' + model_name + ' on ' + task + ' dataset and applying to test set')
    return res
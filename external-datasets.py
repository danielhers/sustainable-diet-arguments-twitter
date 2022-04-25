import pandas as pd

def ibm_argumentative():
    df = pd.read_csv('Ibm-datasets/argumentative/set.csv')
    df = df.drop(columns = ['#positive', '#negative', 'sentence_internal_id', 'val', 'test'])
    df.columns = ['id', 'argument', 'tweet', 'topic', 'argumentative']
    return df

def ibm_evidence():
    df = pd.concat([pd.read_csv('Ibm-datasets/evidence/test.csv'), pd.read_csv('Ibm-datasets/evidence/train.csv')])
    df = df.drop(columns = ['the concept of the topic', 'candidate masked', 'wikipedia article name', 'wikipedia url'])
    df.columns = ['topic', 'tweet', 'evidence']
    return df



def ibm_procon():
    df = pd.read_csv('Ibm-datasets/procon/set.csv')
    df = df.drop(columns = ['split', 'topicTarget', 'topicSentiment', 'claims.claimId', 'claims.claimCorrectedText', 'claims.article.rawFile', 'claims.article.cleanFile', 'claims.article.rawSpan.start', 'claims.article.rawSpan.end', 'claims.article.cleanSpan.start', 'claims.article.cleanSpan.end', 'claims.Compatible','claims.claimSentiment', 'claims.targetsRelation', 'claims.claimTarget.span.end', 'claims.claimTarget.span.start', 'claims.claimTarget.text'])
    df.columns = ['id','topic', 'procon', 'tweet']
    df.procon = df.procon.apply(lambda x: 1 if x == 'PRO' else -1)
    return df
import pandas as pd
import numpy as np

def numerical_df(df):
    df = df.copy()
    idx = df[df.argumentative == 'lacks context'].index
    df.argumentative.update(pd.Series(np.ones(len(idx)), index=idx))
    df.argumentative = pd.to_numeric(df.argumentative)
    df.claim = pd.to_numeric(df.claim)
    df.evidence = pd.to_numeric(df.evidence)
    df.procon = pd.to_numeric(df.procon)
    return df
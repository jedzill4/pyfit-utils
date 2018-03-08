
import pandas as pd
import numpy as np


def stratifiedKFold(df, k, weights=None):
    total_data = len(df)
    min_len = total_data//k
    lenfolds = [min_len for i in range(k)]
    unassigned = total_data - min_len*k
    i = 0
    while unassigned > 0:
        lenfolds[i] += 1
        unassigned -= 1
        i = (i+1)%k

    folds = []

    if type(weights) == type(None):
        dfcopy = df.copy().iloc[np.random.permutation(total_data)]
        for i,size in enumerate(lenfolds):
            sample = dfcopy.sample(size)
            dfcopy.drop( sample.index )
            folds.append(sample)

        return folds

    df.loc[:,'weights'] = weights
    dfcopy = df.copy().iloc[np.random.permutation(total_data)]
    dfcopy.reset_index(inplace=True)
    if np.any( weights < 0 ):
        positives = (dfcopy['weights'] >= 0)
        dfcopy['weights'] = np.abs(dfcopy['weights'])
        list_data = { 0:dfcopy[ positives], 
                      1:dfcopy[~positives] }
        data_types = 2
    else:
        list_data = { 0:dfcopy }
        data_types = 1

    count = np.zeros(data_types).astype(int)
    total_data = np.array([ len(dataset) for key,dataset in list_data.items() ])
    folds = [ pd.DataFrame() for i in range(k)]
    while np.sum(count) < np.sum(total_data):
        for id in range(data_types):
            data_left = total_data[id] - count[id]
            if data_left > 0 :
                sample = list_data[id].sample(min(k,data_left), weights='weights')
                for i, data in enumerate(sample.itertuples()):
                    ddf = pd.DataFrame(dict(data._asdict()),index=[data.Index])
                    folds[i] = pd.concat([folds[i],ddf])
                    list_data[id] = list_data[id].drop(data.Index) 
                    count[id] += 1

    asd = np.concatenate(folds)
    return folds




def samplestodf(folds):
    df = pd.DataFrame()
    for data in folds:
        df = pd.concat([df,data])
    return df

def splitTestValidation(df, ngroups, weights=None):
    folds = stratifiedKFold(df,ngroups,weights=weights)

    i = np.random.randint(0,ngroups)
    val = folds[i]
    del folds[i]
    test = pd.DataFrame()
    for df in folds:
        test = pd.concat([test,df])
    return test, val

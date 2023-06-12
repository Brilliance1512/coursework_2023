import random
from typing import List

import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
from scipy import interpolate
import scipy.stats
from sklearn.cluster import DBSCAN
from kneed import KneeLocator
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


def plot_signal(data: pd.DataFrame, signal_num: int = 0, feature: str = 'signal'):
    plt.plot(range(len(data.iloc[signal_num][feature])), data.iloc[signal_num][feature])
    plt.xlabel("Signal")
    plt.ylabel("Magnitude")
    plt.show()


def interpolate_signal(signal: List[int], target_length: int = 100) -> List[int]:
    resampling_factor = (len(signal)-1) / target_length
    x_old = np.arange(len(signal))
    x_new = np.arange(target_length)
    f = interpolate.interp1d(x_old, signal, kind='linear')
    new_signal = f(x_new * resampling_factor)
    return new_signal.astype(int)


def rescale_signal(emg_signal: List[int], desired_min: int = 0, desired_max: int = 1) -> List[float]:
    current_min = 0
    current_max = 150
    scaling_factor = (desired_max - desired_min) / (current_max - current_min)
    rescaled_signal = [(x - current_min) * scaling_factor + desired_min for x in emg_signal]
    return rescaled_signal


def mutate_signal(signal: List[int]) -> List[int]:
    decision =  random.randint(1, 3)
    if(decision == 1):
        to_remove = random.randint(0, 10)
        signal = signal[:(100 - to_remove)]
    if(decision == 2):
        to_remove = random.randint(0, 10)
        signal = signal[to_remove:]
    if(decision == 3):
        to_add = random.randint(-10, 10)
        for i in range(len(signal)):
            if(signal[i]*(1 + to_add/10) <= 1):
                signal[i] *= 1 + to_add/10
            else:
                signal[i] = 1
    signal = interpolate_signal(signal)
    return signal


def tall_to_wide_bootstrap(data: pd.DataFrame, n_samples: int = 20, mutate: bool = True) -> pd.DataFrame:
    muscule_dict = {i:{j:[] for j in data['Muscule'].unique()} for i in data['Movement'].unique()}
    for _, row in data.iterrows():
        muscule_dict[row['Movement']][row['Muscule']].append(row['signal'])

    new_dataset = []
    for move in data['Movement'].unique():
        for _ in range(n_samples):
            temp = [move]
            for musc in data['Muscule'].unique():
                mute = random.choice(muscule_dict[move][musc])
                if mutate:
                    mute = mutate_signal(mute)
                    
                temp.append([np.mean(mute), np.var(mute), scipy.stats.skew(mute), scipy.stats.kurtosis(mute)])
            new_dataset.append(temp)
    cols = [f'musc_{i}' for i in data['Muscule'].unique()]
    cols = ['move'] + cols
    df = pd.DataFrame(new_dataset, columns = cols)
    parts = [pd.DataFrame(df[f'musc_{i}'].to_list(), columns = [f'musc_{i}_{j}' for j in ['mean', 'var', 'skew', 'kurt']]) for i in data['Muscule'].unique()]
    return pd.concat([df['move']] + parts, axis = 1)
    

def get_dbscan_radius(res):
	neighbors = NearestNeighbors(n_neighbors=19)
	neighbors_fit = neighbors.fit(res)
	distances, _ = neighbors_fit.kneighbors(res)
	distances = np.sort(distances, axis=0)
	distances = distances[:,1]
	knee = KneeLocator(distances, list(range(len(distances))), curve='concave', direction='increasing')
	return knee.elbow


def DBSCAN_filtering(df: pd.DataFrame) -> pd.DataFrame:
    pca = PCA(n_components=2)
    res = pca.fit_transform(df['signal'].tolist())

    rad = get_dbscan_radius(res)

    db = DBSCAN(eps=3.8*rad, min_samples=1).fit(res)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    max_cluster = pd.DataFrame(labels)[0].value_counts(ascending=False).index[0]
    df['clusters'] = labels
    df = df[df['clusters'] == max_cluster]
    df = df.drop('clusters', axis = 1)
    return df


def visualize_clustering(df: pd.DataFrame) -> None:
    pca = PCA(n_components=2)
    res = pca.fit_transform(df['signal'].tolist())

    rad = get_dbscan_radius(res)

    db = DBSCAN(eps=1.1*rad, min_samples=1).fit(res)
    outlier_indices = np.where(db.labels_ == -1)[0]

    plt.scatter(res[:, 0], res[:, 1], c=db.labels_, cmap="viridis")
    plt.scatter(res[outlier_indices, 0], res[outlier_indices, 1],)
    plt.xticks([])
    plt.yticks([])
    plt.legend()
    plt.show()

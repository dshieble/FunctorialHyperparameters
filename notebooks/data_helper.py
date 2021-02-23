from gensim.models.word2vec import Word2Vec
import gensim.downloader as api
from tqdm import tqdm
import random
import numpy as np
import pandas as pd
import seaborn as sns
import time
from sklearn.cluster import KMeans
from collections import defaultdict
from collections import Counter
import numpy as np
from sklearn.metrics import adjusted_rand_score
from sklearn.linear_model import LogisticRegression
from scipy.optimize import linprog
from copy import deepcopy
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from scipy.sparse import vstack 
from sklearn.decomposition import TruncatedSVD



sns.reset_defaults()
sns.set_context(context='talk',font_scale=0.7)
from sklearn.datasets import fetch_20newsgroups


import sys
from umap import UMAP 

def load_mnist(path, kind='train'):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels


def load_umap_mnist(sampling, n_components=10):
    sampling = 20
    path = '/Users/dshiebler/workspace/personal/Category_Theory/FunctorialManifoldLearning/data/fashion'

    rawX, rawy = load_mnist(path, kind='train')
    untransformedX, y = rawX[::sampling], rawy[::sampling]
    X = untransformedX if n_components is None else UMAP(n_components=n_components).fit_transform(untransformedX)
    return X, y


def load_umap_newsgroups(sampling, n_components=10):
    train_bunch = fetch_20newsgroups(subset="train")
    rawX, y = train_bunch['data'], train_bunch['target']
    steps = [
        ('vect', CountVectorizer()),
        ('tfidf', TfidfTransformer()),
    ]
    if n_components is not None:
        steps.append(('umap', UMAP(n_components=10)))
    X = Pipeline(steps).fit_transform(rawX)
    return X[::sampling], y[::sampling]
    

import matplotlib.pyplot as plt
from helpers import is_numeric
import numpy as np
from collections import Counter

def plot_reduced_embeddings(entity_ids, reduced_embeddings, make_figure=True):
    colors = []
    for e in entity_ids:
        if is_numeric(e):
            colors.append("r" if len(e) == 4 else "g")
        else:
            colors.append("b")
    print(Counter(colors))
    if make_figure:
        plt.figure(figsize=(10, 10))
    plt.scatter(
        x=reduced_embeddings[:, 0],
        y=reduced_embeddings[:, 1],
        c=colors,
        alpha=0.5
    )
    mins = np.min(reduced_embeddings, axis=0)
    maxs = np.max(reduced_embeddings, axis=0)
    plt.xlim((mins[0] - abs(mins[0]), maxs[0] + abs(maxs[0])))
    plt.ylim((mins[1] - abs(mins[1]), maxs[1] + abs(maxs[1])))


def get_data(num_embeddings=20000):
    num_embeddings = 20000
    entity_ids = np.load("/Users/dshiebler/workspace/data/glove-wiki-gigaword-50/vocab_words.npy")[:num_embeddings]
    raw_embeddings = np.load("/Users/dshiebler/workspace/data/glove-wiki-gigaword-50/embeddings.npy")[:num_embeddings]
    metric = "euclidean"
    return entity_ids, raw_embeddings, metric

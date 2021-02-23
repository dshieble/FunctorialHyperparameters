from mip import Model, xsum, minimize, BINARY
from tqdm import tqdm
import annoy
import random
import numpy as np
import pandas as pd
import seaborn as sns
import time
from collections import defaultdict
from collections import Counter
import cvxpy
import numpy as np
from scipy.optimize import linprog
from copy import deepcopy
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from hdbscan._hdbscan_tree import (condense_tree,
                            compute_stability,
                            get_clusters,
                            outlier_scores,
                                   do_labelling,
                                   get_probabilities,
                                   get_stability_scores)
from hdbscan.hdbscan_ import _hdbscan_boruvka_kdtree, _hdbscan_prims_kdtree, _hdbscan_generic
from collections import Counter
from hdbscan import HDBSCAN
from mip_helper import get_mip_solution_from_cluster_point_matrix


def build_cluster_point_matrix_from_tree(top_sort_ctree):
    """
    Given a topologically sorted condensed tree, builds a num_clusters x num_points binary matrix
    """
    if len(set([c[0] for c in top_sort_ctree])) <= 1:
        raise ValueError("Error: Only one parent found. No clusters are present in tree!")    
    
    num_points = min([c[0] for c in top_sort_ctree])
    num_clusters = max([c[1] for c in top_sort_ctree]) + 1 - num_points
    A = np.zeros((num_clusters, num_points), dtype=np.bool)

    # We reverse the order of the condensed tree in order to do this sum in topologically sorted order
    for c in reversed(top_sort_ctree):
        parent_cluster_index = c["parent"] - num_points
        if c["child"] < num_points:
            # node->cluster connections
            A[parent_cluster_index, c["child"]] = True
        else:
            # cluster->cluster connections
            child_cluster_index = c["child"] - num_points
            A[parent_cluster_index, :] += A[child_cluster_index, :]
    return A


def combine_trees_into_matrix(condensed_trees):
    # Topologically sort the trees
    sorted_condensed_trees = [sorted(ctree, key=lambda node: node["child"]) for ctree in condensed_trees]

    # Create and stack matrices
    cluster_point_matrix_list = []
    valid_indices = []
    for i, ctree in enumerate(sorted_condensed_trees):
        try:
            cluster_point_matrix = build_cluster_point_matrix_from_tree(ctree)
            cluster_point_matrix_list.append(cluster_point_matrix)
            valid_indices.append(i)
        except ValueError as e:
            print(e)
            continue

    if len(cluster_point_matrix_list) == 0:
        raise ValueError("All trees in the condensed_trees list have no clusters")
    return np.vstack(cluster_point_matrix_list), valid_indices


def get_stability_vector(stabilities_list, normalize=False):
    """
    Build the vector of cluster stabilities from the dictionary of cluster index to stability value
    
    If normalize, then renormalize each clustering's stability score
    """
    multiplier = 100 # use this to make the mip avoid numerical difficulities

    num_points = min(stabilities_list[0].keys())
    all_stabilities_vector = np.zeros(sum([len(s) for s in stabilities_list]))
    max_index = 0
    for stability_dict in stabilities_list:
        stability_sum = np.sum(list(stability_dict.values()))
        for cluster, stability in stability_dict.items():
            ix = int(cluster - num_points)
            assert all_stabilities_vector[max_index + ix] == 0
            all_stabilities_vector[max_index + ix] = multiplier*(stability / stability_sum) if normalize else stability
        max_index += len(stability_dict)
    return all_stabilities_vector



def combine_clusters(cluster_point_matrix, stability_vector, source_indices, m):
    """
    Find all pairs of clusters that have full overlap. Remove the second cluster and add its stability to the first
    
    NOTE: For m > 0, we lose the transitivity property and can be adding together clusters that are quite different
    """
    cluster_point_matrix = np.float32(cluster_point_matrix)
    overlap_matrix = np.matmul(cluster_point_matrix, cluster_point_matrix.T)
    cluster_sizes = np.diag(overlap_matrix)
    indexer = np.ones(cluster_point_matrix.shape[0], dtype=np.bool)

    x_pair_vector, y_pair_vector = np.nonzero((cluster_sizes - overlap_matrix) <= m)


    l_overlap_adjacency = np.zeros(overlap_matrix.shape)
    l_overlap_adjacency[x_pair_vector, y_pair_vector] = 1
    r_overlap_adjacency = np.zeros(overlap_matrix.shape)
    r_overlap_adjacency[y_pair_vector, x_pair_vector] = 1
    overlap_adjacency = csr_matrix(r_overlap_adjacency*l_overlap_adjacency)
    _, components = connected_components(csgraph=overlap_adjacency, directed=False, return_labels=True)
    component_to_indices = defaultdict(list)
    for i, c in enumerate(components):
        component_to_indices[c].append(i)

    seen = set()
    for component, index_list in component_to_indices.items():
        # add all of stabilities/points of the second+ indices in the component to the first index
        l1 = index_list[0]
        assert l1 not in seen
        seen.add(l1)
        for l2 in index_list[1:]:
            assert l2 not in seen
            seen.add(l2)

            source_indices[l1] = source_indices[l1].union(source_indices[l2])
            stability_vector[l1] += stability_vector[l2]
            cluster_point_matrix[l1, :] = np.int32(np.logical_or(
                cluster_point_matrix[l1, :], cluster_point_matrix[l2, :]
            ))
            indexer[l2] = False

    # Also return for each row the index of the parameter values at which the stability was computed
    return (
        cluster_point_matrix[indexer, :],
         stability_vector[indexer],
         [source_indices[i] for i, include in enumerate(indexer) if include])


def get_labels_from_cluster_point_matrix(cluster_point_matrix, stability_vector, source_indices, solver=False):
    # One row for each pair of clusters that overlap, with a 1 in the position of each cluster
    solution = get_mip_solution_from_cluster_point_matrix(cluster_point_matrix, stability_vector, solver)

    # Assert no overlaps
    assert np.max(np.sum(cluster_point_matrix[solution, :], axis=0, dtype=np.float32)) <= 1
    
    solved_matrix = cluster_point_matrix[solution, :]
    cluster_to_source = [source_indices[i] for i, include in enumerate(solution) if include]
    cluster_to_stability = stability_vector[solution]

    solved_matrix_with_extra_row = np.zeros((solved_matrix.shape[0] + 1, solved_matrix.shape[1]))
    solved_matrix_with_extra_row[1:, :] = solved_matrix

    labels = (np.argmax(solved_matrix_with_extra_row, axis=0) - 1)
    return labels, cluster_to_stability, cluster_to_source


def get_labels_from_condensed_trees(condensed_tree_list, stabilities_list, m, solver=False, normalize_stabilities=False):
    raw_cluster_point_matrix, valid_indices = combine_trees_into_matrix(condensed_tree_list)
    stabilities_list = [stabilities_list[i] for i in valid_indices]
    
    raw_stability_vector = get_stability_vector(stabilities_list, normalize=True)
    raw_source_indices = [{ix} for i, s in enumerate(stabilities_list) for ix in [i]*len(s)]

    if m is not None:
        cluster_point_matrix, stability_vector, source_indices = combine_clusters(
            raw_cluster_point_matrix, raw_stability_vector, raw_source_indices, m)
    else:
        cluster_point_matrix, stability_vector, source_indices = (
            raw_cluster_point_matrix, raw_stability_vector, raw_source_indices)

    return get_labels_from_cluster_point_matrix(
        cluster_point_matrix, stability_vector, source_indices, solver=solver)


def run_multiscale_hdbscan(X, alpha_list, min_samples=5, min_cluster_size=10):
    raw_tree_list = []
    condensed_tree_list = []
    stabilities_list = []
    labels_list = []
    for alpha in tqdm(alpha_list):
        (single_linkage_tree, result_min_span_tree) = _hdbscan_prims_kdtree(X,
                                                                       min_samples=min_samples,
                                                                       alpha=alpha,
                                                                       metric='euclidean',
                                                                       p=None,
                                                                       leaf_size=3,
                                                                       gen_min_span_tree=False)
        # parent, child, lambda_val (value at which point departs cluster), child_size
        condensed_tree = condense_tree(single_linkage_tree, min_cluster_size=min_cluster_size)
        stability_dict = compute_stability(condensed_tree)

        raw_tree_list.append(single_linkage_tree)
        condensed_tree_list.append(condensed_tree)
        stabilities_list.append(stability_dict)
    return raw_tree_list, condensed_tree_list, stabilities_list


class MultiscaleHDBSCAN(object):


    def __init__(self, alpha_list, solver="mip", min_samples=5, m=0, min_cluster_size=10, normalize_stabilities=False):
        self.alpha_list = alpha_list
        self.solver = solver
        self.min_samples = min_samples
        self.m = m
        self.min_cluster_size = min_cluster_size
        self.normalize_stabilities = normalize_stabilities

    def fit_predict(self, X):
        _, condensed_tree_list, stabilities_list = run_multiscale_hdbscan(
            X=X,
            alpha_list=self.alpha_list,
            min_samples=self.min_samples,
            min_cluster_size=self.min_cluster_size)
        
        multiscale_labels, multiscale_cluster_to_stability, multiscale_cluster_to_source = get_labels_from_condensed_trees(
            condensed_tree_list=condensed_tree_list,
            stabilities_list=stabilities_list,
            m=self.m,
            solver=self.solver,
            normalize_stabilities=self.normalize_stabilities)
        
        return multiscale_labels

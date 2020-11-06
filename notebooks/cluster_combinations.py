"""
Sparse linear programming solution
https://github.com/martinResearch/PySparseLP

(start out with just optimization, nothing else)
matrix: points x clusters
matrix: clusters x clusters (overlap)

simplify the linear program to have the cluster matrix be size num_clusters x num_clusters?
use boolean matrix multiplication to prevent overlaps?

NOTE: Stability does not get added

"""
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

from hdbscan._hdbscan_tree import (condense_tree,
                            compute_stability,
                            get_clusters,
                            outlier_scores,
                                   do_labelling,
                                   get_probabilities,
                                   get_stability_scores)
from hdbscan.hdbscan_ import _hdbscan_boruvka_kdtree, _hdbscan_prims_kdtree, _hdbscan_generic


import sys
sys.path.append("/Users/dshiebler/workspace/personal/Category_Theory/unsupervised/hdbscan")

from hdbscan import HDBSCAN



def build_cluster_point_matrix_from_tree(top_sort_ctree):
    """
    Given a topologically sorted condensed tree, builds a num_clusters x num_points binary matrix
    """
    num_points = min([c[0] for c in top_sort_ctree])
    num_clusters = max([c[1] for c in top_sort_ctree]) + 1 - num_points
    A = np.zeros((num_clusters, num_points), dtype=np.bool)
    
    # NOTE: We need to do this sum in topologically sorted order. this is possible if we first
    for c in top_sort_ctree:
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
    A = np.vstack([build_cluster_point_matrix_from_tree(ctree) for ctree in sorted_condensed_trees])

    return A

def get_stability_vector(stabilities_list, normalize=False):
    """
    Build the vector of cluster stabilities from the dictionary of cluster index to stability value
    
    If normalize, then renormalize each clustering's stability score
    """
    multiplier = 100 # use this to make the linear program avoid numerical difficulities

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


def remove_overlapping_in_order(A, out):
    """
    TODO: Remove in order of number of points
    """
    out = deepcopy(out)
    while True:
        overlap_counts = np.matmul(A, out)
        is_overlapping = overlap_counts > A[0,0]
        if sum(is_overlapping) == 0:
            break
        selected_indices = sorted(np.arange(len(overlap_counts)), key=lambda ix: -overlap_counts[ix])
        selected_index = next(ix for ix in selected_indices if is_overlapping[ix])
        assert out[selected_index] == 1
        out[selected_index] = 0 # zero out the lowest stability overlapping index
        
#         print("np.sum(is_overlapping)", np.sum(is_overlapping))
#         print("np.matmul(row_selection_weights, out)", np.matmul(row_selection_weights, out))
        
    return out


def solve_zero_one_linear_program(c, A, b, use_exact_solution):
    """Minimize c*x
        x is binary
        A*c <= b
    """
    assert A.shape[0] == c.shape[0]
    assert A.shape[0] == b.shape[0]

    out = None
    if use_exact_solution:
        start = time.time()
        print("Solving integer program of shape {}...".format(A.shape))
#         print(A)
    #     assert False
        # The variable we are solving for
        selection = cvxpy.Variable(c.shape[0], boolean=True)
        weight_constraint = A*selection <= b

        # We tell cvxpy that we want to maximize total utility 
        # subject to weight_constraint. All constraints in 
        # cvxpy must be passed as a list
        problem = cvxpy.Problem(cvxpy.Minimize(c * selection), [weight_constraint])

        # Solving the problem
        problem.solve(solver=cvxpy.GLPK_MI, verbose=True)
        print("Integer program solved in {}!".format(time.time() - start))

        return np.array(list(problem.solution.primal_vars.values())[0], dtype=bool)
    else:
        print("using approximate solution")
        solution = linprog(c=c, A_ub=A, b_ub=b)
        out = remove_overlapping_in_order(A=A, out=np.round(solution.x) > 0)
    assert out is not None
    return out


def combine_clusters(cluster_point_matrix, stability_vector, source_indices):
    """
    Find all pairs of clusters that have full overlap. Remove the second cluster and add its stability to the first
    
    NOTE:
        For m > 0, we lose the transitivity property and can be adding together clusters that are quite different
        If we want to enable m > 0, then we will need to modify the cluster_point_matrix to take cluster intersections as well
    """

    cluster_point_matrix = np.float32(cluster_point_matrix)
    overlap_matrix = np.matmul(cluster_point_matrix, cluster_point_matrix.T)
    cluster_sizes = np.diag(overlap_matrix)
    indexer = np.ones(cluster_point_matrix.shape[0], dtype=np.bool)

    """
    Find all pairs of clusters with symmetric difference less than m, and then iterate through the (l1,l2)
        in the l2 reverse order. This allows us to guarantee that when we hit any (l1,l2),
        we have already hit all (l2, l3)
    """
    x_pair_vector, y_pair_vector = np.nonzero((cluster_sizes - overlap_matrix) == 0)

    # Only choose pairs that are fully equal, not just a containment relationship. These pairs should show up exactly
    #    twice between x_pair_vector and y_pair_vector
    pair_counts = Counter([tuple(sorted([l1, l2])) for (l1, l2) in zip(x_pair_vector, y_pair_vector) if l1 != l2])
    assert max(pair_counts.values()) <= 2
    unordered_pairs = [k for k, v in pair_counts.items() if v >= 2]
    reordered_pairs = [(l1, l2) for (l1, l2) in sorted(unordered_pairs, key=lambda I: -I[1])]
    seen_l2 = set()
    for l1, l2 in reordered_pairs:
        # remove row l2 and add its stability to row l1
        assert l1 not in seen_l2

        source_indices[l1] += source_indices[l2]
        stability_vector[l1] += stability_vector[l2]
        seen_l2.add(l2)
        indexer[l2] = False


    # Also return for each row the index of the parameter values at which the stability was computed
    return (
        cluster_point_matrix[indexer, :],
         stability_vector[indexer],
         [source_indices[i] for i, include in enumerate(indexer) if include])


def get_labels_from_cluster_point_matrix(cluster_point_matrix, stability_vector, source_indices, use_exact_solution=False):
    bool_cluster_point_matrix = np.array(cluster_point_matrix, dtype=np.bool)
    overlap_matrix = np.matmul(bool_cluster_point_matrix, bool_cluster_point_matrix.T)

    raw_overlap_x, raw_overlap_y = np.nonzero(overlap_matrix)
    selected_overlap = raw_overlap_x < raw_overlap_y
    overlap_x, overlap_y = raw_overlap_x[selected_overlap], raw_overlap_y[selected_overlap], 

    # One row for each pair of clusters that overlap, with a 1 in the position of each cluster
    A = np.zeros((cluster_point_matrix.shape[0], cluster_point_matrix.shape[0]))
    A = A + np.eye(cluster_point_matrix.shape[0]) * cluster_point_matrix.shape[0]
    A[overlap_x, overlap_y] = 1
    A[overlap_y, overlap_x] = 1

    b = np.ones(A.shape[0])*cluster_point_matrix.shape[0]

    solution = solve_zero_one_linear_program(
        c=-stability_vector,
        A=A,
        b=b,
        use_exact_solution=use_exact_solution)

#     print("solution", solution)
    # Assert no overlaps
    assert np.max(np.sum(cluster_point_matrix[solution, :], axis=0, dtype=np.float32)) <= 1
    
    solved_matrix = cluster_point_matrix[solution, :]
    cluster_to_source = [source_indices[i] for i, include in enumerate(solution) if include]
    cluster_to_stability = stability_vector[solution]

    solved_matrix_with_extra_row = np.zeros((solved_matrix.shape[0] + 1, solved_matrix.shape[1]))
    solved_matrix_with_extra_row[1:, :] = solved_matrix

    labels = (np.argmax(solved_matrix_with_extra_row, axis=0) - 1)
    return labels, cluster_to_stability, cluster_to_source


def get_labels_from_condensed_trees(condensed_tree_list, stabilities_list, use_exact_solution=False, normalize_stabilities=False):
    raw_cluster_point_matrix = combine_trees_into_matrix(condensed_tree_list)
    raw_stability_vector = get_stability_vector(stabilities_list, normalize=True)
    raw_source_indices = [[ix] for i, s in enumerate(stabilities_list) for ix in [i]*len(s)]

    cluster_point_matrix, stability_vector, source_indices = combine_clusters(
        raw_cluster_point_matrix, raw_stability_vector, raw_source_indices)
    return get_labels_from_cluster_point_matrix(
        cluster_point_matrix, stability_vector, source_indices, use_exact_solution=use_exact_solution)


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
        cluster_labels, _, _ = get_clusters(condensed_tree,
                                                          stability_dict,
                                                          cluster_selection_method='eom',
                                                          allow_single_cluster=False,
                                                          match_reference_implementation=False,
                                                          cluster_selection_epsilon=0.0)



        raw_tree_list.append(single_linkage_tree)
        condensed_tree_list.append(condensed_tree)
        stabilities_list.append(stability_dict)
        labels_list.append(cluster_labels)
    return raw_tree_list, condensed_tree_list, stabilities_list, labels_list


class MultiscaleHDBSCAN(object):


    def __init__(self, alpha_list, use_exact_solution=False, min_samples=5, min_cluster_size=10, normalize_stabilities=False):
        self.alpha_list = alpha_list
        self.use_exact_solution = use_exact_solution
        self.min_samples = min_samples
        self.min_cluster_size = min_cluster_size
        self.normalize_stabilities = normalize_stabilities

        
    def fit_predict(self, X):
        _, condensed_tree_list, stabilities_list, labels_list = run_multiscale_hdbscan(
            X=X, alpha_list=self.alpha_list)
        
        multiscale_labels, multiscale_cluster_to_stability, multiscale_cluster_to_source = get_labels_from_condensed_trees(
            condensed_tree_list=condensed_tree_list,
            stabilities_list=stabilities_list,
            use_exact_solution=self.use_exact_solution,
            normalize_stabilities=self.normalize_stabilities)
        
        return multiscale_labels



    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
# def get_labels_from_cluster_point_matrix(cluster_point_matrix, stability_vector, source_indices):
#     bool_cluster_point_matrix = np.array(cluster_point_matrix, dtype=np.bool)
#     overlap_matrix = np.matmul(bool_cluster_point_matrix, bool_cluster_point_matrix.T)

#     raw_overlap_x, raw_overlap_y = np.nonzero(overlap_matrix)
#     selected_overlap = raw_overlap_x < raw_overlap_y
#     overlap_x, overlap_y = raw_overlap_x[selected_overlap], raw_overlap_y[selected_overlap], 

#     # One row for each pair of clusters that overlap, with a 1 in the position of each cluster
#     A = np.zeros((len(overlap_x), cluster_point_matrix.shape[0]))
#     A[np.arange(len(A)), overlap_x] = 1
#     A[np.arange(len(A)), overlap_y] = 1
#     assert set(np.sum(A, axis=-1)) == {2}

#     solution = solve_zero_one_linear_program(
#             c=stability_vector,
#             A=A,
#             b=np.ones(A.shape[0]))

#     # Assert no overlaps
#     assert np.max(np.sum(cluster_point_matrix[solution, :], axis=0, dtype=np.float32)) <= 1
    
#     solved_matrix = cluster_point_matrix[solution, :]
#     cluster_to_source = [source_indices[i] for i, include in enumerate(solution) if include]
#     cluster_to_stability = stability_vector[solution]

#     solved_matrix_with_extra_row = np.zeros((solved_matrix.shape[0] + 1, solved_matrix.shape[1]))
#     solved_matrix_with_extra_row[1:, :] = solved_matrix

#     labels = (np.argmax(solved_matrix_with_extra_row, axis=0) - 1)
#     return labels, cluster_to_stability, cluster_to_source
        
    
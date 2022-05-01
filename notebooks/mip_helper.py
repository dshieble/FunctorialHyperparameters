from mip import Model, xsum, minimize, BINARY
import time
from copy import deepcopy

import annoy
import numpy as np
import pandas as pd
import seaborn as sns
import cvxpy
from scipy.optimize import linprog

CVXPY = "cvxpy"
MIP = "mip"


def get_mip_solution_from_cluster_point_matrix(cluster_point_matrix, stability_vector, solver):
    """
    Args:
        cluster_point_matrix: Matrix in which rows are clusters and columns are points
        stability_vector: Vector of cluster stabilities.
        solver: Either `CVXPY` or `MIP`            
    Returns:
        An array representing the clusters to select
    """
    bool_cluster_point_matrix = np.array(cluster_point_matrix, dtype=np.bool)
    overlap_matrix = np.dot(bool_cluster_point_matrix, bool_cluster_point_matrix.T)

    raw_overlap_x, raw_overlap_y = np.nonzero(overlap_matrix)
    selected_overlap = raw_overlap_x < raw_overlap_y
    overlap_x, overlap_y = raw_overlap_x[selected_overlap], raw_overlap_y[selected_overlap], 

    A = np.zeros((cluster_point_matrix.shape[0], cluster_point_matrix.shape[0]))
    A = A + np.eye(cluster_point_matrix.shape[0]) * cluster_point_matrix.shape[0]
    A[overlap_x, overlap_y] = 1
    A[overlap_y, overlap_x] = 1

    b = np.ones(A.shape[0])*cluster_point_matrix.shape[0]

    return solve_zero_one_linear_program(
        c=-stability_vector,
        A=A,
        b=b,
        solver=solver)


def solve_zero_one_linear_program(c, A, b, solver):
    """Minimize c*x
        x is binary
        A*c <= b
    """
    assert A.shape[1] == c.shape[0]
    assert A.shape[1] == b.shape[0]

    out = None
    if solver == CVXPY:
        start = time.time()
        print("Solving integer program of shape {}...".format(A.shape))
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

        out = np.array(list(problem.solution.primal_vars.values())[0], dtype=bool)
    elif solver == MIP:
        m = Model()

        x = [m.add_var(var_type=BINARY) for i in range(len(c))]

        m.objective = minimize(xsum(c[i] * x[i] for i in range(len(c))))

        for i in range(A.shape[0]):
            m += xsum(A[i,j] * x[j] for j in range(len(c))) <= b[i]
        m.optimize()

        out = np.array([x[i].x >= 0.99 for i in range(len(c))])
    else:
        raise ValueError("Solver {} not recognized".format(solver))
    assert out is not None
    return out

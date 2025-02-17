################################################################################################
# This code is a slightly modified version of the code from the NPEET library                  #
# https://github.com/gregversteeg/NPEET/blob/master/npeet/entropy_estimators.py                #
################################################################################################

import numpy as np
import numpy.linalg as la
from numpy import log
from scipy.special import digamma
from sklearn.neighbors import BallTree, KDTree

def add_noise(x, intens=1e-10):
    # small noise to break degeneracy, see doc.
    return x + intens * np.random.random_sample(x.shape)


def query_neighbors(tree, x, k):
    return tree.query(x, k=k + 1)[0][:, k]


def count_neighbors(tree, x, r):
    return tree.query_radius(x, r, count_only=True)


def avgdigamma(points, dvec):
    # This part finds number of neighbors in some radius in the marginal space
    # returns expectation value of <psi(nx)>
    tree = build_tree(points)
    dvec = dvec - 1e-15
    num_points = count_neighbors(tree, points, dvec)
    return np.mean(digamma(num_points))


def build_tree(points):
    if points.shape[1] >= 20:
        return BallTree(points, metric='chebyshev')
    return KDTree(points, metric='chebyshev')

def lnc_correction(tree, points, k, alpha):
    e = 0
    n_sample = points.shape[0]
    for point in points:
        # Find k-nearest neighbors in joint space, p=inf means max norm
        knn = tree.query(point[None, :], k=k+1, return_distance=False)[0]
        knn_points = points[knn]
        # Substract mean of k-nearest neighbor points
        knn_points = knn_points - knn_points[0]
        # Calculate covariance matrix of k-nearest neighbor points, obtain eigen vectors
        covr = knn_points.T @ knn_points / k
        _, v = la.eig(covr)
        # Calculate PCA-bounding box using eigen vectors
        V_rect = np.log(np.abs(knn_points @ v).max(axis=0)).sum()
        # Calculate the volume of original box
        log_knn_dist = np.log(np.abs(knn_points).max(axis=0)).sum()

        # Perform local non-uniformity checking and update correction term
        if V_rect < log_knn_dist + np.log(alpha):
            e += (log_knn_dist - V_rect) / n_sample
    return e

def mi(x, y, z=None, k=3, alpha=0):
    """ 
    Mutual information of x and y (conditioned on z if z is not None)
    x, y should be a list of vectors, e.g. x = [[1.3], [3.7], [5.1], [2.4]]
    if x is a one-dimensional scalar and we have four samples
    """
    assert len(x) == len(y), "Arrays should have same length"
    assert k <= len(x) - 1, "Set k smaller than num. samples - 1"
    x, y = np.asarray(x), np.asarray(y)
    x, y = x.reshape(x.shape[0], -1), y.reshape(y.shape[0], -1)
    x = add_noise(x)
    y = add_noise(y)
    points = [x, y]
    if z is not None:
        z = np.asarray(z)
        z = z.reshape(z.shape[0], -1)
        points.append(z)
    points = np.hstack(points)
    # Find nearest neighbors in joint space, p=inf means max-norm
    tree = build_tree(points)
    dvec = query_neighbors(tree, points, k)
    if z is None:
        a, b, c, d = avgdigamma(x, dvec), avgdigamma(
            y, dvec), digamma(k), digamma(len(x))
        if alpha > 0:
            d += lnc_correction(tree, points, k, alpha)
    else:
        xz = np.c_[x, z]
        yz = np.c_[y, z]
        a, b, c, d = avgdigamma(xz, dvec), avgdigamma(
            yz, dvec), avgdigamma(z, dvec), digamma(k)
    return (-a - b + c + d)

################################################################################################
# end of NPEET code                                                                            #
################################################################################################


def TE_knnksg(var_from_data, var_to_data, slice_size=5000, k=3):
    """
    Use kNN-KSG estimator to estimate transfer entropy from var_from to var_to

    TE(X -> Y) = MI(X-;Y0|Y-)

    Parameters:
    ----------
    var_from_data : np.array
        Data of the source variable
    var_to_data : np.array
        Data of the target variable

    Returns:
    ----------
    TE : float
        The estimated transfer entropy from var_from to var_to
    """    

    assert isinstance(var_from_data, np.ndarray), "var_from_data must be a numpy array"
    assert isinstance(var_to_data, np.ndarray), "var_to_data must be a numpy array"
    assert var_from_data.shape[0] == var_to_data.shape[0], "var_from_data and var_to_data must have the same number of samples"

    X_m = var_from_data[:-1] # X-
    Y_m = var_to_data[:-1]   # Y-
    Y_n = var_to_data[1:]    # Y0

    # data is split into smaller slices to avoid huge runtimes at higher dimensions
    slice_size = 5000
    results = []
    for i in range(0, len(X_m), slice_size):
        results.append(mi(X_m[i:i+slice_size], Y_n[i:i+slice_size], Y_m[i:i+slice_size], k=k))
    


    return np.round(np.mean(results),4)
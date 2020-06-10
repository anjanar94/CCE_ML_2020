from sklearn.preprocessing import StandardScaler
import numpy as np
import collections

def perform_pca(X):
    print('Execute PCA ...')
    data = []

    X_std = StandardScaler().fit_transform(X)

    mean_vec = np.mean(X_std, axis=0)
    cov_mat = np.cov(X_std.T)
    print('Covariance matrix \n%s' %cov_mat)

    eig_vals, eig_vecs = np.linalg.eig(cov_mat)
    print('\nEigenvalues \n%s' %eig_vals)
    print('Eigenvectors \n%s' %eig_vecs)

    eig_pairs = {}
    for i in range(len(eig_vals)):
        eig_pairs[eig_vals[i]] = eig_vecs[i]

    eig_pairs_od = {}
    for key in sorted(eig_pairs.keys(), reverse=True):
        eig_pairs_od[key] = eig_pairs[key]

    print(eig_pairs_od)

    return cov_mat, eig_vals, eig_vecs, eig_pairs_od

def dot_product(v1, v2):
    dp = 0.0
    for i in range(len(v1)):
        dp = dp + v1[i]*v2[i]
    print('Dot Product = ' + str(dp))
    return dp

#cov_mat, eig_vals, eig_vecs = perform_pca(X,Y)

#dot_product(eig_vecs[0], eig_vecs[1])
################################

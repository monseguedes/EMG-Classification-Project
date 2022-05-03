"""
@author: Monse Guedes Ayala
@project: Bioliberty EMG Classification Algorithm

Rough implementation of LDA without sklearn.
"""

import numpy as np

class LDA():
    def __int__():
        pass

    def fit(self, X, y):
        target_classes = np.unique(y)

        mean_vectors = []
 
        for class in target_classes:
            mean_vectors.append(np.mean(X[y == class], axis=0))    # Calculate the mean of each class and store it in mean_vectors  

        # Get between class scatter matrix
        data_mean = np.mean(X, axis=0).reshape(1, X.shape[1])
        between_class_scatter_matrix = np.zeros((X.shape[1], X.shape[1]))
        for i, mean_vec in enumerate(mean_vectors):
            n = X[y == i].shape[0]
            mean_vec = mean_vec.reshape(1, X.shape[1])
            mu1_mu2 = mean_vec - data_mean
        
            between_class_scatter_matrix += n * np.dot(mu1_mu2.T, mu1_mu2) 

        scatter_matrix = []    # List of scatter matrices for all classes
 
        # Calculate and add all scatter matrices
        for class, mean in enumerate(mean_vectors):
            Si = np.zeros((X.shape[1], X.shape[1]))
            for row in X[y == class]:
                t = (row - mean).reshape(1, X.shape[1])
                Si += np.dot(t.T, t)
            scatter_matrix.append(Si)

        # Get sum of all scatter matrices
        S = np.zeros((X.shape[1], X.shape[1]))
        for s_i in scatter_matrix:
            S += s_i

        S_inv = np.linalg.inv(S)    # Get inverse of S
        
        S_inv_between_class_scatter_matrix = S_inv.dot(between_class_scatter_matrix)    # This is S inverse times B
        
        eig_vals, eig_vecs = np.linalg.eig(S_inv_between_class_scatter_matrix)    # Get eigen vectors and eigen values

        idx = eig_vals.argsort()[::-1]
 
        eig_vals = eig_vals[idx]    # Not needed

        eig_vecs = eig_vecs[:, idx]    # Order eigen values and eigen vectors
        
        return eig_vecs



import numpy as np
from scipy.special import digamma, comb
from scipy.special import beta as scipy_beta
from scipy.spatial.distance import cdist
import math
from sklearn.cluster import KMeans
from calculate import softmax_matrix, multi_log_self, ln_beta, ln_comb

def myclone_fit(var_reads, ref_reads, K, max_iterations = 1e4, min_iterations=0, cnvs=[], threshold=0.02):
    # Parameters
    N, S = var_reads.shape
    alpha_0 = beta_0 = np.ones((K, S))
    # L_q_threshold = 1e-8

    # Initial
    r = np.zeros((N, K))
    r = k_cluster(var_reads, ref_reads, K, cnvs)
    old_r = r

    alpha = beta = np.ones((K, S))
    c = alpha/(alpha+beta)

    # Update parameters
    iteration = 0
    # L_q_old = 0
    while True:
        iteration += 1
        if iteration % 100 == 0:
            print(f"Epoch {iteration}")
        
        E_ln_c = digamma(alpha) - digamma(alpha+beta)
        E_ln_1_c = digamma(beta) - digamma(alpha+beta)
        rho_component_1 = np.dot(var_reads, E_ln_c.T)
        rho_component_2 = np.dot(ref_reads, E_ln_1_c.T)
        ln_rho = rho_component_1 + rho_component_2

        if iteration > 1:
            r = softmax_matrix(ln_rho)
        
        alpha_component = np.dot(r.T, var_reads)
        alpha = alpha_0 - 1 + alpha_component

        beta_component = np.dot(r.T, ref_reads)
        beta = beta_0 - 1 + beta_component

        epsilon = 1e-5
        alpha[alpha == 0] = epsilon
        beta[beta == 0] = epsilon

        c = alpha/(alpha+beta)
        c[alpha <= epsilon] = 0

        # Converge Condition
        kl_threshold = 1e-8
        if iteration > min_iterations and old_r.shape == r.shape:
            if check_kl_divergence(old_r, r, epsilon=kl_threshold):
                # print(old_r)
                # print(r)
                break
        old_r = r

        if iteration >= max_iterations: break

        zero_rows = np.all(c == 0, axis=1)
        zero_row_indices = np.where(zero_rows)[0]
        alpha = np.delete(alpha, zero_row_indices, axis=0)
        beta = np.delete(beta, zero_row_indices, axis=0)
        alpha_0 = np.delete(alpha_0, zero_row_indices, axis=0)
        beta_0 = np.delete(beta_0, zero_row_indices, axis=0)

        # dis_epsilon = 0.025
        # c_distances = cdist(c, c, 'euclidean')
        # c_pairs = np.argwhere((c_distances > 0) & (c_distances < dis_epsilon))
        # if len(c_pairs) > 0:
        #     to_delete_indices = [c_pairs[0][0]]
        #     alpha = np.delete(alpha, to_delete_indices, axis=0)
        #     beta = np.delete(beta, to_delete_indices, axis=0)
        #     alpha_0 = np.delete(alpha_0, to_delete_indices, axis=0)
        #     beta_0 = np.delete(beta_0, to_delete_indices, axis=0)
        
        dis_epsilon = threshold
        c_row_num = c.shape[0]
        zero_pos = np.where(c == 0, 0, 1)
        found_match = False
        for i in range(c_row_num-1):
            if found_match:
                break
            for j in range(i+1, c_row_num):
                if not np.array_equal(zero_pos[i], zero_pos[j]): continue
                # nonzero_num = np.count_nonzero(zero_pos[i])
                nonzero_num = S
                dis_theroshold = np.sqrt(nonzero_num) * dis_epsilon
                e_distance = np.linalg.norm(c[i] - c[j])
                if e_distance < dis_theroshold:
                    to_delete_index = i
                    found_match = True
                    break
        if found_match:
            to_delete_indices = [to_delete_index]
            alpha = np.delete(alpha, to_delete_indices, axis=0)
            beta = np.delete(beta, to_delete_indices, axis=0)
            alpha_0 = np.delete(alpha_0, to_delete_indices, axis=0)
            beta_0 = np.delete(beta_0, to_delete_indices, axis=0)

        if iteration == 1:
            np.set_printoptions(precision=3, suppress=True)
            # print()
            # print(alpha)
            # print(beta)
            # print(r)
            # print(c)
            # print(c_distances)
        
    print("Total Interations: ", iteration)
    # print(alpha)
    # print(beta)
    # print(c)
    # print(r)
    
    return r, c

def k_cluster(var_reads, ref_reads, K, cnvs):
    vaf = var_reads / (var_reads + ref_reads)
    kmeans_model = KMeans(n_clusters=K)
    kmeans_model.fit(vaf)
    clus_labels = kmeans_model.labels_
    inital_r = np.eye(K)[clus_labels]
    return inital_r

def check_kl_divergence(matrix1, matrix2, epsilon=1e-10):
    def smooth_probabilities(matrix, epsilon=1e-10):
        return (matrix + epsilon) / np.sum(matrix + epsilon, axis=1, keepdims=True)
    
    assert matrix1.shape == matrix2.shape
    
    matrix1_smoothed = smooth_probabilities(matrix1)
    matrix2_smoothed = smooth_probabilities(matrix2)
    
    kl_divergences = np.sum(matrix1_smoothed * np.log(matrix1_smoothed / matrix2_smoothed), axis=1)
    
    return np.all(kl_divergences < epsilon)

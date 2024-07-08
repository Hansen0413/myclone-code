import numpy as np
import scipy.special as sp
from scipy.stats import entropy
from scipy.spatial import distance
import random
import time

def timed(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        ela = end - start
        f_ela = time.strftime("%H:%M:%S", time.gmtime(ela))
        print(f"{func.__name__} Running Time: {f_ela}")
        return result
    return wrapper

def softmax_matrix(rho): # every row -> softmax
    softmax_rows = np.exp(rho - np.max(rho, axis=1, keepdims=True))
    softmax_rows /= np.sum(softmax_rows, axis=1, keepdims=True)
    return softmax_rows

def multi_log_self(r): # r * np.log(r)
    N, K = r.shape
    new_r = np.zeros((N,K))
    for i in range(N):
        for j in range(K):
            if r[i,j] == 0:
                new_r[i,j] = 0
            else:
                new_r[i,j] = r[i,j] * np.log(r[i,j])
    return new_r

def ln_beta(A,B):
    M, N = A.shape
    R = np.zeros((M,N))
    for i in range(M):
        for j in range(N):
            R[i,j] = sp.gammaln(A[i,j]) + sp.gammaln(B[i,j]) - sp.gammaln(A[i,j] + B[i,j])
    return R

def ln_comb(A,B):
    M, N = A.shape
    R = np.zeros((M,N))
    for i in range(M):
        for j in range(N):
            R[i,j] = sp.gammaln(A[i,j]+1) - sp.gammaln(B[i,j]+1) - sp.gammaln(A[i,j]-B[i,j]+1)
    return R

def cal_JSD(p, q):
    m = 0.5 * (p + q)
    kl_p = entropy(p, m)
    kl_q = entropy(q, m)
    jsd = 0.5 * (kl_p + kl_q)
    return jsd

def AJSD_old(ori_A,ori_B):
    A = np.copy(ori_A)
    B = np.copy(ori_B)
    assert A.shape[0] == A.shape[1] and A.shape[1] == B.shape[0] and B.shape[0] == B.shape[1]
    N = A.shape[0]
    total_score = 0
    for i in range(N):
        A_row = A[i, :]
        B_row = B[i, :]
        # A_row[i] = 0
        # B_row[i] = 0
        s_A_row = standard_epsilon(A_row, 0.01)
        s_B_row = standard_epsilon(B_row, 0.01)
        score = cal_JSD(s_A_row, s_B_row)
        total_score += score
    total_score /= N
    return np.exp(-total_score)

@timed
def AJSD(ori_A,ori_B, alpha=0):
    A = np.copy(ori_A)
    B = np.copy(ori_B)
    assert A.shape[0] == A.shape[1] and A.shape[1] == B.shape[0] and B.shape[0] == B.shape[1]
    N = A.shape[0]
    total_score = 0
    for i in range(N):
        A_row = A[i, :]
        B_row = B[i, :]
        s_A_row = standard_epsilon(A_row, alpha)
        s_B_row = standard_epsilon(B_row, alpha)
        score = distance.jensenshannon(s_A_row, s_B_row)
        total_score += score ** 2
    total_score /= N
    # avg_js_distance = distance.jensenshannon(A, B, axis=1)
    # total_score = np.sum(avg_js_distance ** 2) / len(avg_js_distance)
    return np.exp(-total_score)

def standard_epsilon(ori_row, alpha):
    row = np.copy(ori_row)
    assert np.all(row >= 0) and np.sum(row) > 0
    row /= np.sum(row)
    N = len(row)
    s_row = np.zeros(N)
    for i in range(N):
        s_row[i] = ((1-alpha) * row[i] + alpha) / (1-alpha + N*alpha)
    return s_row

def Similarity_matrix(A,B):
    assert A.shape == B.shape
    n_same_elements = np.sum(A==B)
    sim = n_same_elements / A.size
    return sim

def random_partition(number, parts):
    partition = []
    
    for _ in range(parts - 1):
        part = random.randint(1, number - (parts - len(partition)) + 1)
        partition.append(part)
        number -= part
    
    partition.append(number)
    return partition

def random_N_distribution(num_elements):
    random_probs = np.random.rand(num_elements)
    normalized_probs = random_probs / np.sum(random_probs)
    return normalized_probs

def cal_expected_cn(mut_vaf, clus_vaf, mut_cn, ref_cn, true_cn, mut_id):
    if mut_cn == 0 and ref_cn == 0:
        return 0, 0, 0
    
    # Case 1: mut_cn change
    new_mut_cn = ref_cn * mut_vaf / (2*clus_vaf - mut_vaf)
    # Case 2: ref_cn change
    new_ref_cn = mut_cn * (2*clus_vaf - mut_vaf) / mut_vaf

    e_total_cn_1 = new_mut_cn + ref_cn
    e_total_cn_2 = mut_cn + new_ref_cn
    # if mut_id == 106496896:
    #     print('------')
    #     print(e_total_cn_1)
    #     print(e_total_cn_2)
    #     print('clus_vaf:', clus_vaf)
    #     print(true_cn)

    if mut_cn == 0:
        return e_total_cn_2, mut_cn, round(new_ref_cn)
    elif (abs(e_total_cn_1-true_cn) < abs(e_total_cn_2-true_cn)) or (ref_cn == 0):
        return e_total_cn_1, round(new_mut_cn), ref_cn
    else:
        return e_total_cn_2, mut_cn, round(new_ref_cn)

def best_k_means(array):
    pass
import numpy as np
from scipy.stats import poisson, binom

def cn_to_logratio(cn, cancer_frac, ploidy):
    return np.log2(cancer_frac * (cn / ploidy - 1) + 1)

def expected_hsaf(cancer_frac, b_allele_cn, total_cn, depth=0, iterations=10_000, use_table=False):
    assert b_allele_cn <= total_cn
    true_hsaf = (cancer_frac * (b_allele_cn - 1) + 1) / (cancer_frac * (total_cn - 2) + 2)
    
    # if depth == 0:
    #     return true_hsaf
    # # elif use_table:
    # #     return hsaf_deviation_from_table(true_hsaf, depth)
    # else:
    #     # Run a simulation to estimate the expected HSAF 
    #     return simulate_hsaf_deviation(true_hsaf, depth, iterations=iterations)
    return true_hsaf

def simulate_hsaf_deviation(true_hsaf, depth, iterations=10_000):
    true_hsaf = float(true_hsaf)
    deviation = np.zeros(iterations)
    d = 0
    step = int(np.ceil(iterations / 100))
    
    for k in range(1, iterations + 1, step):
        dpt = np.random.poisson(depth)
        dist = binom(dpt, true_hsaf)
        
        for _ in range(min(step, iterations - k + 1)):
            d += 1
            deviation[d] = 0.5 + abs(0.5 - dist.rvs() / dpt)
    
    hsaf = np.median(deviation)
    
    if true_hsaf < 0.5:
        hsaf = 1 - hsaf
    
    return hsaf
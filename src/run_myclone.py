from fitting import myclone_fit
from methods import SNV_matrices, readtsv, output_tsv, create_dir, predict_time, record_time, cln_SNV_matrices, cluster_lis
import numpy as np
import pandas as pd
from calculate import AJSD, Similarity_matrix, best_k_means
import os
import time
import warnings
import sys
import argparse

class Clone:
    def __init__(self, id, mut_ids):
        self.id = id
        self.ccfs = []
        self.mut_ids = mut_ids
        self.avg_vafs = []
    
    def add_mut(self, mut_id):
        self.mut_ids.append(mut_id)

def read_purity_values(tsvs):
    purity_values = []
    
    for tsv in tsvs:
        purity_file = os.path.splitext(tsv)[0] + "_purity.tsv"
        try:
            with open(purity_file, 'r') as f:
                purity_value = float(f.readline().strip())
                purity_values.append(purity_value)
        except Exception as e:
            raise ValueError(f"Error reading purity file '{purity_file}': {e}")
    
    return purity_values

def r_assign_matrix(r):
    print("Running r_assign..")

    N = r.shape[0]
    r_assign = np.argmax(r, axis=1)
    assign_matrix = np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if j < i:
                assign_matrix[i,j] = assign_matrix[j,i]
                continue
            if r_assign[i] == r_assign[j]:
                assign_matrix[i,j] = 1
    
    r_01_matrix = np.zeros(r.shape)
    for i in range(N):
        r_01_matrix[i, r_assign[i]] = 1
    
    return assign_matrix, r_01_matrix

def pyclone_assign_matrix(loci_tsv, id_lis):
    N = len(id_lis)
    assign_m = np.zeros((N,N))
    df = readtsv(loci_tsv)
    for i in range(N):
        for j in range(N):
            if j < i:
                assign_m[i,j] = assign_m[j,i]
                continue
            clus_id_i = df[df['mutation_id'] == id_lis[i]]['cluster_id'].values[0]
            clus_id_j = df[df['mutation_id'] == id_lis[j]]['cluster_id'].values[0]
            if clus_id_i == clus_id_j:
                assign_m[i,j] = 1
    return assign_m

def cal_avg_vafs(clone, tsvs):
    clone.avg_vafs = []

    dfs = []
    for tsv in tsvs:
        df = readtsv(tsv)
        dfs.append(df)
    
    mut_ids = clone.mut_ids
    for df in dfs:
        vafs = []
        for mut_id in mut_ids:
            vaf = df[df['mutation_id'] == mut_id]['VAF'].values[0]
            vafs.append(vaf)
        avg_vaf = sum(vafs)/len(vafs)
        clone.avg_vafs.append(avg_vaf)

def get_direct_clone_lis(r_assign, id_lis):
    assert(r_assign.shape[0] == len(id_lis))

    clone_list = []
    for col_index, col in enumerate(r_assign.T):
        mut_indices = np.where(col == 1)[0].tolist()
        mut_ids = [id_lis[i] for i in mut_indices]
        clone = Clone(id=col_index, mut_ids=mut_ids)
        clone_list.append(clone)
    
    return clone_list

def get_clone_lis(r_assign, id_lis, tsvs):
    print("Getting clone lis..")
    assert(r_assign.shape[0] == len(id_lis))

    clone_list = []
    for col_index, col in enumerate(r_assign.T):
        mut_indices = np.where(col == 1)[0].tolist()
        mut_ids = [id_lis[i] for i in mut_indices]
        clone = Clone(id=col_index, mut_ids=mut_ids)
        clone_list.append(clone)
    print(f"Len1: {len(clone_list)}")
    
    dfs = []
    for tsv in tsvs:
        df = readtsv(tsv)
        dfs.append(df)
    
    del_indices_0 = []
    add_clones_0 = []
    for index, clone in enumerate(clone_list):
        mut_ids = clone.mut_ids
        ifzero_tuples = []
        for mut_id in mut_ids:
            if_zeros = []
            for df in dfs:
                var_counts = df[df['mutation_id'] == mut_id]['var_counts'].values[0]
                if var_counts == 0:
                    if_zeros.append(0)
                else:
                    if_zeros.append(1)
            ifzero_tuples.append(tuple(if_zeros))
        simple_ifzero_tuples = list(set(ifzero_tuples))

        if len(simple_ifzero_tuples) > 1:
            del_indices_0.append(index)
            for target_ifzero_tuple in simple_ifzero_tuples:
                new_mut_ids = []
                for mut_index in range(len(ifzero_tuples)):
                    if ifzero_tuples[mut_index] == target_ifzero_tuple:
                        new_mut_ids.append(mut_ids[mut_index])
                new_clone = Clone(id=len(clone_list)+len(add_clones_0), mut_ids=new_mut_ids)
                add_clones_0.append(new_clone)
    clone_list_1 = [clone_list[i] for i in range(len(clone_list)) if i not in del_indices_0] + add_clones_0

    print(f"Len2: {len(clone_list_1)}")
    # print_clones(clone_list_1)
    
    del_indices = []
    add_clones = []
    for index, clone in enumerate(clone_list_1):
        mut_ids = clone.mut_ids
        cn_tuples = []
        for mut_id in mut_ids:
            total_cns = []
            for df in dfs:
                total_cns.append(df[df['mutation_id'] == mut_id]['total_cn'].values[0])
            cn_tuples.append(tuple(total_cns))
        simple_cn_tuples = list(set(cn_tuples))

        if len(simple_cn_tuples) > 1:
            del_indices.append(index)
            for target_cn_tuple in simple_cn_tuples:
                new_mut_ids = []
                for mut_index in range(len(cn_tuples)):
                    if cn_tuples[mut_index] == target_cn_tuple:
                        new_mut_ids.append(mut_ids[mut_index])
                new_clone = Clone(id=len(clone_list)+len(add_clones_0)+len(add_clones), mut_ids=new_mut_ids)
                add_clones.append(new_clone)
    final_list = [clone_list_1[i] for i in range(len(clone_list_1)) if i not in del_indices] + add_clones

    print(f"Len3: {len(final_list)}")

    for final_index, clone in enumerate(final_list):
        clone.id = final_index

    return final_list

def judge_if_cnved(clone, df):
    # total_vaf = df[df['mutation_id'].isin(clone.mut_ids)]['VAF'].sum()
    # if total_vaf == 0: return False

    for mut_id in clone.mut_ids:
        total_cn = df[df['mutation_id'] == mut_id]['total_cn'].values[0]
        if total_cn != 2:
            return True
    return False

def get_max_ccf(clone, df):
    vafs = []
    for mut_id in clone.mut_ids:
        vaf = df[df['mutation_id'] == mut_id]['VAF'].values[0]
        vafs.append(vaf)
    avg_vaf = sum(vafs)/len(vafs)

    cn_lis = list(set([df[df['mutation_id'] == mut_id]['total_cn'].values[0] for mut_id in clone.mut_ids]))
    assert(len(cn_lis) == 1)

    max_ccf = avg_vaf * cn_lis[0]
    if avg_vaf > 0.55 and cn_lis[0] == 2.0:
        max_ccf = avg_vaf
    return min(max_ccf, 1)

def cal_true_ccf(N, total_cn, main_cn, vaf):
    N_ccf = (total_cn-2)/(N-2)
    if N_ccf > 1.1: return -1
    elif N_ccf > 1:
        N_ccf = 1
    true_ccf = vaf * total_cn - (main_cn-1) * N_ccf
    return true_ccf

def match_ccf(target_ccf, ccf_lis, threshold):
    for id, ccf in enumerate(ccf_lis):
        if np.abs(ccf-target_ccf) < threshold * max(ccf_lis):
            return id
    return -1

def match_ccfs(target_ccfs, ccfs_lis, threshold):
    min_e_distance = 10000
    S = len(target_ccfs)
    for id, ccfs in enumerate(ccfs_lis):
        t_zero_pos = np.where(np.array(target_ccfs) == 0, 0, 1)
        s_zero_pos = np.where(np.array(ccfs) == 0, 0, 1)
        if not np.array_equal(t_zero_pos, s_zero_pos): continue

        # nonzero_num = np.count_nonzero(t_zero_pos)
        nonzero_num = S
        dis_theroshold = np.sqrt(nonzero_num) * threshold
        e_distance = np.linalg.norm(np.array(target_ccfs) - np.array(ccfs))
        if e_distance < dis_theroshold:
            return id, e_distance
        if e_distance < min_e_distance:
            min_e_distance = e_distance
    return -1, min_e_distance

def predict_tumor_fraction(df, clone_lis):
    max_ccf_lis = []
    cln_ccf_lis = []
    for clone in clone_lis:
        max_ccf_lis.append(get_max_ccf(clone, df))
        if not judge_if_cnved(clone, df):
            cln_ccf_lis.append(get_max_ccf(clone, df))
    sorted_indices = sorted(range(len(max_ccf_lis)), key=lambda i: max_ccf_lis[i], reverse=True)

    for index in sorted_indices:
        clone_i = clone_lis[index]
        if_cnved_i = judge_if_cnved(clone_i, df)
        if not if_cnved_i:
            tumor_fraction = max_ccf_lis[index]
            break
        else:
            total_cn = df[df['mutation_id'] == clone_i.mut_ids[0]]['total_cn'].values[0]
            if_match_cln_ccf = False
            for N in range(3,6):
                if if_match_cln_ccf == True: break
                for main_cn in range(2,N):
                    prob_ccf = cal_true_ccf(N, total_cn, main_cn, clone_i.avg_vafs[0])
                    if match_ccf(prob_ccf, cln_ccf_lis, threshold=0.025) != -1:
                        if_match_cln_ccf = True
                        break
            if if_match_cln_ccf:
                continue
            else:
                tumor_fraction = max_ccf_lis[index]
                break
    
    return tumor_fraction

def predict_tumor_fractions(tsvs, clone_lis, threshold):
    print("Predicting tumor fractions..")

    # cal avg_vafs
    for clone in clone_lis:
        cal_avg_vafs(clone, tsvs)

    dfs = []
    for tsv in tsvs:
        df = readtsv(tsv)
        dfs.append(df)

    cln_ccfs_lis = []
    cln_clones = []
    cnved_clones = []
    cnved_clone_df_ids = []
    for clone in clone_lis:
        is_cnved = False
        for df_id, df in enumerate(dfs):
            if judge_if_cnved(clone, df):
                is_cnved = True
                cnved_clone_df_ids.append(df_id)
                break
        if not is_cnved:
            cln_clones.append(clone)
            ccfs = []
            for df in dfs:
                ccfs.append(get_max_ccf(clone, df))
            cln_ccfs_lis.append(ccfs)
        else:
            cnved_clones.append(clone)
    
    # cln_clones = clus_cln_clones(cln_clones)
    # for clone in cln_clones:
    #     cal_avg_vafs(clone, tsvs)
    # cln_ccfs_lis = []
    # for clone in cln_clones:
    #     ccfs = []
    #     for df in dfs:
    #         ccfs.append(get_max_ccf(clone, df))
    #     cln_ccfs_lis.append(ccfs)

    final_lis = cln_clones
    # update cln ccfs
    for c_order, clone in enumerate(final_lis):
        clone.ccfs = cln_ccfs_lis[c_order]
    # print("CLEAN clone num: ", len(cln_clones))
    # print("CNVed clone num: ", len(cnved_clones))
    # print_clones(cln_clones)
    cnved_ccfs_lis = []
    for c_clone_index, c_clone in enumerate(cnved_clones):
        # print(f"Processing {clone_index}th clone..")
        cnv_sample_id = cnved_clone_df_ids[c_clone_index]
        match_df_0 = dfs[cnv_sample_id]
        total_cn_0 = match_df_0[match_df_0['mutation_id'] == c_clone.mut_ids[0]]['total_cn'].values[0]

        if_match_cln_ccf = False
        min_e_distance = 10000
        most_prob_ccfs = [0,0,0]
        if total_cn_0 < 2:
            for main_cn in range(0,2):
                prob_ccfs = []
                for i in range(cnv_sample_id):
                    prob_ccfs.append(get_max_ccf(c_clone, dfs[i]))
                
                for i in range(cnv_sample_id, len(dfs)):
                    match_df = dfs[i]
                    total_cn = match_df[match_df['mutation_id'] == c_clone.mut_ids[0]]['total_cn'].values[0]
                    prob_ccfs.append(cal_true_ccf(1, total_cn, main_cn, c_clone.avg_vafs[i]))

                match_id, e_distance = match_ccfs(prob_ccfs, cln_ccfs_lis, threshold=threshold)
                if match_id != -1:
                    if_match_cln_ccf = True
                    break
                if e_distance < min_e_distance:
                    min_e_distance = e_distance
                    most_prob_ccfs = prob_ccfs
        
        else:
            for N in range(3,6):
                if if_match_cln_ccf == True: break
                for main_cn in range(1,N):
                    prob_ccfs = []
                    for i in range(cnv_sample_id):
                        prob_ccfs.append(get_max_ccf(c_clone, dfs[i]))
                    
                    for i in range(cnv_sample_id, len(dfs)):
                        match_df = dfs[i]
                        total_cn = match_df[match_df['mutation_id'] == c_clone.mut_ids[0]]['total_cn'].values[0]
                        prob_ccfs.append(cal_true_ccf(N, total_cn, main_cn, c_clone.avg_vafs[i]))
                    
                    match_id, e_distance = match_ccfs(prob_ccfs, cln_ccfs_lis, threshold=threshold)
                    if match_id != -1:
                        if_match_cln_ccf = True
                        break
                    if e_distance < min_e_distance:
                        min_e_distance = e_distance
                        most_prob_ccfs = prob_ccfs
        
        if if_match_cln_ccf:
            match_clone = cln_clones[match_id]
            assert c_clone.id != match_clone.id
            for mut_id in c_clone.mut_ids:
                match_clone.add_mut(mut_id)
        else:
            c_clone.ccfs = most_prob_ccfs
            final_lis.append(c_clone)
            ccfs = []
            for df in dfs:
                ccfs.append(get_max_ccf(c_clone, df))
            cnved_ccfs_lis.append(ccfs)
    
    ccfs_lis = cln_ccfs_lis + cnved_ccfs_lis
    tumor_fractions = []
    for i in range(len(dfs)):
        tumor_fractions.append(max([s[i] for s in ccfs_lis]))
    
    return tumor_fractions

def predict_tumor_fraction_cln(df, clone_lis):
    cln_clones = [clone for clone in clone_lis if not judge_if_cnved(clone, df)]
    cln_clones = [clone for clone in cln_clones if len(clone.mut_ids) > 5]
    tmp_lis = [get_max_ccf(clone, df) for clone in cln_clones]
    tmp_lis = [a for a in tmp_lis if a < 1.1]
    max_ccf = max(tmp_lis)
    if max_ccf > 1:
        max_ccf = 1
    return max_ccf

def predict_tumor_fractions_cln(tsvs, clone_lis):
    print("Predicting tumor fractions..")

    # cal avg_vafs
    for clone in clone_lis:
        cal_avg_vafs(clone, tsvs)

    dfs = []
    for tsv in tsvs:
        df = readtsv(tsv)
        dfs.append(df)
    
    tumor_fractions = []
    for df in dfs:
        tumor_fractions.append(predict_tumor_fraction_cln(df, clone_lis))
    
    return tumor_fractions

def update_df(df, tumor_fraction):
    for row_index, row in df.iterrows():
        var_reads = row['var_counts']
        ref_reads = row['ref_counts']
        vaf = var_reads / (var_reads + ref_reads)
        total_cn = row['total_cn']
        new_vaf = min(vaf * total_cn / (total_cn - 2 * (1-tumor_fraction)), 1)
        df.loc[row_index, 'VAF'] = new_vaf
        new_total_cn = (total_cn - 2 * (1-tumor_fraction)) / tumor_fraction
        new_total_cn = round(new_total_cn, 2)
        df.loc[row_index, 'total_cn'] = new_total_cn
        new_ref_reads = max(round(ref_reads - (1-tumor_fraction) * (var_reads+ref_reads)), 0)
        df.loc[row_index, 'ref_counts'] = new_ref_reads

        if new_vaf > 0.55 and new_total_cn == 2.0: 
            tmp_reads = round(var_reads/2)
            new_var_reads = var_reads - tmp_reads
            new_ref_reads = new_ref_reads + tmp_reads
            new_vaf = new_vaf / 2
            df.loc[row_index, 'VAF'] = new_vaf
            df.loc[row_index, 'var_counts'] = new_var_reads
            df.loc[row_index, 'ref_counts'] = new_ref_reads
    
    return df

def update_tsvs(tsvs, tumor_fractions):
    assert(len(tsvs) == len(tumor_fractions))

    tmp_dir = 'tmp/'
    if not os.path.exists(tmp_dir):
        os.makedirs(tmp_dir)
    
    dfs = []
    for tsv in tsvs:
        df = readtsv(tsv)
        dfs.append(df)
    
    index = 0
    for df, tumor_fraction in zip(dfs, tumor_fractions):
        index += 1
        output_name = tmp_dir + f'new-0{index}.tsv'
        new_df = update_df(df, tumor_fraction)
        new_df.to_csv(output_name, sep='\t', index=False)

def match_cnved_clones(clone_lis, tsvs, threshold):
    print("Matching CNVed muts..")

    # cal avg_vafs
    for clone in clone_lis:
        cal_avg_vafs(clone, tsvs)

    dfs = []
    for tsv in tsvs:
        df = readtsv(tsv)
        dfs.append(df)

    cln_ccfs_lis = []
    cln_clones = []
    cnved_clones = []
    cnved_clone_df_ids = []
    for clone in clone_lis:
        is_cnved = False
        for df_id, df in enumerate(dfs):
            if judge_if_cnved(clone, df):
                is_cnved = True
                cnved_clone_df_ids.append(df_id)
                break
        if not is_cnved:
            cln_clones.append(clone)
            ccfs = []
            for df in dfs:
                ccfs.append(get_max_ccf(clone, df))
            cln_ccfs_lis.append(ccfs)
        else:
            cnved_clones.append(clone)
    
    cln_clones = clus_cln_clones(cln_clones)
    for clone in cln_clones:
        cal_avg_vafs(clone, tsvs)
    cln_ccfs_lis = []
    for clone in cln_clones:
        ccfs = []
        for df in dfs:
            ccfs.append(get_max_ccf(clone, df))
        cln_ccfs_lis.append(ccfs)

    final_lis = cln_clones
    # update cln ccfs
    for c_order, clone in enumerate(final_lis):
        clone.ccfs = cln_ccfs_lis[c_order]
    print("CLEAN clone num: ", len(cln_clones))
    print("CNVed clone num: ", len(cnved_clones))
    # print_clones(cln_clones)
    for c_clone_index, c_clone in enumerate(cnved_clones):
        # print(f"Processing {clone_index}th clone..")
        cnv_sample_id = cnved_clone_df_ids[c_clone_index]
        match_df_0 = dfs[cnv_sample_id]
        total_cn_0 = match_df_0[match_df_0['mutation_id'] == c_clone.mut_ids[0]]['total_cn'].values[0]

        if_match_cln_ccf = False
        min_e_distance = 10000
        most_prob_ccfs = [0,0,0]
        if total_cn_0 < 2:
            for main_cn in range(0,2):
                prob_ccfs = []
                for i in range(cnv_sample_id):
                    prob_ccfs.append(get_max_ccf(c_clone, dfs[i]))
                
                for i in range(cnv_sample_id, len(dfs)):
                    match_df = dfs[i]
                    total_cn = match_df[match_df['mutation_id'] == c_clone.mut_ids[0]]['total_cn'].values[0]
                    prob_ccfs.append(cal_true_ccf(1, total_cn, main_cn, c_clone.avg_vafs[i]))

                match_id, e_distance = match_ccfs(prob_ccfs, cln_ccfs_lis, threshold=threshold)
                if match_id != -1:
                    if_match_cln_ccf = True
                    break
                if e_distance < min_e_distance:
                    min_e_distance = e_distance
                    most_prob_ccfs = prob_ccfs
        
        else:
            for N in range(3,6):
                if if_match_cln_ccf == True: break
                for main_cn in range(1,N):
                    prob_ccfs = []
                    for i in range(cnv_sample_id):
                        prob_ccfs.append(get_max_ccf(c_clone, dfs[i]))
                    
                    for i in range(cnv_sample_id, len(dfs)):
                        match_df = dfs[i]
                        total_cn = match_df[match_df['mutation_id'] == c_clone.mut_ids[0]]['total_cn'].values[0]
                        prob_ccfs.append(cal_true_ccf(N, total_cn, main_cn, c_clone.avg_vafs[i]))
                    
                    match_id, e_distance = match_ccfs(prob_ccfs, cln_ccfs_lis, threshold=threshold)
                    if match_id != -1:
                        if_match_cln_ccf = True
                        break
                    if e_distance < min_e_distance:
                        min_e_distance = e_distance
                        most_prob_ccfs = prob_ccfs
        
        if if_match_cln_ccf:
            match_clone = cln_clones[match_id]
            assert c_clone.id != match_clone.id
            for mut_id in c_clone.mut_ids:
                match_clone.add_mut(mut_id)
        else:
            c_clone.ccfs = most_prob_ccfs
            final_lis.append(c_clone)
    
    for index, clone in enumerate(final_lis):
        clone.id = index + 1
    print(f"Final Clone Num: {len(final_lis)}")
    print("============================================================================================")
    
    return final_lis

def predict_assign_square(clone_lis, id_lis):
    N = len(id_lis)
    assign_matrix = np.zeros((N,N))
    for clone in clone_lis:
        index_lis = [id_lis.index(mut_id) for mut_id in clone.mut_ids]
        for i in index_lis:
            for j in index_lis:
                assign_matrix[i,j] = 1
    
    return assign_matrix

def print_clone(clone):
    print('------------------------------')
    print(f'Clone {clone.id+1}')
    print(f"Mutation num: {len(clone.mut_ids)}")
    print(f"Mutation_ids: {clone.mut_ids}")
    print(f"Avg VAFs: {clone.avg_vafs}")

def print_clones(clone_lis):
    for clone in clone_lis:
        print_clone(clone)
    print('------------------------------')

def output_clones(clone_lis, output_dir):
    print("Outputing clones..")

    new_df_data = {
        "mutation_id": [],
        "myclone_id": [],
        "myclone_ccf": []
    }
    new_df = pd.DataFrame(new_df_data)

    for clone in clone_lis:
        mut_ids = clone.mut_ids
        ccfs = clone.ccfs
        ccfs_str = ','.join([str(c) for c in ccfs])
        for mut_id in mut_ids:
            new_df = new_df.append({'mutation_id': f"{mut_id}", 'myclone_id': f"{clone.id}", 'myclone_ccf': ccfs_str}, ignore_index=True)
    
    output_file = os.path.join(output_dir, "myclone_result.tsv")
    output_tsv(new_df, output_file)

def clus_cln_clones_old(clone_lis):
    final_clone_lis = []
    array = np.zeros((len(clone_lis), len(clone_lis[0].avg_vafs)))
    for i in range(len(clone_lis)):
        avg_vafs = clone_lis[i].avg_vafs
        for j in range(len(avg_vafs)):
            array[i, j] = avg_vafs[j]
    r_dic = best_k_means(array)
    for index, id_lis in enumerate(r_dic.values()):
        for i in range(len(id_lis)):
            if i == 0:
                final_clone_lis.append(clone_lis[id_lis[i]])
            else:
                mut_ids = clone_lis[id_lis[i]].mut_ids
                for mut in mut_ids:
                    final_clone_lis[index].add_mut(mut)
    return final_clone_lis

def add_clone(clone_i, clone_j):
    mut_ids = clone_j.mut_ids
    for mut in mut_ids:
        clone_i.add_mut(mut)
    return clone_i

def clus_cln_clones(clone_lis, threshold=0.03):
    N = len(clone_lis)
    S = len(clone_lis[0].avg_vafs)
    for i in range(N):
        for j in range(i+1, N):
            vaf_i = clone_lis[i].avg_vafs
            vaf_j = clone_lis[j].avg_vafs
            if np.linalg.norm(np.array(vaf_i) - np.array(vaf_j)) < np.sqrt(S) * threshold:
                clone_lis[i] = add_clone(clone_lis[i], clone_lis[j])
                del clone_lis[j]
                return clus_cln_clones(clone_lis, threshold=threshold)
    return clone_lis

def run_myclone(tsvs, output_dir, purity_lis=[], cnv_threshold=0.05):
    # Prepare
    create_dir(output_dir)
    sample_num = len(tsvs)
    
    if len(purity_lis) > 0:
        assert len(purity_lis) == sample_num
        tumor_fractions = purity_lis
    else:
        # Fisrt clustering
        var_matrix, ref_matrix, id_lis = SNV_matrices(tsvs)
        r, c = myclone_fit(var_matrix, ref_matrix, K=min(10, len(id_lis)), max_iterations=1000, min_iterations=100, threshold=0.02)
        r_square, r_assign = r_assign_matrix(r)
        clone_lis = get_clone_lis(r_assign, id_lis, tsvs)

        if len(id_lis) > 100:
            tumor_fractions = predict_tumor_fractions_cln(tsvs, clone_lis)
        else:
            tumor_fractions = predict_tumor_fractions(tsvs, clone_lis, cnv_threshold)
    print("Tumor fractions: ", tumor_fractions)

    update_tsvs(tsvs, tumor_fractions)
    new_tsvs = [f'tmp/new-0{i+1}.tsv' for i in range(sample_num)]
    new_var_matrix, new_ref_matrix, id_lis = SNV_matrices(new_tsvs)
    new_r, new_c = myclone_fit(new_var_matrix, new_ref_matrix, K=min(100, len(id_lis)), max_iterations=1000, min_iterations=100, threshold=0.03)

    new_r_square, new_r_assign = r_assign_matrix(new_r)
    new_clone_lis = get_clone_lis(new_r_assign, id_lis, new_tsvs)
    final_clone_lis = match_cnved_clones(new_clone_lis, new_tsvs, threshold=cnv_threshold)

    output_clones(final_clone_lis, output_dir)


if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    parser = argparse.ArgumentParser(description='Process multiple tsv files and save the output to a directory.')
    parser.add_argument('-i', '--input', type=str, nargs='+', help='Input file paths.', required=True)
    parser.add_argument('-o', '--output', type=str, help='Output directory path.', required=True)
    parser.add_argument('--tumor_purity', action='store_true', help='Use provided tumor purity data instead of automatic inference.')

    args = parser.parse_args()
    tsvs = args.input
    out_folder = args.output
    use_tumor_purity = args.tumor_purity

    if use_tumor_purity:
        purity_lis = read_purity_values(tsvs)
    else:
        purity_lis = []

    max_retries = 10 
    for attempt in range(max_retries):
        try:
            run_myclone(tsvs, out_folder, purity_lis=purity_lis)
            break
        except Exception as e:
            if attempt < max_retries - 1:  
                print(f"Try {attempt + 1}/{max_retries} Fail: {e}")
            else:
                raise  
    
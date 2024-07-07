import pandas as pd
import numpy as np
# from calculate import cal_expected_cn
import os
import math
import json
import ast
import time

class Cluster:
    def __init__(self):
        self.mutations = []
        self.vafs = [] # every element is S-length list
        self.vaf = [] # caled by alpha&beta in fitting
    
    def avg_vaf(self):
        n = len(self.mutations)
        S = len(self.vafs[0])
        avg_vaf = []
        for s in range(S):
            vafs_s = [vaf[s] for vaf in self.vafs]
            avg_vaf_s = sum(vafs_s) / n
            avg_vaf.append(avg_vaf_s)
        return avg_vaf

def timed(func, *args, **kwargs):
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    print(f"{func.__name__} Running Time: {end - start} s")
    return result

def readtsv(tsv_file):
    df = pd.read_csv(tsv_file, delimiter='\t')
    return df

def output_tsv(df, path):
    create_file_dir(path)
    df.to_csv(path, sep='\t', index=False)    

def readjson(json_file):
    with open(json_file, 'r', encoding='utf-8') as fin:
        data = json.load(fin)
    return data

def predict_time(s_time, i, N):
    c_time = time.time()
    c_ela = c_time - s_time
    f_c_ela = time.strftime("%H:%M:%S", time.gmtime(c_ela))
    c_total = c_ela / i * N
    f_c_total = time.strftime("%H:%M:%S", time.gmtime(c_total))
    f_c_avg = time.strftime("%H:%M:%S", time.gmtime(c_ela/i))
    print("-----------------------------------------------------")
    print("Index {} complete".format(i))
    print("Time: {}, Total Time: {}".format(f_c_ela, f_c_total))
    print("Avg Time: {}".format(f_c_avg))
    print("-----------------------------------------------------")

def create_file_dir(path):
    out_dir = os.path.dirname(path)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

def create_dir(path):
    out_dir = path
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

def record_time(s_time, id, out_file):
    folder_path = os.path.dirname(out_file)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    n_time = time.time()
    ela = n_time - s_time
    if id == 1:
        with open(out_file, 'w') as fout:
            fout.write(f"[Sample {id}] {ela}\n")
    else:
        with open(out_file, 'a') as fout:
            fout.write(f"[Sample {id}] {ela}\n")

##############################################################

def SNV_matrices(tsv_files):
    S = len(tsv_files)
    dfs = []
    for tsv_file in tsv_files:
        df = readtsv(tsv_file)
        dfs.append(df)
    
    # get mutation_id set
    mutation_ids = list(dfs[0]['mutation_id'])
    for df in dfs[1:]:
        assert mutation_ids == list(df['mutation_id'])
    N = len(mutation_ids)
    
    # cal var_reads & ref_counts matrix
    var_matrix = np.zeros((N, S), dtype=int)
    ref_matrix = np.zeros((N, S), dtype=int)
    for sample_idx, df in enumerate(dfs):
        for mutation_idx, mutation_id in enumerate(mutation_ids):
            row = df[df['mutation_id'] == mutation_id]
            if not row.empty:
                var_counts = int(row['var_counts'].values[0])
                ref_counts = int(row['ref_counts'].values[0])
                var_matrix[mutation_idx, sample_idx] = var_counts
                ref_matrix[mutation_idx, sample_idx] = ref_counts

    return var_matrix, ref_matrix, mutation_ids

def cln_SNV_matrices(tsvs):
    S = len(tsvs)
    dfs = []
    for tsv_file in tsvs:
        df = readtsv(tsv_file)
        dfs.append(df)
    
    # get mutation_id set
    mutation_ids = list(dfs[0]['mutation_id'])
    for df in dfs[1:]:
        assert mutation_ids == list(df['mutation_id'])

    # get clean mutation_id set
    cln_mutation_ids = dfs[0][dfs[0]['total_cn'] == 2]['mutation_id'].tolist()
    for df in dfs[1:]:
        tmp_mutation_ids = df[df['total_cn'] == 2]['mutation_id'].tolist()
        cln_mutation_ids = [id for id in cln_mutation_ids if id in tmp_mutation_ids]
    N = len(cln_mutation_ids)
    
    # cal var_reads & ref_counts matrix
    var_matrix = np.zeros((N, S), dtype=int)
    ref_matrix = np.zeros((N, S), dtype=int)
    for sample_idx, df in enumerate(dfs):
        for mutation_order, mutation_id in enumerate(cln_mutation_ids):
            row = df[df['mutation_id'] == mutation_id]
            if not row.empty:
                var_counts = int(row['var_counts'].values[0])
                ref_counts = int(row['ref_counts'].values[0])
                var_matrix[mutation_order, sample_idx] = var_counts
                ref_matrix[mutation_order, sample_idx] = ref_counts

    cnved_mut_ids = [mut_id for mut_id in mutation_ids if mut_id not in cln_mutation_ids]
    return var_matrix, ref_matrix, cln_mutation_ids, cnved_mut_ids

def cluster_lis_old(r_matrix, id_lis, tsvs):
    S = len(tsvs)
    dfs = []
    for tsv_file in tsvs:
        df = readtsv(tsv_file)
        dfs.append(df)
    
    clusters = []
    used_indices = []
    for row in r_matrix:
        row = list(row)
        one_indices = [index for index, e in enumerate(row) if e == 1]

        if one_indices[0] not in used_indices:
            new_clus = Cluster()
            for index in one_indices:
                mut_id = id_lis[index]
                new_clus.mutations.append(mut_id)
                used_indices.append(index)

                vaf_lis = []
                for s in range(S):
                    row = dfs[s][dfs[s]['mutation_id'] == mut_id]
                    vaf_s = float(row['VAF'].values[0])
                    vaf_lis.append(vaf_s)
                new_clus.vafs.append(vaf_lis)
            clusters.append(new_clus)

    return clusters

def cluster_lis(r_matrix, id_lis, c, tsvs):
    S = c.shape[1]
    dfs = []
    for tsv_file in tsvs:
        df = readtsv(tsv_file)
        dfs.append(df)
    
    clusters = []
    r_matrix_t = r_matrix.T
    for clus_index, r_row in enumerate(r_matrix_t):
        r_row = list(r_row)
        one_indices = [index for index, e in enumerate(r_row) if e == 1]

        new_clus = Cluster()
        for index in one_indices:
            mut_id = id_lis[index]
            new_clus.mutations.append(mut_id)

            vaf_lis = []
            for s in range(S):
                mut_row = dfs[s][dfs[s]['mutation_id'] == mut_id]
                var_c_s = int(mut_row['var_counts'].values[0])
                ref_c_s = int(mut_row['ref_counts'].values[0])
                vaf_s = var_c_s / (var_c_s + ref_c_s)
                vaf_lis.append(vaf_s)
            new_clus.vafs.append(vaf_lis)
        
        new_clus.vaf = list(c[clus_index])
        clusters.append(new_clus)

    return clusters

# def assign_trunk_cnved_mutations(cnved_mut_ids, clus_lis, tsvs):
#     S = len(tsvs)
#     dfs = []
#     for tsv_file in tsvs:
#         df = readtsv(tsv_file)
#         dfs.append(df)
#     ori_clus_num = len(clus_lis)
    
#     cn_diff_threshold = 0.2
#     for mut_id in cnved_mut_ids:
#         clus_assigned = False
#         for clus in clus_lis[:ori_clus_num]:
#             clus_vaf = clus.vaf
#             mut_cn = ref_cn = 1
#             cn_diffs = []
#             not_this_clus = False
#             for s in range(S):
#                 mut_row = dfs[s][dfs[s]['mutation_id'] == mut_id]
#                 var_c_s = int(mut_row['var_counts'].values[0])
#                 ref_c_s = int(mut_row['ref_counts'].values[0])
#                 mut_vaf_s = var_c_s / (var_c_s + ref_c_s)
#                 total_cn_s = float(mut_row['total_cn'].values[0])

#                 if (mut_vaf_s != 0 and clus_vaf[s] == 0) or (mut_vaf_s == 0 and clus_vaf[s] != 0):
#                     not_this_clus = True
#                     break
#                 if mut_vaf_s == 0 and clus_vaf[s] == 0:
#                     continue
                
#                 expected_cn_s, mut_cn, ref_cn = cal_expected_cn(mut_vaf_s, clus_vaf[s], mut_cn, ref_cn, total_cn_s, mut_id)
#                 # if mut_id == 106496896:
#                 #     print(clus.mutations)
#                 #     print(s)
#                 #     print(expected_cn_s, mut_cn, ref_cn)
#                 #     print()
#                 cn_diffs.append(abs(expected_cn_s - total_cn_s))
            
#             # judge if expected copy number close to true copy number
#             if not_this_clus: continue
#             nonzero_n = len(cn_diffs)
#             if np.linalg.norm(np.array(cn_diffs)) < nonzero_n * cn_diff_threshold:
#                 clus_assigned = True
#                 clus.mutations.append(mut_id)
#                 vaf_lis = []
#                 for s in range(S):
#                     mut_row = dfs[s][dfs[s]['mutation_id'] == mut_id]
#                     var_c_s = int(mut_row['var_counts'].values[0])
#                     ref_c_s = int(mut_row['ref_counts'].values[0])
#                     vaf_s = var_c_s / (var_c_s + ref_c_s)
#                     vaf_lis.append(vaf_s)
#                 clus.vafs.append(vaf_lis)
#                 break
        
#         if not clus_assigned:
#             new_clus = Cluster()
#             new_clus.mutations.append(mut_id)
#             vaf_lis = []
#             for s in range(S):
#                 mut_row = dfs[s][dfs[s]['mutation_id'] == mut_id]
#                 var_c_s = int(mut_row['var_counts'].values[0])
#                 ref_c_s = int(mut_row['ref_counts'].values[0])
#                 vaf_s = var_c_s / (var_c_s + ref_c_s)
#                 vaf_lis.append(vaf_s)
#             new_clus.vafs.append(vaf_lis)
#             clus_lis.append(new_clus)
    
#     return clus_lis

def get_sciclone_vaf_dat(file_path):
    data = pd.read_csv(file_path, sep="\t")
    file_name = file_path.split('/')[-1].rstrip('.tsv')

    converted_data = pd.DataFrame({
        "chr": data["chr"],
        "pos": data["position"],
        "ref_reads": data["ref_counts"],
        "var_reads": data["var_counts"],
        "vaf": data["VAF"] * 100
    })

    folder_path = os.path.dirname(file_path)
    # folder_name = os.path.basename(folder_path)
    output_dir = f'{folder_path}/sciclone'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = f'{output_dir}/{file_name}.dat'
    converted_data.to_csv(output_file, sep="\t", index=False, header=False)

def get_phylowgs_ssm_file(tsvs):
    dfs = []
    for tsv in tsvs:
        df = readtsv(tsv)
        dfs.append(df)
    row_num = dfs[0].shape[0]

    folder_path = os.path.dirname(tsvs[0])
    output_dir = f'{folder_path}/phylowgs'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = f'{output_dir}/ssm_data.txt'

    with open(output_file, "w") as out:
        out.write("id\tgene\ta\td\tmu_r\tmu_v\n")
        df_1 = dfs[0]

        for i in range(row_num):
            mut_id = df_1['mutation_id'].iloc[i]
            out.write(f"s{i}\t{mut_id}")
            ref_counts = []
            total_counts = []
            for df in dfs:
                ref_counts.append(str(df['ref_counts'].iloc[i]))
                total_counts.append(str(df['var_counts'].iloc[i] + df['ref_counts'].iloc[i]))
            ref_counts_str = ','.join(ref_counts)
            total_counts_str = ','.join(total_counts)
            out.write(f"\t{ref_counts_str}\t{total_counts_str}\t0.999\t0.5\n")

def get_phylowgs_cnv_file(tsvs):
    dfs = []
    for tsv in tsvs:
        df = readtsv(tsv)
        dfs.append(df)
    row_num = dfs[0].shape[0]

    folder_path = os.path.dirname(tsvs[0])
    output_dir = f'{folder_path}/phylowgs'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    output_file = f'{output_dir}/cnv_data.txt'

    with open(output_file, "w") as out:
        out.write("cnv\ta\td\tssms\tphysical_cnvs\n")
        cnv_index = -1
        for i in range(row_num):
            total_cn = 2
            for df in dfs:
                cn = df['total_cn'].iloc[i]
                if cn != 2:
                    total_cn = cn
                    break
            if total_cn != 2:
                cnv_index += 1
                chr = df['chr'].iloc[i]
                pos = df['position'].iloc[i]
                if total_cn < 2:
                    major_cn = 1
                    minor_cn = 0
                    cell_prev = 2 - total_cn
                elif total_cn > 2:
                    major_cn = math.ceil(total_cn) - 1
                    minor_cn = 1
                    cell_prev = (total_cn - 2) / (major_cn - 1)
                
                zero_str = ','.join([str(0) for _ in range(len(tsvs))])
                thousand_str = ','.join([str(2000) for _ in range(len(tsvs))])
                out.write(f"c{cnv_index}\t{zero_str}\t{thousand_str}\ts{i},{minor_cn},{major_cn}\tchrom={chr},start={pos-1},end={pos+1},major_cn={major_cn},minor_cn={minor_cn},cell_prev={cell_prev:.3f}\n")

def get_pyclone_file(tsvs):
    dfs = []
    tsv_names = []
    for tsv in tsvs:
        df = readtsv(tsv)
        dfs.append(df)
        tsv_names.append(tsv.split('/')[-1].split('.')[0])
    
    folder_path = os.path.dirname(tsvs[0])
    output_dir = f'{folder_path}/pyclone-vi'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    new_dfs = []
    for i, df in enumerate(dfs):
        new_df = df.copy()
        sample_id = tsv_names[i].split('_')[0]
        new_df['sample_id'] = sample_id
        new_df = new_df.rename(columns={'var_counts':'alt_counts'})
        new_df = new_df[['mutation_id', 'sample_id', 'ref_counts', 'alt_counts', 'major_cn', 'minor_cn', 'normal_cn']].copy()
        new_dfs.append(new_df)
    
    final_df = pd.concat(new_dfs, axis=0)
    output_file = f'{output_dir}/sb_somatic.tsv'
    final_df.to_csv(output_file, sep="\t", index=False)

def get_fastclone_file(tsvs):
    dfs = []
    tsv_names = []
    for tsv in tsvs:
        df = readtsv(tsv)
        dfs.append(df)
        tsv_names.append(tsv.split('/')[-1].split('.')[0])
    
    folder_path = os.path.dirname(tsvs[0])
    output_dir = f'{folder_path}/fastclone'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    new_dfs = []
    for i, df in enumerate(dfs):
        new_df = df.copy()
        new_df = new_df.rename(columns={'mutation_id':'true_mut_id'})
        new_df['mutation_id'] = 0
        # sample_id = tsv_names[i].split('_')[0]
        # new_df['sample_id'] = sample_id
        new_df = new_df[['mutation_id', 'ref_counts', 'var_counts', 'major_cn', 'minor_cn', 'normal_cn', 'true_mut_id']].copy()
        new_dfs.append(new_df)
    
    final_df = pd.concat(new_dfs, axis=0)
    for i in range(final_df.shape[0]):
        final_df['mutation_id'].loc[i] = f'A{i+1}'
    output_file = f'{output_dir}/sb_somatic.tsv'
    final_df.to_csv(output_file, sep="\t", index=False)

def change_sciclone_results(s_df_path, ori_df_path):
    s_df = readtsv(s_df_path)
    ori_df = readtsv(ori_df_path)
    new_df_data = {
        "mutation_id": [],
        "sciclone_clone_id": []
    }
    new_df = pd.DataFrame(new_df_data)

    for index, row in ori_df.iterrows():
        chr = row['chr']
        pos = row['position']
        condition = (s_df['chr'] == chr) & (s_df['st'] == pos)
        filtered_rows = s_df[condition]

        if not filtered_rows.empty:
            s_row = filtered_rows.iloc[0]
            clone_id = s_row['cluster']
            if pd.isna(clone_id):
                clone_id = -1
        else:
            clone_id = -1
        
        mut_id = row['mutation_id']
        new_df = new_df.append({'mutation_id': f"{int(mut_id)}", 'sciclone_clone_id': f"{int(clone_id)}"}, ignore_index=True)

    return new_df

def change_phylowgs_results(phylo_folder, ori_df_path):
    summ_json_path = os.path.join(phylo_folder, "example_data.summ.json")
    data = readjson(summ_json_path)
    ori_df = readtsv(ori_df_path)
    new_df_data = {
        "mutation_id": [],
        "phylowgs_clone_id": []
    }
    new_df = pd.DataFrame(new_df_data)

    max_llh_idx = max(data['trees'], key=lambda x: data['trees'][x]['llh'])
    tree_json_path = os.path.join(phylo_folder, f"trees/{max_llh_idx}.json")
    tree_data = readjson(tree_json_path)

    for index, row in ori_df.iterrows():
        clone_id = -1
        for cluster_idx, cluster_data in tree_data['mut_assignments'].items():
            if f"s{index}" in cluster_data['ssms']:
                clone_id = cluster_idx
                break
        
        mut_id = row['mutation_id']
        new_df = new_df.append({'mutation_id': f"{int(mut_id)}", 'phylowgs_clone_id': f"{int(clone_id)}"}, ignore_index=True)

    return new_df

def change_pyclone_vi_results(s_df_path, ori_df_path):
    s_df = readtsv(s_df_path)
    ori_df = readtsv(ori_df_path)
    new_df_data = {
        "mutation_id": [],
        "pyclone-vi_clone_id": []
    }
    new_df = pd.DataFrame(new_df_data)

    for index, row in ori_df.iterrows():
        mut_id = row['mutation_id']
        clone_id = s_df[s_df['mutation_id'] == mut_id]['cluster_id'].values[0]

        new_df = new_df.append({'mutation_id': f"{int(mut_id)}", 'pyclone-vi_clone_id': f"{int(clone_id)}"}, ignore_index=True)

    return new_df

def change_pyclone_vi_results_real(s_df_path, ori_df_path):
    s_df = readtsv(s_df_path)
    ori_df = readtsv(ori_df_path)
    new_df_data = {
        "mutation_id": [],
        "pyclone-vi_clone_id": [],
        "pyclone-vi_ccfs": []
    }
    new_df = pd.DataFrame(new_df_data)

    for index, row in ori_df.iterrows():
        mut_id = row['mutation_id']
        clone_id = s_df[s_df['mutation_id'] == mut_id]['cluster_id'].values[0]
        ccfs = s_df[s_df['mutation_id'] == mut_id]['cellular_prevalence'].values
        ccfs_str = ','.join([str(ccf) for ccf in ccfs])
        new_df = new_df.append({'mutation_id': f"{mut_id}", 'pyclone-vi_clone_id': f"{int(clone_id)}", 'pyclone-vi_ccfs': ccfs_str}, ignore_index=True)

    return new_df

def change_fastclone_results(r_folder, data_path, ori_df_path):
    ori_df = readtsv(ori_df_path)
    r_path = os.path.join(r_folder, 'scores.csv')
    r_df = pd.read_csv(r_path)
    data_df = readtsv(data_path)
    new_df_data = {
        "mutation_id": [],
        "fastclone_clone_id": []
    }
    new_df = pd.DataFrame(new_df_data)

    r_df = r_df.set_index(r_df.columns[0])
    max_columns = {}
    for index, row in r_df.iterrows():
        max_col_index = row.idxmax()
        max_columns[index] = max_col_index

    for index, row in ori_df.iterrows():
        true_mut_id = row['mutation_id']
        mut_id = data_df[data_df['true_mut_id'] == true_mut_id]['mutation_id'].values[0]
        if mut_id in max_columns.keys():
            clone_id = max_columns[mut_id]
        else:
            clone_id = -1

        new_df = new_df.append({'mutation_id': f"{int(true_mut_id)}", 'fastclone_clone_id': f"{int(clone_id)}"}, ignore_index=True)

    return new_df

def change_cfdna_wgs_results_old(s_txt_path, ori_df_path):
    with open(s_txt_path) as fin:
        contents = fin.readlines()
    binary_list = [int(ch) for ch in contents[0] if ch in '01']
    cluster_list = ast.literal_eval(contents[1])

    ori_df = readtsv(ori_df_path)
    mut_ids = ori_df['mutation_id'].tolist()
    print(len(mut_ids))
    print(len(binary_list))
    assert len(mut_ids) == len(binary_list)
    clused_mut_ids = [mut_ids[i] for i in range(len(mut_ids)) if binary_list[i] == 1]
    assert len(clused_mut_ids) == len(cluster_list)
    not_clused_mut_ids = [mut_ids[i] for i in range(len(mut_ids)) if binary_list[i] == 0]

    new_df_data = {
        "mutation_id": [],
        "cfdna-wgs_clone_id": []
    }
    new_df = pd.DataFrame(new_df_data)

    for cluster_id, mut_id in zip(cluster_list, clused_mut_ids):
        clone_id = cluster_id if (cluster_id > 0) else -1
        new_df = new_df.append({'mutation_id': f"{int(mut_id)}", 'cfdna-wgs_clone_id': f"{int(clone_id)}"}, ignore_index=True)

    for mut_id in not_clused_mut_ids:
        new_df = new_df.append({'mutation_id': f"{int(mut_id)}", 'cfdna-wgs_clone_id': "-1"}, ignore_index=True)
    
    return new_df

def change_cfdna_wgs_results(cfdna_wgs_dir, ori_df_path):
    s_txt_path = os.path.join(cfdna_wgs_dir, "sb_cluster.txt")
    with open(s_txt_path) as fin:
        contents = fin.readlines()
    binary_list = [int(ch) for ch in contents[0] if ch in '01']
    cluster_list = ast.literal_eval(contents[1])

    j = 0
    final_cluster_lis = []
    for i in range(len(binary_list)):
        if binary_list[i] == 0:
            final_cluster_lis.append(-1)
        else:
            if cluster_list[j] == 0:
                final_cluster_lis.append(-1)
            else:
                final_cluster_lis.append(cluster_list[j])
            j += 1
    assert j == len(cluster_list)

    somatic_vcf_path = os.path.join(cfdna_wgs_dir, "sb_somatic.vcf")
    s_df = readtsv(somatic_vcf_path)
    ori_df = readtsv(ori_df_path)

    new_df_data = {
        "mutation_id": [],
        "cfdna-wgs_clone_id": []
    }
    new_df = pd.DataFrame(new_df_data)

    for index, row in ori_df.iterrows():
        chr = row['chr']
        pos = row['position']
        condition = (s_df['CHROM'] == f"chr{int(chr)}") & (s_df['POSITION'] == pos)
        filtered_rows = s_df[condition]

        if not filtered_rows.empty:
            s_row_index = filtered_rows.index.tolist()[0]
            clone_id = final_cluster_lis[s_row_index]
        else:
            clone_id = -1
        
        mut_id = row['mutation_id']
        new_df = new_df.append({'mutation_id': f"{int(mut_id)}", 'cfdna-wgs_clone_id': f"{int(clone_id)}"}, ignore_index=True)
    
    return new_df

def build_descendants_dict(tree):
    descendants_dict = {}
    
    def dfs(node, descendants):
        if node not in tree:
            return
        for child in tree[node]:
            descendants.append(child)
            dfs(child, descendants)
            
    for parent, children in tree.items():
        descendants = []
        for child in children:
            descendants.append(child)
            dfs(child, descendants)
        descendants_dict[parent] = descendants
        
    return descendants_dict

def change_phylowgs_ccf_results(phylo_folder, ori_df_path):
    summ_json_path = os.path.join(phylo_folder, "example_data.summ.json")
    data = readjson(summ_json_path)
    ori_df = readtsv(ori_df_path)
    new_df_data = {
        "mutation_id": [],
        "phylowgs_ccfs": []
    }
    new_df = pd.DataFrame(new_df_data)

    max_llh_idx = max(data['trees'], key=lambda x: data['trees'][x]['llh'])
    # print(max_llh_idx)
    tree_json_path = os.path.join(phylo_folder, f"trees/{max_llh_idx}.json")
    tree_data = readjson(tree_json_path)

    tree_total_data = data['trees'][f"{max_llh_idx}"]
    pops = tree_total_data['populations']
    sample_num = len(pops['0']['cellular_prevalence'])
    pop_ccfs = [[] for _ in range(sample_num)]
    for key, value in pops.items():
        if key == '0': continue
        for index, ccf in enumerate(value['cellular_prevalence']):
            pop_ccfs[index].append(ccf)
    
    clone_tree = tree_total_data['structure']
    clone_tree = {int(k):v for k,v in clone_tree.items()}
    descendants_dict = build_descendants_dict(clone_tree)
    node_num = len(pop_ccfs[0])
    for node in range(1, node_num+1):
        if node not in descendants_dict:
            descendants_dict[node] = [node]
        else:
            descendants_dict[node].append(node)

    for index, row in ori_df.iterrows():
        clone_id = -1
        for cluster_idx, cluster_data in tree_data['mut_assignments'].items():
            if f"s{index}" in cluster_data['ssms']:
                clone_id = cluster_idx
                break
        
        descendants = descendants_dict[int(clone_id)]
        ccfs = []
        for i in range(sample_num):
            ccf = sum([pop_ccfs[i][clone_index-1] for clone_index in descendants])
            ccfs.append(ccf)
        ccfs_str = ','.join([str(ccf) for ccf in ccfs])

        mut_id = row['mutation_id']
        new_df = new_df.append({'mutation_id': f"{int(mut_id)}", 'phylowgs_ccfs': ccfs_str}, ignore_index=True)

    return new_df

def change_pyclone_vi_ccf_results(s_df_path, ori_df_path):
    s_df = readtsv(s_df_path)
    ori_df = readtsv(ori_df_path)
    new_df_data = {
        "mutation_id": [],
        "pyclone-vi_ccfs": []
    }
    new_df = pd.DataFrame(new_df_data)

    for index, row in ori_df.iterrows():
        mut_id = row['mutation_id']
        ccfs = s_df[s_df['mutation_id'] == mut_id]['cellular_prevalence'].values
        ccfs_str = ','.join([str(ccf) for ccf in ccfs])
        new_df = new_df.append({'mutation_id': f"{int(mut_id)}", 'pyclone-vi_ccfs': ccfs_str}, ignore_index=True)

    return new_df

def change_cfdna_wgs_ccf_results(cfdna_wgs_dir, ori_df_paths):
    s_txt_path = os.path.join(cfdna_wgs_dir, "sb_cluster.txt")
    with open(s_txt_path) as fin:
        contents = fin.readlines()
    binary_list = [int(ch) for ch in contents[0] if ch in '01']
    cluster_list = ast.literal_eval(contents[1])

    ccf_df_path = os.path.join(cfdna_wgs_dir, "sb_populations.tsv")
    ccf_df = readtsv(ccf_df_path)

    j = 0
    final_cluster_lis = []
    for i in range(len(binary_list)):
        if binary_list[i] == 0:
            final_cluster_lis.append(-1)
        else:
            if cluster_list[j] == 0:
                final_cluster_lis.append(-1)
            else:
                final_cluster_lis.append(cluster_list[j])
            j += 1
    assert j == len(cluster_list)

    somatic_vcf_path = os.path.join(cfdna_wgs_dir, "sb_somatic.vcf")
    s_df = readtsv(somatic_vcf_path)
    ori_dfs = []
    for df_path in ori_df_paths:
        ori_df = readtsv(df_path)
        ori_dfs.append(ori_df)
    ori_df = ori_dfs[0]

    new_df_data = {
        "mutation_id": [],
        "cfdna-wgs_ccfs": []
    }
    new_df = pd.DataFrame(new_df_data)

    for index, row in ori_df.iterrows():
        chr = row['chr']
        pos = row['position']
        mut_id = row['mutation_id']
        condition = (s_df['CHROM'] == f"chr{int(chr)}") & (s_df['POSITION'] == pos)
        filtered_rows = s_df[condition]

        if not filtered_rows.empty:
            s_row_index = filtered_rows.index.tolist()[0]
            clone_id = final_cluster_lis[s_row_index]
        else:
            clone_id = -1
        
        if clone_id > 0:
            ccf_rows = ccf_df[ccf_df['Clone'] == clone_id]
            ccfs = ccf_rows['CCF'].tolist()
            ccfs_str = ','.join([str(c) for c in ccfs])
        else:
            ccfs = []
            for df_i in ori_dfs:
                vaf = df_i[df_i['mutation_id'] == mut_id]['VAF'].values[0]
                ccfs.append(vaf*2)
            ccfs_str = ','.join([str(c) for c in ccfs])
        
        new_df = new_df.append({'mutation_id': f"{int(mut_id)}", 'cfdna-wgs_ccfs': ccfs_str}, ignore_index=True)
    
    return new_df

def change_fastclone_ccf_results(r_folder, data_path, ori_df_path):
    ori_df = readtsv(ori_df_path)
    r_path = os.path.join(r_folder, 'scores.csv')
    r_df = pd.read_csv(r_path)
    r_clone_path = os.path.join(r_folder, 'subclones.csv')
    r_clone_df = pd.read_csv(r_clone_path)
    data_df = readtsv(data_path)
    new_df_data = {
        "mutation_id": [],
        "fastclone_ccfs": []
    }
    new_df = pd.DataFrame(new_df_data)

    r_df = r_df.set_index(r_df.columns[0])
    max_columns = {}
    for index, row in r_df.iterrows():
        max_col_index = row.idxmax()
        max_columns[index] = max_col_index

    r_clone_df = r_clone_df.set_index(r_clone_df.columns[0])
    ccf_columns = {}
    for index, row in r_clone_df.iterrows():
        ccf = row['prop']
        ccf_columns[index] = ccf

    for index, row in ori_df.iterrows():
        true_mut_id = row['mutation_id']
        mut_id = data_df[data_df['true_mut_id'] == true_mut_id]['mutation_id'].values[0]
        if mut_id in max_columns.keys():
            clone_id = max_columns[mut_id]
            ccf = ccf_columns[int(clone_id)]
        else:
            clone_id = -1
            ccf = 0
        
        ccfs_str = str(ccf)
        new_df = new_df.append({'mutation_id': f"{int(true_mut_id)}", 'fastclone_ccfs': ccfs_str}, ignore_index=True)

    return new_df
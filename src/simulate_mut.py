import random
from copy import deepcopy
import numpy as np
from scipy.stats import binom, poisson
import pandas as pd
import os
import math

from copynum import cn_to_logratio, expected_hsaf
from methods import readtsv, output_tsv, get_sciclone_vaf_dat, get_phylowgs_ssm_file, get_phylowgs_cnv_file, get_pyclone_file, get_fastclone_file
from calculate import random_partition, random_N_distribution

class Mutation:
    def __init__(self, id, maternal, macn, total_cn, pop_id):
        self.id = id
        self.maternal = maternal  # Bool, Present on maternal or paternal allele?
        self.macn = macn          # Mutant allele copy number
        self.total_cn = total_cn
        self.pop_id = pop_id

class Block: # length is 1_000_000
    def __init__(self, chr, start_pos, cn_paternal, cn_maternal, arm, mutations):
        self.chr = chr
        self.start_pos = start_pos
        self.cn_paternal = cn_paternal
        self.cn_maternal = cn_maternal
        self.arm = arm  # 0 = centromere, 1 = p-arm, 2 = q-arm
        self.mutations = mutations

class Population:
    def __init__(self, parent, blocks, id):
        self.parent = parent
        self.blocks = blocks
        self.id = id

def rand_allele(block):
    if block.cn_maternal + block.cn_paternal == 0:
        return random.random() < 0.5
    return random.random() < block.cn_maternal / (block.cn_maternal + block.cn_paternal)

def duplicate_genome(pop):
    for block in pop.blocks:
        block.cn_paternal *= 2
        block.cn_maternal *= 2
        for mut in block.mutations:
            mut.macn *= 2

def mutate_cnv(pop, mut_ids):
    while True:
        # Generate a CNA
        change = -1 if random.random() < 0.5 else 1  # 50/50 chance of gain/deletion
        chr = random.randint(1, 22)  # Affected chromosome

        if random.random() < 0.5:  # Whole chromosome CNA
            blocks = [b for b in pop.blocks if b.chr == chr and b.arm != 0]
        else:
            # Chromosome arm CNA
            arm = random.randint(1, 2)
            blocks = [b for b in pop.blocks if b.chr == chr and b.arm == arm]
            if blocks and random.random() < 0.5:
                # Simulate a CNA that only affects a small region
                if random.random() < 0.5:
                    blocks = blocks[:random.randint(1, len(blocks))]
                else:
                    blocks = blocks[random.randint(0, len(blocks)-1):]
        
        if not blocks: continue

        # Randomize which parental allele is affected
        maternal = rand_allele(blocks[0])

        # Make large homozygous deletions impossible
        if change < 0 and sum(1 for b in blocks if b.cn_maternal + b.cn_paternal <= 1) > 10: continue

        # Do not modify zero copy regions
        blocks = [b for b in blocks if (maternal and b.cn_maternal > 0) or (not maternal and b.cn_paternal > 0)]

        if not blocks: continue

        for block in blocks:
            if maternal:
                for mut in block.mutations:
                    if mut.maternal and random.random() < mut.macn / block.cn_maternal:
                        mut.macn += change
                    mut.total_cn += change
                block.cn_maternal += change
            else:
                for mut in block.mutations:
                    if not mut.maternal and random.random() < mut.macn / block.cn_paternal:
                        mut.macn += change
                    mut.total_cn += change
                block.cn_paternal += change

        break   # Successfully mutated the genome

def mutate(pop, mut_ids):
    while True:
        # Generate a point mutation
        # Randomize the genomic block within which the mutation occurs.
        valid_blocks = [b for b in pop.blocks if b.arm != 0]  # Avoid centromeres
        total = sum(b.cn_paternal + b.cn_maternal for b in valid_blocks)
        # Calculate probabilities for each block
        block_probs = [(b.cn_paternal + b.cn_maternal) / total for b in valid_blocks]
        # Randomly select a block based on probabilities
        selected_block = random.choices(valid_blocks, weights=block_probs)[0]
        # Create a new mutation with MACN of 1 and append it to the selected block's mutations
        while True:
            mutation_id = random.randint(0, 999_999_999)
            if mutation_id not in mut_ids:
                mut_ids.append(mutation_id)
                break
        mutation = Mutation(mutation_id, rand_allele(selected_block), 1, selected_block.cn_paternal+selected_block.cn_maternal, pop_id=pop.id)
        selected_block.mutations.append(mutation)

        break   # Successfully mutated the genome

def wgs(num_samples, output_folder, depth, mut_N, clone_num, logratio_stdev=0.7, maf_threshold=0.1, mutant_read_threshold=8):
    # Generate a germline genome
    d = readtsv("../data/hg38.chrom.sizes")
    centromeres = readtsv("../data/hg38.centromere.bed")
    germline = []
    for c in range(1, 23):
        chr_len = int(d.loc[c-1, 'length'])
        centro_start = int(centromeres.loc[c-1, 'start'])
        centro_end = int(centromeres.loc[c-1, 'end'])
        for pos in range(500_000, chr_len+1, 1_000_000):
            if centro_start <= pos <= centro_end:
                arm = 0
            elif pos < centro_start:
                arm = 1
            else:
                arm = 2
            germline.append(Block(c, pos, 1, 1, arm, []))
    
    # Assign sample names
    samples = [f"{s+1:02d}" for s in range(num_samples)]
    S = len(samples)

    # Assign a randomized cancer fraction and diploid level for each sample
    cancer_frac = [random.uniform(0.3, 0.8) for s in range(S)]
    # cancer_frac = [random.uniform(0.8, 1) for s in range(S)]
    diploid_level = random.uniform(-0.5, 0.5)

    # Generate a phylogenetic tree of populations
    mut_ids = []
    mut_N = mut_N
    # First, we generate the truncal branch
    trunk = Population(0, deepcopy(germline), id=1)  # Truncal node
    for k in range(random.randint(1*mut_N, 2*mut_N)):
        mutate(trunk, mut_ids)
    # for k in range(random.randint(6, 10)):
    for k in range(random.randint(2, 3)):
        mutate_cnv(trunk, mut_ids)
    for k in range(random.randint(2*mut_N, 3*mut_N)):
        mutate(trunk, mut_ids)

    # Given clone nums, randomly generate the tree struture 
    P = clone_num
    populations = [trunk]
    for i in range(2, P+1):
        parent = random.randint(1,i-1) # parent 0 represents new trunk
        if parent > 0:
            subclone = Population(parent, deepcopy(populations[parent-1].blocks), id=i)
        else:
            subclone = Population(parent, deepcopy(germline), id=i)
        for k in range(random.randint(2*mut_N, 3*mut_N)):
            mutate(subclone, mut_ids)
            if random.random() < 0.005:
                mutate_cnv(subclone, mut_ids)
        populations.append(subclone)

    assert all(b.cn_paternal >= 0 and b.cn_maternal >= 0 for b in trunk.blocks)

    # Generate population CCFs for each sample
    increments = random_partition(P,S) # every sample gets new clones
    nonzero_ends = [sum(increments[:i+1]) for i in range(S)] # index 1~P
    nonzero_begins = [1] # every sample may lose some clones
    for j in range(1,S):
        nonzero_begins.append(random.randint(nonzero_begins[j-1], nonzero_ends[j-1]+1))
    nonzero_ids_lis = [list(range(nonzero_begins[i], nonzero_ends[i]+1)) for i in range(len(nonzero_ends))]
    # if a clone disappear in one sample, then all of its children have nonzero ccf
    for j in range(1,S):
        for index in range(nonzero_ends[j]+1, P+1):
            if 0 < populations[index-1].parent < nonzero_begins[j]:
                nonzero_ids_lis[j].append(index)
    # cal ccf
    pop_ccf = np.zeros((S,P))
    for s in range(S):
        nonzero_ccfs = random_N_distribution(len(nonzero_ids_lis[s]))
        for j, ccf in zip(nonzero_ids_lis[s], nonzero_ccfs):
            pop_ccf[s,j-1] = ccf
    
    # output_dir_name
    output_dir_index = 1
    while True:
        output_dir_name = os.path.join(output_folder, f"s_data_{output_dir_index}") 
        if os.path.exists(output_dir_name):
            output_dir_index += 1
        else:
            os.makedirs(output_dir_name)
            break

    # Print the ground truth information
    with open(f"{output_dir_name}/clones_info.txt", "w") as out:
        for s in range(S):
            # out.write(f"Diploid level: {diploid_level:.2f}\n")
            out.write(f"Sample {samples[s]} with cancer fraction {cancer_frac[s] * 100:.1f}%:\n")
            for p in range(P):
                out.write(f"- Population {p + 1}: {pop_ccf[s,p] * 100:.1f}% CCF\n")
            out.write('\n')
    
    # # Generate coverage logratios for each sample
    # os.makedirs(f"{output_dir_name}/cfdna-wgs")
    # for s in range(S):
    #     with open(f"{output_dir_name}/cfdna-wgs/{samples[s]}_logratio.igv", "w") as out:
    #         out.write(f"CHROM\tSTART\tEND\tFEATURE\t{samples[s]}\n")
    #         for b, block in enumerate(trunk.blocks):
    #             for pos in range(500_00, 1000_000, 1000_00):
    #                 if block.arm == 0: continue  # Omit centromeres
    #                 total_cn = sum(
    #                     pop_ccf[s,p] * (populations[p].blocks[b].cn_paternal + populations[p].blocks[b].cn_maternal)
    #                     for p in range(P)
    #                 )
    #                 expected_lr = cn_to_logratio(total_cn, cancer_frac[s], 2) + diploid_level
    #                 logratio = np.random.normal(expected_lr, logratio_stdev)
    #                 out.write(f"chr{block.chr}\t{block.start_pos + pos}\t{block.start_pos + pos + 1}\t\t{logratio:.3f}\n")
    
    # # Generate heterozygous SNP allele fractions for each sample
    # for s in range(S):
    #     with open(f"{output_dir_name}/cfdna-wgs/{samples[s]}_hetz_snp.tsv", "w") as out:
    #         out.write("CHROM\tPOSITION\tALT_FRAC\tDEPTH\n")
    #         for b, block in enumerate(trunk.blocks):
    #             if block.arm == 0: continue  # Omit centromeres
                
    #             for pos in range(10_000, 1_000_000, 100_000):
    #                 total_cn = sum(
    #                     pop_ccf[s,p] * (populations[p].blocks[b].cn_paternal + populations[p].blocks[b].cn_maternal)
    #                     for p in range(P)
    #                 )
    #                 cn_paternal = sum(
    #                     pop_ccf[s,p] * populations[p].blocks[b].cn_paternal
    #                     for p in range(P)
    #                 )
    #                 true_hsaf = expected_hsaf(cancer_frac[s], cn_paternal, total_cn)

    #                 cn_adjusted_depth = (cancer_frac[s] * total_cn / 2 * depth) + ((1 - cancer_frac[s]) * depth)
    #                 noisy_depth = np.random.poisson(cn_adjusted_depth)

    #                 rand_values = [np.random.binomial(noisy_depth, true_hsaf) for _ in range(9)]
    #                 median_value = np.median([0.5 + abs(0.5 - rand_val / noisy_depth) for rand_val in rand_values])
    #                 out.write(f"chr{block.chr}\t{block.start_pos + pos}\t{median_value:.5f}\t{depth}\n")
    
    # Generate somatic mutations for each sample
    for s in range(S):
        with open(f"{output_dir_name}/{samples[s]}_somatic.tsv", "w") as out:
            out.write("mutation_id\tvar_counts\tref_counts\ttotal_cn\tVAF\tsubclone_id\tchr\tposition\tnormal_cn\tmajor_cn\tminor_cn\n")
            for b, block in enumerate(trunk.blocks):
                pop_cn = [p.blocks[b].cn_paternal + p.blocks[b].cn_maternal for p in populations]

                uniq_id = []
                for p in range(P):
                    for mut in populations[p].blocks[b].mutations:
                        uniq_id.append(mut.id)
                uniq_id = list(set(uniq_id))

                # block length is 1000_000, position is random on block, same across samples
                positions = [999_999//len(uniq_id)*k for k in range(len(uniq_id))]

                for mut_id, position in zip(uniq_id, positions):
                    pop_macn = np.zeros(P, dtype=int)
                    for p in range(P):
                        for m in populations[p].blocks[b].mutations:
                            if m.id == mut_id:
                                pop_macn[p] = m.macn
                                pop_id = m.pop_id  # get pop_id
                                break
                    
                    # calculate var_counts & ref_counts (ref = total - var)
                    total_cn = np.sum(pop_ccf[s, :] * pop_cn)
                    total_macn = np.sum(pop_ccf[s, :] * pop_macn)

                    cn_adjusted_depth = (cancer_frac[s] * total_cn / 2 * depth) + (1 - cancer_frac[s]) * depth
                    # print(cn_adjusted_depth)
                    noisy_depth = np.random.poisson(cn_adjusted_depth)
                    # noisy_depth = cn_adjusted_depth

                    true_maf = total_macn * cancer_frac[s] / (cancer_frac[s] * total_cn + (1 - cancer_frac[s]) * 2)

                    mut_reads = np.random.binomial(noisy_depth, true_maf)
                    # mut_reads = round(noisy_depth * true_maf)
                    total_reads = noisy_depth

                    var_counts = mut_reads
                    ref_counts = total_reads - mut_reads
                    
                    # # Don't show a mutation if it doesn't pass detection thresholds
                    # if not (mut_reads >= mutant_read_threshold and mut_reads / total_reads >= maf_threshold): continue

                    # generate cn data for Pyclone input
                    normal_cn = 2
                    final_cn = cancer_frac[s] * total_cn + (1 - cancer_frac[s]) * normal_cn
                    # assert round(total_cn) >= 1
                    if round(total_cn) < 1: continue
                    
                    if final_cn < 1.99:
                        major_cn = 1
                        minor_cn = 0
                    elif 1.99 <= final_cn <= 2.01:
                        major_cn = 1
                        minor_cn = 1
                    elif final_cn > 2.01:
                        major_cn = math.ceil(final_cn) - 1
                        minor_cn = 1

                    out.write(f"{mut_id}")
                    out.write(f"\t{var_counts}\t{ref_counts}\t{final_cn:.2f}\t{var_counts/(var_counts+ref_counts)}\t{pop_id}")
                    out.write(f"\t{block.chr}\t{block.start_pos + position}")
                    out.write(f"\t{normal_cn}\t{major_cn}\t{minor_cn}")
                    out.write("\n")
    
    # Delete the mutations with vaf=0 in all samples
    dfs = []
    for s in range(S):
        df = readtsv(f"{output_dir_name}/{samples[s]}_somatic.tsv")
        dfs.append(df)
    
    mut_ids = df['mutation_id'].tolist()
    del_mut_ids = []
    for mut_id in mut_ids:
        vafs = []
        for df in dfs:
            vaf = df[df['mutation_id'] == mut_id]['VAF'].values[0]
            vafs.append(vaf)
        if sum(vafs) == 0:
            del_mut_ids.append(mut_id)
    
    new_dfs = []
    for df in dfs:
        new_df = df.copy()
        for del_mut_id in del_mut_ids:
            new_df = new_df[new_df['mutation_id'] != del_mut_id]
        new_dfs.append(new_df)
    
    for s in range(S):
        new_dfs[s].to_csv(f"{output_dir_name}/{samples[s]}_somatic.tsv", sep='\t', index=False)

    # # Write the ground truth CN state into a file (used for plotting)
    # chromosome_data = readtsv('../data/hg38.chrom.sizes')
    # CHR_NAMES = chromosome_data.loc[:,'chr'].tolist()
    # CHR_SIZES = chromosome_data.loc[:,'length'].astype(int).tolist()
    # CHR_STARTS = [0] + [CHR_STARTS[i - 1] + CHR_SIZES[i - 1] for i in range(1, len(CHR_SIZES))]

    # Generate a text file of subclonal CN profiles
    with open(f"{output_dir_name}/cn_profiles.tsv", "w") as out:
        out.write("CHROM\tSTART\tEND\t")
        for i in range(P):
            out.write(f"Pop{i+1}\t")
        out.write("\n")

        blocks = populations[0].blocks
        start_pos = blocks[0].start_pos
        chr = blocks[0].chr
        cn = [(populations[p].blocks[0].cn_paternal, populations[p].blocks[0].cn_maternal) for p in range(P)]

        for k in range(1, len(blocks)):
            new_cn = [(populations[p].blocks[k].cn_paternal, populations[p].blocks[k].cn_maternal) for p in range(P)]
            
            if blocks[k].arm == 0 or blocks[k].chr != chr or new_cn != cn:
                if blocks[k-1].arm == 0: continue
                
                end_pos = blocks[k-1].start_pos
                out.write(f"chr{chr}\t{start_pos}\t{end_pos}")
                for p in range(P):
                    out.write(f"\t{cn[p][0]}+{cn[p][1]}")
                out.write("\n")

                chr = blocks[k].chr
                start_pos = blocks[k].start_pos
                cn = new_cn
    
    # Generate a text file of tree structure
    with open(f"{output_dir_name}/tree_structure.txt", "w") as out:
        out.write("Parent\tChild\n")
        tuple_lis = []
        for pop in populations:
            tuple_lis.append((pop.parent, pop.id))
        tuple_lis = sorted(tuple_lis, key=lambda x: (x[0], x[1]))
        for tuple in tuple_lis:
            out.write(f"{tuple[0]}\t{tuple[1]}\n")
    

    # ## Generate sciclone files
    # sciclone_dir = f'{output_dir_name}/sciclone'
    # if not os.path.exists(sciclone_dir):
    #     os.makedirs(sciclone_dir)
    
    # # Generate sciclone CNV file
    # for s in range(S):
    #     with open(f"{sciclone_dir}/{samples[s]}_cnv.tsv", "w") as out:
    #         # out.write("chr\tstart\tstop\tsegment_mean\n")

    #         blocks = populations[0].blocks
    #         start_pos = blocks[0].start_pos
    #         chr = blocks[0].chr
    #         cn = [populations[p].blocks[0].cn_paternal + populations[p].blocks[0].cn_maternal for p in range(P)]

    #         for k in range(1, len(blocks)):
    #             new_cn = [populations[p].blocks[k].cn_paternal + populations[p].blocks[k].cn_maternal for p in range(P)]
                
    #             if blocks[k].arm == 0 or blocks[k].chr != chr or new_cn != cn:
    #                 if blocks[k-1].arm == 0: continue
                    
    #                 end_pos = blocks[k-1].start_pos
    #                 out.write(f"{chr}\t{start_pos}\t{end_pos}")
    #                 avg_cn = np.sum(pop_ccf[s, :] * cn)
    #                 out.write(f"\t{avg_cn:.2f}")
    #                 out.write("\n")

    #                 chr = blocks[k].chr
    #                 start_pos = blocks[k].start_pos
    #                 cn = new_cn
    
    # # Generate sciclone VAF file
    # tsvs = []
    # for s in range(S):
    #     tsv_path = f"{output_dir_name}/{samples[s]}_somatic.tsv"
    #     tsvs.append(tsv_path)
    #     get_sciclone_vaf_dat(tsv_path)
    

    # ## Generate phyloWGS & Pyclone & FastClone files
    # get_phylowgs_ssm_file(tsvs)
    # get_phylowgs_cnv_file(tsvs)

    # get_pyclone_file(tsvs)
    # get_fastclone_file(tsvs)

    # ## Generate cfdna-wgs files
    # cfdna_wgs_dir = f'{output_dir_name}/cfdna-wgs'
    # if not os.path.exists(cfdna_wgs_dir):
    #     os.makedirs(cfdna_wgs_dir)
    
    # # Generate logratio & hetz_snp files
    # file_names = [
    #     "sb-01_logratio.igv",
    #     "sb-02_logratio.igv",
    #     "sb-03_logratio.igv",
    #     "sb-01_hetz_snp.tsv",
    #     "sb-02_hetz_snp.tsv",
    #     "sb-03_hetz_snp.tsv"
    # ]
    # for file_name in file_names:
    #     source_file_path = os.path.join("output/many_muts_new", "s_data_cnv_2", "cfdna-wgs", file_name)
    #     with open(source_file_path, "r") as source_file:
    #         source_content = source_file.read()
    #     with open(f"{cfdna_wgs_dir}/{file_name}", "w") as target_file:
    #         target_file.write(source_content)

    # # Generate cfdna-wgs vaf file
    # with open(f"{cfdna_wgs_dir}/somatic.vcf", "w") as out:
    #     samples_str = '\t'.join(samples)
    #     out.write(f"CHROM\tPOSITION\tREF\tALT\tGENE\tEFFECT\tNOTES\t{samples_str}\n")
        
    #     for b, block in enumerate(trunk.blocks):
    #         pop_cn = [p.blocks[b].cn_paternal + p.blocks[b].cn_maternal for p in populations]

    #         uniq_id = []
    #         for p in range(len(populations)):
    #             for mut in populations[p].blocks[b].mutations:
    #                 uniq_id.append(mut.id)
    #         uniq_id = list(set(uniq_id))

    #         positions = [999_999//len(uniq_id)*k for k in range(len(uniq_id))]

    #         for mut_id, position in zip(uniq_id, positions):
    #             pop_macn = np.zeros(len(populations), dtype=int)
    #             for p in range(len(populations)):
    #                 for m in populations[p].blocks[b].mutations:
    #                     if m.id == mut_id:
    #                         pop_macn[p] = m.macn
    #                         break

    #             mut_reads = np.zeros(len(samples), dtype=int)
    #             total_reads = np.zeros(len(samples), dtype=int)
    #             for s in range(len(samples)):
    #                 total_cn = sum(pop_ccf[s] * pop_cn)
    #                 total_macn = sum(pop_ccf[s] * pop_macn)

    #                 cn_adjusted_depth = (cancer_frac[s] * total_cn / 2 * depth) + ((1 - cancer_frac[s]) * depth)

    #                 noisy_depth = np.random.poisson(cn_adjusted_depth)

    #                 true_maf = total_macn * cancer_frac[s] / (cancer_frac[s] * total_cn + ((1 - cancer_frac[s]) * 2))

    #                 mut_reads[s] = np.random.binomial(noisy_depth, true_maf)
    #                 total_reads[s] = noisy_depth

    #             # # Don't show a mutation if it doesn't pass detection thresholds in any sample
    #             # if not any(mut_reads[s] >= mutant_read_threshold and mut_reads[s] / total_reads[s] >= maf_threshold for s in range(len(samples))):
    #             #     continue

    #             out.write(f"chr{block.chr}\t{block.start_pos + position}\tA\tT\t\t\t")
    #             for s in range(len(samples)):
    #                 out.write(f"\t{mut_reads[s]}:{total_reads[s]}:*")
    #             out.write('\n')
    
    # # Change mut_ids for testing
    # last_tsv_path = f"{output_dir_name}/{samples[S-1]}_somatic.tsv"
    # last_df = readtsv(last_tsv_path).copy()
    # last_df = last_df.sort_values(by=['subclone_id', 'total_cn'])
    # ordered_mut_ids = last_df['mutation_id'].tolist()

    # for s in range(S):
    #     tsv_path = f"{output_dir_name}/{samples[s]}_somatic.tsv"
    #     ori_df = readtsv(tsv_path)
    #     col_names = ori_df.columns.tolist()

    #     for col in ['var_counts', 'ref_counts', 'subclone_id', 'chr', 'position', 'normal_cn', 'major_cn', 'minor_cn']:
    #         ori_df[col] = ori_df[col].astype(int).astype(str)

    #     df = pd.DataFrame(columns=col_names)
    #     for mut_id in ordered_mut_ids:
    #         new_row = ori_df[ori_df['mutation_id'] == mut_id].iloc[0]
    #         df.loc[len(df)] = new_row

    #     last_clone_id = -1
    #     index = 1
    #     for row_id, row in df.iterrows():
    #         clone_id = row['subclone_id']
    #         if clone_id != last_clone_id:
    #             index = 1
    #             last_clone_id = clone_id
    #         df.loc[row_id, 'mutation_id'] = f"{int(clone_id)}-{index}"
    #         index += 1
        
    #     output_tsv(df, tsv_path)
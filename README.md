# MyClone

MyClone is a probabilistic method that reconstructs the clonal composition of tumors from deep sequencing genomic data. It has high clustering accuracy and extremely fast runtime. It integrates SNV and CNA information to achieve more accurate reconstruction, and it can reconstruct the complete clonal landscape by integrating data from multiple sequencing samples of the same patient.

# Installation

FastClone needs Python 3.5 or later version. It needs logbook, python-fire,
scikit-learn, and pandas. To install the package using Pip,

```
git clone https://github.com/GuanLab/FastClone_GuanLab.git
pip install FastClone_GuanLab/
```

(Please make sure you have the slash at the end, which forces pip to install from local directory, otherwise it will run into error)

You also can directly pip install FastClone with the command below.
```
pip install fastclone-guanlab
```
# Usage

## Input format

The input consists of several TSV files, representing multiple sequencing samples from the same patient. The mandatory columns of one TSV file are as follows.

- mutation_id: A unique identifier for the mutation. This should be the same across sequencing samples.
- ref_counts: The number of reads overlapping the locus matching the reference allele.
- var_counts: The number of reads overlapping the locus matching the variant allele.
- total_cn: The average copy number at the gene site. For example, with a tumor purity of 70%, a subclonal CNA that alters the copy number from 2 to 3 at the affected sites occurs in 50% of the tumor cells. Then, the average copy number at this site is calculated as 3 * 50% * 70% + 2 * (1 - 50% * 70%) = 2.35.
- VAF: The variant allele fraction of the mutation.

Any other columns will be ignored. Note that when there are multiple sequencing samples, the set of mutations for each sample should be the same. An input example can be found in "example/input".

## Usage

The MyClone program requires two mandatory arguments:

- `-i`: A space delimited set of TSV files formatted as specified in the input format section.
- `-o`: A directory where the program will output the results.

Then the MyClone program can be run by executing the following command:
```
python src/run_myclone.py -i [TSV file 1] [TSV file 2] ... [TSV file n] -o [Output Folder]
```

Based on the sample input data we provided, you can run the following command to test the program's executability:
```
python src/run_myclone.py -i example/input/01.tsv example/input/02.tsv example/input/03.tsv -o output/
```

## Output format

The output is a TSV file that contains three columns of information:

- mutation_id: A unique identifier for the mutation which is consistent with the input data.
- myclone_id: The clone index (which starts from 1) predicted by MyClone for the mutation.
- myclone_ccf: The cancer cell fraction (CCF) predicted by MyClone for the mutation. When there are multiple sequencing samples in the input, the CCFs are separated by commas, listing the CCF of the mutation in each sample in order of the input.

# Simulated Data


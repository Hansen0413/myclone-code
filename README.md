# MyClone

MyClone is a probabilistic method that reconstructs the clonal composition of tumors from deep sequencing genomic data. It has high clustering accuracy and extremely fast runtime. It integrates SNV and CNA information to achieve more accurate reconstruction, and it can reconstruct the complete clonal landscape by integrating data from multiple sequencing samples of the same patient.

# Installation

MyClone is a program written entirely in Python. It needs Python 3.9 or later version.

An installation-free way to run MyClone is by using the [Docker image](https://hub.docker.com/r/hansen0413/myclone-code). 

To run MyClone, you can directly deploy the project locally by using the command:

```
git clone https://github.com/Hansen0413/myclone-code.git
```

Next, switch the working directory to the main directory of the project (i.e., the directory where `requirements.txt` is located), and run the following command to install the dependencies: (Since the code uses functions that are not present in the latest library versions, it is recommended to create a new Python environment and install the libraries according to the versions specified in `requirements.txt`.)

```
pip install -r requirements.txt
```

After the dependencies are installed, keep the working directory and follow the subsequent instructions to run MyClone.

# Usage

## Input format

The input consists of several TSV files, representing multiple sequencing samples from the same patient. The mandatory columns of one TSV file are as follows.

- mutation_id: A unique identifier for the mutation. This should be the same across sequencing samples.
- ref_counts: The number of reads overlapping the locus matching the reference allele.
- var_counts: The number of reads overlapping the locus matching the variant allele.
- total_cn: The average copy number at the gene site. For example, with a tumor purity of 70%, a subclonal CNA that alters the copy number from 2 to 3 at the affected sites occurs in 50% of the tumor cells. Then, the average copy number at this site is calculated as 3 * 50% * 70% + 2 * (1 - 50% * 70%) = 2.35.
- VAF: The variant allele fraction of the mutation.

Any other columns will be ignored. Note that when there are multiple sequencing samples, the set of mutations for each sample should be the same. An input example can be found in `example/input`.

If the tumor purity for each sample is provided in the input, a corresponding tumor purity file exists in the same folder as each sample file. The naming convention follows `{sample_name}_purity.tsv` (e.g., if the sample file is `01.tsv`, the corresponding tumor purity file is `01_purity.tsv`). The file contains only a single floating-point number (between 0 and 1), representing the tumor purity of the sample. Examples can be found in `example/input`.

## Usage

The MyClone program requires two mandatory arguments:

- `-i`: A space delimited set of TSV files formatted as specified in the input format section.
- `-o`: A directory where the program will output the results.

and one optional argument:

- `--tumor_purity`: Infer using the tumor purity from the input if this argument is provided.

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

We also provide a program for generating simulated sequencing samples, which is based on modifications to the program in https://github.com/annalam/cfdna-wgs-manuscript-code.

You can run the command `python src/run_simulate.py` to generate simulated data, which conforms to the MyClone input format.

The program has one required argument and several optional arguments:

- `-o`: A directory where the program will output the results.
- `--num_patients`: Number of simulated patient samples (default: 100).
- `--num_samples`: Number of sequencing samples per patient (default: 3).
- `--depth`: Average sequencing depth (default: 2000).
- `--mut_N`: Average mutation number of each subclone (It should be an integer multiple of 2.5, and the default value is 2.5).
- `--clone_num`: Number of clones (default: 6).

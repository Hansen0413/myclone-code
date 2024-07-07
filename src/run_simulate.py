import simulate_mut
import os
import argparse
import shutil

def delete_last_folder(folder_name, i):
    folder_to_delete = os.path.join(folder_name, f"s_data_{i}")
    shutil.rmtree(folder_to_delete)

def main():
    parser = argparse.ArgumentParser(description='Generate simulated mutation files')
    parser.add_argument('--num_patients', type=int, default=100, help='Number of simulated patient samples (default: 100).')
    parser.add_argument('-o', '--output_folder', type=str, help='Output folder path (required).', required=True)
    parser.add_argument('--num_samples', type=int, default=3, help='Number of sequencing samples per patient (default: 3).')
    parser.add_argument('--depth', type=int, default=2000, help='Average sequencing depth (default: 2000).')
    parser.add_argument('--mut_N', type=int, default=2.5, help='Average mutation number of each subclone (It should be an integer multiple of 2.5, and the default value is 2.5).')
    parser.add_argument('--clone_num', type=int, default=6, help='Number of clones (default: 6).')
    args = parser.parse_args()

    output_folder = args.output_folder
    num_samples = args.num_samples
    depth = args.depth
    mut_N = int(args.mut_N / 2.5)
    clone_num = args.clone_num

    max_iterations = args.num_patients  # Max iteration number
    i = 1  # Initial index

    while i <= max_iterations:
        try:
            simulate_mut.wgs(num_samples, output_folder, depth=depth, mut_N=mut_N, clone_num=clone_num)
            i += 1
        except Exception as e:
            print(e)
            delete_last_folder(output_folder, i)
            print(f"Folder {i} deleted.")
            continue

        if (i-1) % 1 == 0:
            print(f"{i-1} examples generated.")

if __name__ == "__main__":
    main()
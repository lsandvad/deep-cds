import os
import subprocess
from tqdm import tqdm

# Path to deamSim
deamsim_path = "/home/nrt204/FragmentPredictor/gargammel/src/deamSim"
data_path = "/tmp/nrt204/FragmentPredictor/data/processed_data/reads_processed/test"

testsets_to_simulate = [
    "without_errors_60bp",
    "without_errors_75bp",
    "without_errors_100bp",
    "without_errors_150bp",
    "without_errors_300bp"
]

for testset in testsets_to_simulate:
    print("Simulating ancient damage for test set:", testset)
    fasta_dir = os.path.join(data_path, testset, "fasta")
    output_dir = os.path.join(data_path, testset, "fasta_ancient_damage")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    accession_files_fasta = os.listdir(fasta_dir)
    for acc_file in tqdm(accession_files_fasta):
        input_path = os.path.join(fasta_dir, acc_file)
        acc_name = acc_file.replace(".fasta.gz", "")
        output_path = os.path.join(output_dir, "{}_ancient.fasta.gz".format(acc_name))

        # Apply ancient damage
        subprocess.call([
            deamsim_path,
            '-damage', '0.024,0.36,0.0097,0.68',
            '-o', output_path,
            input_path
        ])

        break
    break

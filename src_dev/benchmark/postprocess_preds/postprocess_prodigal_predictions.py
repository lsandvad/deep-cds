import os
import pickle

from tqdm import tqdm

project_root = "../../.."

test_accessions = open(f"{project_root}/data/processed_data/genome_partitions/test_partition_accessions.txt").read().splitlines()


def process_model_preds(test_accession, testset_type):
    """
    Process model predictions from a GFF file for a given test accession.

    Args:
        test_accession (str): The accession identifier for the test dataset.
        testset_type (str): The testset type directory.

    Returns:
        model_dict (dict): A dictionary with CDS > 60bp.
    """

    model_dict = dict()

    with open(f"{project_root}/data/processed_data/predictions/raw_predictions/prodigal_preds/{testset_type}/{test_accession}/{test_accession}.gff", "r") as file:
        file.readline()  # Skip first line

        for line in file:
            fields = line.split("\t")
            read_name = fields[0].split("|")[0]
            attr_type = fields[2]
            strand = fields[6]
            assert strand == "+", "Complement strand predictions not filtered out properly!"

            if attr_type == "CDS":

                if read_name not in model_dict.keys():
                    model_dict[read_name] = dict()
                    model_dict[read_name]["cds_coords"] = []
                    model_dict[read_name]["cds_fragments_connection"] = []

                    counter_cds_on_read = 0

                cds_start = int(fields[3])
                cds_end = int(fields[4])

                if cds_start % 3 == 1:
                    rf = 0
                elif cds_start % 3 == 2:
                    rf = 1
                elif cds_start % 3 == 0:
                    rf = 2

                cds_coords = [cds_start, cds_end, str(rf)]

                # Save sequences of length 60 or more (prodigal can only go down to this length,
                # but due to FGS' processing (does not predict the last codon in RF0),
                # we miss some sequences which are also removed from test set)
                if cds_end - cds_start + 1 > 60:
                    model_dict[read_name]["cds_coords"].append(cds_coords)
                    model_dict[read_name]["cds_fragments_connection"].append([counter_cds_on_read])
                    counter_cds_on_read += 1

    model_dict = {
        read_name: data
        for read_name, data in model_dict.items()
        if data["cds_coords"]  # Check if cds_coords is not empty
    }

    return model_dict


if __name__ == "__main__":

    data_dirs = ['without_errors_60bp',
                 'without_errors_75bp',
                 'without_errors_100bp',
                 'without_errors_150bp',
                 'without_errors_300bp',
                 'without_errors_700bp',
                 'without_errors_1000bp',
                 'with_errors_5e-06i_0.004s_60bp',
                 'with_errors_1.25e-05i_0.01s_60bp',
                 'with_errors_3.75e-05i_0.03s_60bp',
                 'with_errors_5e-06i_0.004s_75bp',
                 'with_errors_1.25e-05i_0.01s_75bp',
                 'with_errors_3.75e-05i_0.03s_75bp',
                 'with_errors_5e-06i_0.004s_100bp',
                 'with_errors_1.25e-05i_0.01s_100bp',
                 'with_errors_3.75e-05i_0.03s_100bp',
                 'with_errors_5e-06i_0.004s_150bp',
                 'with_errors_1.25e-05i_0.01s_150bp',
                 'with_errors_3.75e-05i_0.03s_150bp',
                 'with_errors_5e-06i_0.004s_300bp',
                 'with_errors_1.25e-05i_0.01s_300bp',
                 'with_errors_3.75e-05i_0.03s_300bp',
                 "HiSeq2500_150bp",
                 "MiSeq_v3_300bp",
                 "NextSeq500_150bp"]

    for testset_type in tqdm(data_dirs, desc="Processing predictions for testsets.."):
        print(testset_type)

        for test_accession in test_accessions:
            os.makedirs(f"{project_root}/data/processed_data/predictions/processed_predictions/prodigal_preds/{testset_type}/{test_accession}", exist_ok=True)

            model_preds_dict = process_model_preds(test_accession, testset_type)

            with open(f"{project_root}/data/processed_data/predictions/processed_predictions/prodigal_preds/{testset_type}/{test_accession}/model_preds_dict.pkl", "wb") as processed_preds_file:
                pickle.dump(model_preds_dict, processed_preds_file)

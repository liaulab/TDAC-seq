import pandas as pd
from tdac_seq.ddda_dataset import ddda_dataset
import re
import itertools
import os
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
import sys
from tdac_seq.refseq import extract_ref_seq, regions_df_to_dict
import pickle

assert len(sys.argv) == 2
subset = [int(sys.argv[1])]

parent_working_dir_prefix = "output/"

regions = pd.read_csv('data/regions.tsv', sep='\t', index_col="name")
datasets = pd.read_csv('data/datasets.tsv', sep='\t', index_col="ID")
if subset is not None:
    if all(isinstance(x, int) for x in subset):
        datasets = datasets.iloc[subset].copy()
    elif all(isinstance(x, str) for x in subset):
        datasets = datasets.loc[subset].copy()
    else:
        raise ValueError("subset must be a sequence of dataset ID strings or a sequence of dataset index integers")
datasets['regions'] = datasets['regions'].str.split(',')
datasets['working_dir'] = parent_working_dir_prefix + datasets.index
assert set(itertools.chain.from_iterable(datasets['regions'])).issubset(regions.index), "Some regions are not in the regions.tsv file"
if not os.path.exists('data/ref.fa'):
    extract_ref_seq(genome_file='/n/holystore01/LABS/liau_lab/Users/sps/genomes/hg38_ucsc/hg38.fa', region_dict=regions_df_to_dict(regions), out_fasta_path='data/ref.fa')

for id, data_info in datasets.iterrows():
    try:
        force_dedup = False
        logging.info(f"Processing dataset {id}")

        object_fname = os.path.join(data_info['working_dir'], "ddda_data.pkl")
        if os.path.exists(object_fname):
            logging.info(f"Loading existing dataset object: {object_fname}")
            ddda_data: ddda_dataset = pickle.load(open(object_fname, "rb"))
        else:
            logging.error(f"Could not find existing dataset object: {object_fname}. Skipping.")
            continue

        if ddda_data.check_valid_alignment():
            logging.info("Alignment looks good")
        else:
            logging.error("Could not find valid alignment. Need to align first. Skipping.")
            continue

        if getattr(ddda_data, "deduped_inds", None):
            logging.info(f"deduped_inds found. Force dedup = {force_dedup}")
        else:
            logging.info("deduped_inds not found. Will dedup")
            force_dedup = True

        # Dedup reads
        if force_dedup:
            logging.info("Deduping data")
            ddda_data.dedup_all()

            logging.info(f"Dumping into {object_fname}")
            with open(object_fname, "wb") as f:
                pickle.dump(ddda_data, f)

        logging.info(f"Done with dataset {id}")
    except Exception as e:
        import traceback
        logging.error(f"Error processing dataset {id}. Skipping.")
        traceback.print_exc()

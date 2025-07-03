import pandas as pd
from tdac_seq.ddda_dataset import ddda_dataset
import itertools
import os
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
import sys
from tdac_seq.refseq import regions_df_to_dict
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

for id, data_info in datasets.iterrows():
    try:
        force_align = False
        logging.info(f"Processing dataset {id}")

        object_fname = os.path.join(data_info['working_dir'], "ddda_data.pkl")
        if os.path.exists(object_fname):
            logging.info(f"Loading existing dataset object: {object_fname}")
            ddda_data: ddda_dataset = pickle.load(open(object_fname, "rb"))
        else:
            # Initialize ddda_dataset
            ddda_data = ddda_dataset(
                ID = id, 
                region_dict = regions_df_to_dict(regions.loc[data_info['regions']]), 
                ref_fasta_path = 'data/ref.fa',
                fastq_file = data_info['fastq_file'],
                working_dir = data_info['working_dir']
            )
            force_align = True

        if force_align == False and not hasattr(ddda_data, 'ABE_edit_dict'):
            logging.info("Missing ABE_edit_dict, so forcing re-alignment")
            force_align = True

        # Load DddA bias model
        ddda_data.load_bias_model("data/bias_dict.pkl")

        # Align reads
        gap_threshold = 500
        if force_align:
            logging.info("Aligning data")
            ddda_data.align_reads(
                start_gap_threshold = gap_threshold,
                end_gap_threshold = gap_threshold,
            )
        
        # Check alignment worked
        if not ddda_data.check_valid_alignment(gap_threshold, gap_threshold):
            logging.error("Invalid alignment. Skipping.")
            continue

        if force_align:
            logging.info(f"Dumping into {object_fname}")
            with open(object_fname, "wb") as f:
                pickle.dump(ddda_data, f)

        logging.info(f"Done with dataset {id}")
    except Exception as e:
        import traceback
        logging.error(f"Error processing dataset {id}. Skipping.")
        traceback.print_exc()

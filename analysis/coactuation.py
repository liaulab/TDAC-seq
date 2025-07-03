import pandas as pd
from tdac_seq.ddda_dataset import ddda_dataset
import re
import itertools
import os
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
import sys
import pickle
import numpy as np
from tdac_seq.analysis.genome_tracks import make_genome_tracks_plot

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
datasets['regions_of_interest'] = datasets['regions_of_interest'].str.split(',').fillna('')

assert set(itertools.chain.from_iterable(datasets['regions'])).issubset(regions.index), "Some regions are not in the regions.tsv file"

for id, data_info in datasets.iterrows():
    try:
        logging.info(f"Processing dataset {id}")

        object_fname = os.path.join(data_info['working_dir'], "ddda_data.pkl")
        if not os.path.exists(object_fname):
            logging.error(f"Pre-aligned data not found: {object_fname}. Skipping dataset {id}")
            continue

        logging.info(f"Loading pre-aligned data from {object_fname}")
        with open(object_fname, "rb") as f:
            ddda_data: ddda_dataset = pickle.load(f)

        # Make genome tracks
        for locus, ref_seq in ddda_data.ref_seq_dict.items():
            plot_dir = f"plots/{id}"
            os.makedirs(plot_dir, exist_ok=True)

            edits = ddda_data.edit_dict[locus][ddda_data.deduped_inds[locus]]
            editable = np.array([i for i, b in enumerate(ref_seq) if b in "CG" and i > 500 and i < len(ref_seq) - 500])
            strands = ddda_data.read_strands[locus][ddda_data.deduped_inds[locus]]
            del_matrix = ddda_data.del_dict[locus][ddda_data.deduped_inds[locus]]
            genomic_region = ddda_data.region_dict[locus]

            plot_fname = os.path.join(plot_dir, f'genome_tracks_{locus}.pdf')
            additional_rois = {} # maps roi to filename
            for roi in data_info['regions_of_interest']:
                m = re.match(r'(\w+):(\d+)-(\d+)', roi).groups()
                m = (m[0], int(m[1]), int(m[2]))
                additional_rois[m] = os.path.join(plot_dir, f'genome_tracks_{locus}_({m[0]}_{m[1]}-{m[2]}).pdf')
            logging.info(f"Making genome tracks plot for {locus}")
            make_genome_tracks_plot(
                genomic_region=genomic_region,
                edits=edits[:10000].toarray(),
                editable=editable,
                ref_seq=ref_seq,
                strands=strands[:10000],
                plot_fname=plot_fname,
                additional_rois=additional_rois,
            )
            logging.info(f"Done making genome tracks plot for {locus}. Output: {plot_fname}")

        logging.info(f"Done with dataset {id}")
    except Exception as e:
        import traceback
        logging.error(f"Error processing dataset {id}. Skipping.")
        traceback.print_exc()

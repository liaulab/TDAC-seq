import pandas as pd
import numpy as np
from Bio.Seq import reverse_complement
import matplotlib.pyplot as plt
from tqdm import tqdm
from tdac_seq.crispr import find_cutsite

def make_deletion_coverage_plot(del_matrix: np.ndarray, guides: pd.Series, ref_seq: str, plot_fname: str):
    '''
    :param del_matrix: array of shape (reads, sequence) with 1 for deleted base and 0 for undeleted base
    '''
    # Find sgRNA positions in the target locus
    cut_sites = [find_cutsite(ref_seq, spacer) for _, spacer in guides.items()]

    # Visualize deletion coverage and sgRNA positions
    del_coverage = np.squeeze(np.array(np.mean(del_matrix, axis=0)))
    plt.figure(figsize=(12, 4))
    plt.plot(np.arange(len(del_coverage)), del_coverage)
    for cut_site in cut_sites:
        plt.axvline(x = cut_site, color="red", linestyle="dashed")
    plt.ylabel("Deletion coverage")
    plt.xlabel("Position in the locus")
    plt.savefig(plot_fname, bbox_inches="tight")
    plt.close()

def make_guide_effects_heatmap(del_matrix, edits, strands, guides: pd.Series, ref_seq: str):
    del_edits = np.zeros((len(guides), edits.shape[1]))
    
    # Separately, also find reads without any deletion as a control
    read_with_del = np.max(del_matrix, axis=1).astype(bool).toarray()[..., 0]
    undel_reads = ~read_with_del
    undel_reads[1000:] = False
    undel_edits = np.mean(edits[~read_with_del & (strands == 0), :], axis=0) + np.mean(edits[~read_with_del & (strands == 1), :], axis=0)

    for i, (guide_id, spacer) in tqdm(enumerate(guides.items()), total=len(guides)):
        cut_site = find_cutsite(ref_seq, spacer)

        # Find reads where the sgRNA target site is covered by a deletion
        del_start, del_end = cut_site - 20, cut_site + 20
        target_site_del = np.max(del_matrix[:, del_start:del_end], axis=1).astype(bool).toarray()[..., 0]
        
        # Only keep reads where positions outside of the vicinity of the sgRNA target site are not deleted
        upstream_filter = np.max(del_matrix[:, :del_start], axis=1).astype(bool).toarray()[..., 0]
        downstream_filter = np.max(del_matrix[:, del_end:], axis=1).astype(bool).toarray()[..., 0]

        del_read_inds = target_site_del & ~upstream_filter & ~downstream_filter

        # Calculate average editing rate on deleted reads for both C-to-T and G-to-A strands
        del_edits[i, :] = np.mean(edits[del_read_inds & (strands == 0), :], axis=0) + np.mean(edits[del_read_inds & (strands == 1), :], axis=0)
    
    return del_edits, undel_edits

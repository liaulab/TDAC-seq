import numpy as np
import matplotlib.pyplot as plt
import pyBigWig
from tdac_seq.dad import call_dads_hmm
import seaborn as sns

def make_dad_length_plot(genomic_region: tuple[str, int, int], edits: np.ndarray, ref_seq: str, strands: np.ndarray, plot_fname: str):
    peaks = pyBigWig.open("https://www.encodeproject.org/files/ENCFF086JCJ/@@download/ENCFF086JCJ.bigBed").entries(*genomic_region)

    dad = call_dads_hmm(edits, ref_seq, strands).astype(int)
    dad_start = np.nonzero(np.diff(dad) == 1)[1]
    dad_end = np.nonzero(np.diff(dad) == -1)[1]
    dad_length = dad_end - dad_start

    peak_start, peak_end, _ = zip(*peaks)
    peak_start = np.array(peak_start) - genomic_region[1]
    peak_end = np.array(peak_end) - genomic_region[1]

    # find dads that overlap with peaks
    dad_peak_overlap = np.zeros_like(dad_start, dtype=bool)
    for i, (start, end) in enumerate(zip(dad_start, dad_end)):
        dad_peak_overlap[i] = np.any((peak_start <= end) & (start <= peak_end))

    fig, ax = plt.subplots(figsize=(2, 4))
    sns.boxplot(x=dad_peak_overlap, y=dad_length, showfliers=False, ax=ax)
    ax.set_xlabel("DADs overlapping\nATAC peak")
    ax.set_ylabel("DAD length (bp)")
    fig.savefig(plot_fname, bbox_inches="tight")
    plt.close(fig)

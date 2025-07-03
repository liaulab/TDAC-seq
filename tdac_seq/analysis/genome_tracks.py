from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import pyBigWig
from tdac_seq.plots import BedEntry, BedTrack, HiCTrack, ArchesTrack, plot_genome_tracks, BigWigTrack, HiCTrackSquare, GenomeTrack, OverlaidBigWigTrack
from itertools import combinations
import scipy.stats as ss
from tdac_seq.dad import call_dads_hmm_bias
from tdac_seq.crispr import find_cutsite
import pandas as pd
from typing import Literal
from tdac_seq.analysis.coactuation import calc_coactuation
import matplotlib.ticker
import matplotlib.colors
import logging

def _tracks_reference(genomic_region: tuple[str, int, int]) -> list[GenomeTrack]:
    tracks = []
    bigwigtracks = {
        "ATAC-seq": "https://www.encodeproject.org/files/ENCFF357GNC/@@download/ENCFF357GNC.bigWig",
        "H3K4me1": "https://www.encodeproject.org/files/ENCFF100FDI/@@download/ENCFF100FDI.bigWig",
        "H3K4me3": "https://www.encodeproject.org/files/ENCFF405ZDL/@@download/ENCFF405ZDL.bigWig",
        "H3K27ac": "https://www.encodeproject.org/files/ENCFF465GBD/@@download/ENCFF465GBD.bigWig",
    }
    tracks.extend([BigWigTrack(track_name, pyBigWig.open(track_url).values(*genomic_region)) for track_name, track_url in bigwigtracks.items()])
    # # annotate predicted transcription factor binding sites
    # entries = list(BedEntry(start=entry[0] - genomic_region[1], end=entry[1] - genomic_region[1], name=entry[2].split("\t")[3], score=int(entry[2].split("\t")[1])) for entry in pyBigWig.open("https://frigg.uio.no/JASPAR/JASPAR_TFBSs/2024/JASPAR2024_hg38.bb").entries(*genomic_region))
    # tracks.append(BedTrack("JASPAR TFBS", [entry for entry in entries if entry.score >= 500]))
    return tracks

def _tracks_ddda(edits: np.ndarray, editable: np.ndarray, dads: np.ndarray) -> list[GenomeTrack]:
    tracks = []
    # raw DddA edits track
    edits_plot = edits[:10000]
    num_edits = np.sum(edits_plot, axis=1)
    edits_argsort = np.argsort(num_edits)[::-1]
    cmap = matplotlib.colors.ListedColormap([(1, 1, 1, 1), plt.cm.Purples(255)])
    tracks.append(HiCTrackSquare("Raw DddA edits", (edits_plot == 1)[edits_argsort], cmap=cmap, norm=plt.Normalize(0, 1)))
    # runs of DddA edits
    tracks.append(HiCTrackSquare("DADs", dads[edits_argsort], cmap=cmap, norm=plt.Normalize(0, 1)))
    return tracks

def _tracks_correlation(dads: np.ndarray, method: Literal["pearsonr", "pearsonr_pval", "coactuation"] = "coactuation") -> tuple[list[GenomeTrack], np.ndarray]:
    # Call DADs
    edits_pool = dads
    tracks = []
    # DddA correlation track
    if method == "pearsonr":
        with np.errstate(invalid='ignore'):
            edits_corr = np.corrcoef(edits_pool.T)
        edits_pval = np.abs(edits_corr)
        tracks.append(HiCTrack(f"DAD correlation\nN={len(dads)}", edits_corr))
        tracks.append(ArchesTrack("Correlated", (BedEntry(start=i, end=j, name=None, score=edits_pval[i, j]) for i, j in np.argwhere(edits_pval > 0.3)), vmin=0.3, vmax=0.6, ylim=(1, dads.shape[1]), above=False))
    elif method == "pearsonr_pval":
        edits_corr = np.full((edits_pool.shape[1], edits_pool.shape[1]), np.nan)
        edits_pval = np.full((edits_pool.shape[1], edits_pool.shape[1]), np.nan)
        inds = list(combinations(range(edits_pool.shape[1]), 2))
        for i, j in tqdm(inds):
            assert i < j
            edits_corr[i, j], edits_pval[i, j] = ss.pearsonr(edits_pool[:, i], edits_pool[:, j])
        tracks.append(HiCTrack(f"DAD correlation\nN={len(dads)}", edits_corr))
        tracks.append(ArchesTrack("Significantly correlated", (BedEntry(start=i, end=j, name=None, score=-np.log10(max(1e-330, edits_pval[i, j]))) for i, j in np.argwhere(edits_pval < 1e-300)), vmin=300, vmax=330, ylim=(1, dads.shape[1]), above=False))
    elif method == "coactuation":
        edits_corr = calc_coactuation(edits_pool)
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("WhiteOrangeRed", ['white', 'orange', 'red'])
        tracks.append(HiCTrack(f"DAD co-accessibility\nN={len(dads)}", edits_corr, cmap=cmap, norm=plt.Normalize(0, edits_corr.max()), cbar_kwargs=dict(format=matplotlib.ticker.PercentFormatter(1., decimals=0))))
    else:
        raise ValueError(f"Invalid method: {method}")

    return tracks, edits_corr

def make_genome_tracks_plot(genomic_region: tuple[str, int, int], edits: np.ndarray, editable: np.ndarray, ref_seq: str, strands: np.ndarray, plot_fname: str, additional_rois: dict[tuple[str, int, int], str] = {}):
    '''
    Make genome tracks plot
    :param genomic_region: tuple of chromosome, start, end
    :param edits: array of shape (reads, sequence) with 1 for edited base and 0 for unedited base
    :param editable: array indexing editable bases
    '''

    logging.info("Calling DADs")
    dads = call_dads_hmm_bias(edits, ref_seq, strands).toarray()

    logging.info("Retrieving reference data")
    tracks_ref = _tracks_reference(genomic_region)
    logging.info("Sorting reads")
    tracks_ddda = _tracks_ddda(edits, editable, dads)
    logging.info("Analyzing co-actuation")
    tracks_corr, _ = _tracks_correlation(dads)

    logging.info("Making plots")
    additional_rois[None] = plot_fname
    for roi, roi_plot_fname in additional_rois.items():
        if roi is None:
            xlim = (0, edits.shape[1])
        else:
            assert roi[0] == genomic_region[0]
            assert roi[1] >= genomic_region[1]
            assert roi[2] <= genomic_region[2]
            xlim = (roi[1] - genomic_region[1], edits.shape[1] - (genomic_region[2] - roi[2]))
        fig = plot_genome_tracks(
            tracks=[*tracks_ref, *tracks_ddda, *tracks_corr],
            xlim=xlim)
        fig.savefig(roi_plot_fname, bbox_inches="tight")
        plt.close(fig)
        logging.info(f"Done making plot: {roi_plot_fname}")

def make_genome_tracks_plot_for_guides(genomic_region: tuple[str, int, int], guides: pd.Series, del_matrix: np.ndarray, edits: np.ndarray, ref_seq: str, plot_fname_guides: str, plot_fname_pca: str, strands: np.ndarray, editable: np.ndarray, additional_rois: dict[tuple[str, int, int], str] = {}):

    tracks_ref = _tracks_reference(genomic_region)

    read_with_del = np.max(del_matrix, axis=1).astype(bool).toarray()[..., 0]
    undel_reads = ~read_with_del
    undel_reads[10000:] = False
    edits_sub_undel = edits[undel_reads, :].toarray()
    dads_sub_undel = call_dads_hmm_bias(edits_sub_undel, ref_seq, strands[undel_reads]).toarray()
    aggregate_edits_undel = edits_sub_undel.mean(axis=0)
    aggregate_dad_undel = dads_sub_undel.mean(axis=0)
    tracks_corr_wt, edits_corr_wt = _tracks_correlation(dads_sub_undel)
    tracks_ddda_wt = _tracks_ddda(edits_sub_undel, editable, dads_sub_undel)

    additional_rois[None] = plot_fname_guides
    for roi, roi_plot_fname in additional_rois.items():
        if roi is None:
            xlim = (0, edits.shape[1])
        else:
            assert roi[0] == genomic_region[0]
            assert roi[1] >= genomic_region[1]
            assert roi[2] <= genomic_region[2]
            xlim = (roi[1] - genomic_region[1], edits.shape[1] - (genomic_region[2] - roi[2]))
        fig = plot_genome_tracks([*tracks_ref, *tracks_ddda_wt, *tracks_corr_wt], xlim=xlim)
        fig.savefig(roi_plot_fname.format("wt"), bbox_inches="tight")
        plt.close(fig)

    edits_corr = np.full((len(guides), len(ref_seq), len(ref_seq)), np.nan)
    for i, (id, spacer) in tqdm(enumerate(guides.items()), total=len(guides)):
        cut_site = find_cutsite(ref_seq, spacer)

        # Find reads where the sgRNA target site is covered by a deletion
        del_start, del_end = cut_site - 20, cut_site + 20
        target_site_del = np.max(del_matrix[:, del_start:del_end], axis=1).astype(bool).toarray()[..., 0]
        
        # Only keep reads where positions outside of the vicinity of the sgRNA target site are not deleted
        upstream_filter = np.max(del_matrix[:, :del_start], axis=1).astype(bool).toarray()[..., 0]
        downstream_filter = np.max(del_matrix[:, del_end:], axis=1).astype(bool).toarray()[..., 0]

        del_read_inds = target_site_del & ~upstream_filter & ~downstream_filter

        edits_del = edits[del_read_inds, :].toarray()
        dads_del = call_dads_hmm_bias(edits_del, ref_seq, strands[del_read_inds]).toarray()
        tracks_corr, edits_corr[i] = _tracks_correlation(dads_del)
        tracks_ddda = _tracks_ddda(edits_del, editable, dads_del)
        track_line_edits = OverlaidBigWigTrack("DddA mutation rate", np.stack([aggregate_edits_undel, edits_del.mean(axis=0)]))
        track_line = OverlaidBigWigTrack("Proportion of reads with a DAD", np.stack([aggregate_dad_undel, dads_del.mean(axis=0)]))

        additional_rois[None] = plot_fname_guides
        for roi, roi_plot_fname in additional_rois.items():
            if roi is None:
                xlim = (0, edits.shape[1])
            else:
                assert roi[0] == genomic_region[0]
                assert roi[1] >= genomic_region[1]
                assert roi[2] <= genomic_region[2]
                xlim = (roi[1] - genomic_region[1], edits.shape[1] - (genomic_region[2] - roi[2]))
            fig = plot_genome_tracks([*tracks_ref, track_line_edits, track_line, *tracks_ddda, *tracks_corr], axvline=cut_site, xlim=xlim)
            fig.savefig(roi_plot_fname.format(id), bbox_inches="tight")
            plt.close(fig)

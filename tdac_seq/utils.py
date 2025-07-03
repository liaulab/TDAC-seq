import os
import subprocess
import re
import itertools
import mappy
import numpy as np
import pandas as pd
from tqdm import tqdm
import networkx as nx
import logging
from .dad import call_dads_hmm_bias
import time
import scipy.signal
from typing import Literal

def get_fastq_iter(fastq_file: str) -> tuple[str, str, str]:
    """
    Returns an iterator over the fastq file or directory of fastq files. Each iteration returns read ID, sequence, and quality.
    """
    if os.path.isfile(fastq_file):
        return mappy.fastx_read(fastq_file)
    elif os.path.isdir(fastq_file):
        return itertools.chain.from_iterable(mappy.fastx_read(os.path.join(fastq_file, f)) for f in os.listdir(fastq_file))
    elif ";" in fastq_file:
        return itertools.chain.from_iterable(get_fastq_iter(f) for f in fastq_file.split(";"))
    else:
        raise FileNotFoundError(f"fastq file {fastq_file} not found")

def count_reads_in_fastq(fname: str) -> int | None:
    '''
    Returns the number of reads in a gzipped fastq file or directory of gzipped fastq files.
    On timeout, returns None.
    '''
    out = 0
    start = time.time()
    TIMEOUT = 60
    if os.path.isfile(fname):
        try:
            ps = subprocess.Popen(("zless" if fname.endswith('.gz') else "less", fname), stdout=subprocess.PIPE)
            output = subprocess.check_output(('wc', '-l'), stdin=ps.stdout, timeout=TIMEOUT)
            ps.wait()
            out = int(output.strip()) // 4
        except subprocess.TimeoutExpired:
            ps.kill()
            return None
    elif os.path.isdir(fname):
        for f in os.listdir(fname):
            if time.time() - start > TIMEOUT:
                return None
            elif (out_cur := count_reads_in_fastq(os.path.join(fname, f))) is not None:
                out += out_cur
            else:
                return None
    elif ";" in fname:
        for f in fname.split(";"):
            if time.time() - start > TIMEOUT:
                return None
            elif (out_cur := count_reads_in_fastq(f)) is not None:
                out += out_cur
            else:
                return None
    else:
        raise ValueError(f"Invalid file {fname}")

    return out

def parse_cigar(cigar):
    """
    Parse a cigar string and return a list of tuples of cigar operations and their lengths.
    """
    return re.findall(r"([=:*+-~])([\dATCGNatcgn]+)", cigar)

def ref_bp_consumed(cs_op: str, cs_arg: str) -> int:
    '''
    Returns the number of bases consumed by the cigar string operation.
    '''
    if cs_op == ":":
        return int(cs_arg)
    elif cs_op == "-":
        return len(cs_arg)
    elif cs_op == "+":
        return 0
    elif cs_op == "*":
        return 1
    raise ValueError("Invalid cigar string")

def cigar_to_seq(cs: str, refseq: str, start: int = 0) -> str:
    '''
    Converts a cigar string to raw sequence.
    '''
    seq = []
    cur_pos = start
    for cs_op, cs_arg in parse_cigar(cs):
        if cs_op == ":":
            seq.append(refseq[cur_pos:cur_pos + int(cs_arg)])
            cur_pos += int(cs_arg)
        elif cs_op == "-":
            cur_pos += len(cs_arg)
        elif cs_op == "+":
            seq.append(cs_arg)
            cur_pos += 0
        elif cs_op == "*":
            seq.append(cs_arg[1])
            cur_pos += 1
        else:
            raise ValueError("Invalid cigar string")
    return "".join(seq).upper()

def load_unibind_sites(unibind_dir):
    '''
    Load Unibind TF sites and return a combined PyRanges object
    '''
    import pyranges as pr
    import tqdm
    unibind_subdirs = os.listdir(unibind_dir)
    TF_sites = []
    for subdir in tqdm.tqdm(os.listdir(unibind_dir)):
        if re.search("K562", subdir) is not None:
            TF = re.split("\\.", subdir)[-1]
            bed_files = os.listdir(os.path.join(unibind_dir, subdir))
            for bed_file in bed_files:
                bed = pd.read_table(os.path.join(unibind_dir, subdir, bed_file), header=None)
                bed.columns = ["Chromosome", "Start", "End", "Seq", "Score", "Strand"]
                bed["TF"] = TF
                TF_sites.append(bed)
    TF_sites = pd.concat(TF_sites, axis=0)
    TF_sites = pr.PyRanges(TF_sites)
    return TF_sites

def generate_low_saturation_color():
    '''
    Generate a random base value for the RGB components
    '''
    import random
    base = random.randint(0, 255)
    
    # Generate small random differences to add to the base value to get low saturation
    diff1 = random.randint(-20, 20)
    diff2 = random.randint(-20, 20)
    
    # Make sure the RGB values are within the valid range (0-255)
    r = min(max(base + diff1, 0), 255)
    g = min(max(base + diff2, 0), 255)
    b = min(max(base - diff1 - diff2, 0), 255)

    return "#{:02x}{:02x}{:02x}".format(r, g, b)

def masked_smooth(x, mask, radius=50):
    smoothed_x = []
    for i in range(len(x)):
        window_x = x[max(0, i - radius):min(i + radius, len(x))]
        window_mask = mask[max(0, i - radius):min(i + radius, len(x))]
        smoothed_x.append(np.mean(window_x[window_mask == 0]))
    return smoothed_x

def get_unibind(chr: str, start: int, end: int) -> pd.DataFrame:
    """
    Returns a DataFrame of Unibind TFBSs in the given genomic region. Columns are 'chrom', 'chromStart', 'chromEnd', 'strand', 'tf', 'itemRgb'.
    """
    COL_NAMES = ['chrom', 'chromStart', 'chromEnd', 'strand', 'tf', 'itemRgb']
    df_iter = pd.read_csv('output/tfs/hg38_K562_TFBSs.bed.gz', sep='\t', names=COL_NAMES, iterator=True, chunksize=10000)
    df = pd.concat(
        itertools.chain(
            map(
                lambda _df: _df[(_df['chrom'] == chr) & (_df['chromEnd'] >= start) & (_df['chromStart'] <= end)],
                tqdm(df_iter, total=451, leave=False, desc='Loading TFBSs')),
            [pd.read_csv('data/tfs_HBG.tsv', sep='\t', usecols=COL_NAMES)], # manually add stuff
        ))
    assert all(~df['itemRgb'].isna()), "itemRgb column contains empty entries"
    df = df.drop_duplicates()
    # combine overlapping TFs
    G = nx.Graph()
    for i, row in df.iterrows():
        G.add_node(i, **row)
    for i, j in itertools.combinations(G.nodes, 2):
        if G.nodes[i]['chrom'] == G.nodes[j]['chrom'] and G.nodes[i]['chromEnd'] >= G.nodes[j]['chromStart'] and G.nodes[i]['chromStart'] <= G.nodes[j]['chromEnd'] and G.nodes[i]['tf'] == G.nodes[j]['tf']:
            G.add_edge(i, j)
    df_deduped = []
    for component in nx.connected_components(G):
        tfs_to_merge = df.loc[list(component)]
        deduped_row = tfs_to_merge.iloc[0].copy()
        deduped_row['chromStart'] = min(tfs_to_merge['chromStart'])
        deduped_row['chromEnd'] = max(tfs_to_merge['chromEnd'])
        if tfs_to_merge['strand'].nunique() == 1:
            deduped_row['strand'] = tfs_to_merge['strand'].iloc[0]
        else:
            deduped_row['strand'] = '.' # both strands
        df_deduped.append(deduped_row)
    unibind_tfs = pd.DataFrame(df_deduped) if df_deduped else pd.DataFrame(columns=['chrom', 'chromStart', 'chromEnd', 'strand', 'tf', 'itemRgb'])
    unibind_tfs['color'] = unibind_tfs['itemRgb'].apply(lambda x: tuple(int(y) / 255 for y in x.split(',')))
    return unibind_tfs

def get_atac_peaks(chr: str, start: int, end: int) -> list[tuple[int, int]]:
    import pyBigWig
    peaks = pyBigWig.open("https://www.encodeproject.org/files/ENCFF086JCJ/@@download/ENCFF086JCJ.bigBed").entries(chr, start, end) or []
    peaks = list(sorted(set((peak[0] - start, peak[1] - start) for peak in peaks)))
    return peaks


def diff_DddA_helper(
    ddda_data, fg_read_inds, bg_read_inds,
    locus, down_sample_n=10000, use_DAD=False, dedup_thresh_factor = None):
    '''
    Helper function for calculating differential DddA edits given a foreground and background set of reads
    '''

    # Get strandedness of each read
    strands = ddda_data.read_strands[locus]
    strand_read_inds = {
        "C_to_T" : np.where(strands == 0)[0],
        "G_to_A" : np.where(strands == 1)[0]
    }

    results = {}

    ###################################
    # Down-sampling and deduplication #
    ###################################
    
    # Down-sample the number of reads
    min_num = min(len(fg_read_inds), len(bg_read_inds), down_sample_n)
    if min_num < 5:
        return None
    print(min_num)
    fg_read_inds = np.random.choice(fg_read_inds, min_num, replace=False)
    bg_read_inds = np.random.choice(bg_read_inds, min_num, replace=False)
    
    # De-duplicate reads
    fg_read_ids = ddda_data.dedup_reads(
        locus = locus, 
        read_ids = np.array(ddda_data.read_ids[locus])[fg_read_inds],
        threshold_factor = dedup_thresh_factor or 0.01,
    )
    bg_read_ids = ddda_data.dedup_reads(
        locus = locus, 
        read_ids = np.array(ddda_data.read_ids[locus])[bg_read_inds],
        threshold_factor = dedup_thresh_factor or 0.01,
    )
    
    # For each read ID, get its index in the full read ID list
    locus_ids = ddda_data.read_ids[locus]
    locus_id_dict = dict(zip(locus_ids, np.arange(len(locus_ids))))
    fg_read_inds = np.array([*map(locus_id_dict.get, fg_read_ids)])
    bg_read_inds = np.array([*map(locus_id_dict.get, bg_read_ids)])

    #################################################################
    # Calculate DddA editing rate for ABE edited and unedited reads #
    #################################################################
    
    for strand in  ["C_to_T", "G_to_A"]:

        # Calculate average DddA editing rate on ABE edited reads for the current selected strand
        fg_read_inds_stranded = np.intersect1d(fg_read_inds, strand_read_inds[strand])
        print("len(fg_read_inds_stranded", len(fg_read_inds_stranded))
        if len(fg_read_inds_stranded) < 1:
            return None
        if not use_DAD:
            # Use raw edit rate
            fg_DddA_edits = np.array(np.mean(ddda_data.edit_dict[locus][fg_read_inds_stranded, :], axis=0))
        else:
            # Use HMM to convert raw edit data to DddA accessible domain (DADs)
            fg_DddA_edits = call_dads_hmm_bias(
                edits=ddda_data.edit_dict[locus][fg_read_inds_stranded, :].toarray(),
                ref_seq=ddda_data.ref_seq_dict[locus],
                strands=ddda_data.read_strands[locus][fg_read_inds_stranded]
            )

        # Calculate average DddA editing rate on ABE unedited reads for the current selected strand
        bg_read_inds_stranded = np.intersect1d(bg_read_inds, strand_read_inds[strand])
        print("len(bg_read_inds_stranded)", len(bg_read_inds_stranded))
        if len(bg_read_inds_stranded) < 1:
            return None
        if not use_DAD:
            # Use raw edit rate
            bg_DddA_edits = np.array(np.mean(ddda_data.edit_dict[locus][bg_read_inds_stranded, :], axis=0))
        else:
            # Use HMM to convert raw edit data to DddA accessible domain (DADs)
            bg_DddA_edits = call_dads_hmm_bias(
                edits=ddda_data.edit_dict[locus][bg_read_inds_stranded, :].toarray(),
                ref_seq=ddda_data.ref_seq_dict[locus],
                strands=ddda_data.read_strands[locus][bg_read_inds_stranded]
            )
            
        strand_results = {
            "fg_read_inds" : fg_read_inds_stranded,
            "bg_read_inds" : bg_read_inds_stranded,
            "fg_DddA_edits" : fg_DddA_edits,
            "bg_DddA_edits" : bg_DddA_edits
        }

        results[strand] = strand_results
    return results

def export_to_tsv(
        ddda_data, export_DddA_edit=True, export_del=False, 
        export_ABE_edit=False, export_strand=True, export_dir=None,
        overwrite=False
    ):
    '''
    Export key data slots in ddda_data to tsv files. This saves the read-by-position matrices such as the DddA
    edit matrix to a IJV COO matrix format (3 columns)
    '''

    for locus in ddda_data.ref_seq_dict:
    
        if export_dir==None:
            export_dir=ddda_data.working_dir
        if not os.path.exists(export_dir):
            os.system("mkdir -p " + export_dir)
        
        if export_DddA_edit:
            print("Exporting DddA edits for dataset " + ddda_data.ID + " at locus " + locus)
            save_path = os.path.join(export_dir, ddda_data.ID + "_" + locus + "_DddA_edit_coo_mtx.tsv")
            if not os.path.exists(save_path) or overwrite:
                edit_coo_mtx = ddda_data.edit_dict[locus].tocoo()
                edit_df = pd.DataFrame({"i":edit_coo_mtx.row, "j":edit_coo_mtx.col, "v":edit_coo_mtx.data})
                edit_df.to_csv(save_path, sep="\t", index=None)
                
        if export_del:
            print("Exporting deletions for dataset " + ddda_data.ID + " at locus " + locus)
            save_path = os.path.join(export_dir, ddda_data.ID + "_" + locus + "_del_coo_mtx.tsv")
            if not os.path.exists(save_path) or overwrite:
                del_coo_mtx = ddda_data.del_dict[locus].tocoo()
                del_df = pd.DataFrame({"i":del_coo_mtx.row, "j":del_coo_mtx.col, "v":del_coo_mtx.data})
                del_df.to_csv(save_path, sep="\t", index=None)
                
        if export_ABE_edit:
            print("Exporting ABE edits for dataset " + ddda_data.ID + " at locus " + locus)
            save_path = os.path.join(export_dir, ddda_data.ID + "_" + locus + "_ABE_edit_coo_mtx.tsv")
            if not os.path.exists(save_path) or overwrite:
                ABE_edit_coo_mtx = ddda_data.ABE_edit_dict[locus].tocoo()
                ABE_edit_df = pd.DataFrame({"i":ABE_edit_coo_mtx.row, "j":ABE_edit_coo_mtx.col, "v":ABE_edit_coo_mtx.data})
                ABE_edit_df.to_csv(save_path, sep="\t", index=None)
                
        if export_strand:
            print("Exporting aligned reads for dataset " + ddda_data.ID + " at locus " + locus)
            save_path = os.path.join(export_dir, ddda_data.ID + "_" + locus + "_read_strands.tsv")
            if not os.path.exists(save_path) or overwrite:
                strand_df = pd.DataFrame({"strand":ddda_data.read_strands[locus]})
                strand_df.to_csv(save_path, sep="\t", index=None)

def smooth(x: np.ndarray, window: int = 100, how: Literal['mean', 'sum'] = 'mean') -> np.ndarray:
    if how == 'mean':
        fill_value = 1 / window
    elif how == 'sum':
        fill_value = 1
    else:
        raise ValueError("Invalid how")

    if len(x.shape) == 1:
        return scipy.signal.convolve(x, np.full(window, fill_value), mode='same')
    elif len(x.shape) == 2:
        return scipy.signal.convolve2d(x, np.full((1, window), fill_value), mode='same')
    else:
        raise ValueError("Invalid shape")

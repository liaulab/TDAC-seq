from Bio import Seq
from .utils import parse_cigar, ref_bp_consumed, diff_DddA_helper
import pandas as pd
from tqdm import tqdm
import scipy.sparse
import numpy as np
from .ddda_dataset import ddda_dataset

def find_cutsite(ref_seq: str, guide: str) -> int | None:
    '''
    Zero-indexed cutsite position in ref_seq for guide. Cut is immediately before the base at the indicated position. Assume there's at most one cutsite. If guide is not found, return None.
    '''
    assert len(guide) == 20, "Guide should be 20bp long"
    # + strand
    if (cutsite := ref_seq.find(guide)) != -1:
        return cutsite + 17
    # - strand
    elif (cutsite := ref_seq.find(Seq.reverse_complement(guide))) != -1:
        return cutsite + 3
    return None

def find_guide_strand(ref_seq: str, guide: str) -> bool | None:
    '''
    Return True if guide is on the + strand, False if guide is on the - strand, and None if guide is not found.
    '''
    assert len(guide) == 20, "Guide should be 20bp long"
    if ref_seq.find(guide) != -1:
        return True
    elif ref_seq.find(Seq.reverse_complement(guide)) != -1:
        return False
    return None

def slice_cs_left(cs: str, pos: int, start: int = 0) -> str | None:
    '''
    Zero-indexed. Returns the cigar substring entries starting (and including) pos. Assuming input cigar string starts at reference position `start`
    '''
    if pos < start:
        return None
    cs_info = parse_cigar(cs)
    current_pos = start
    out = ""
    for cs_op, cs_arg in cs_info:
        consumed = ref_bp_consumed(cs_op, cs_arg)
        current_pos += consumed
        if current_pos <= pos:
            continue
        if current_pos - consumed < pos:
            # need to break it up
            if cs_op == ":":
                out += f":{current_pos - pos}"
            elif cs_op == "-":
                out += f"-{cs_arg[pos - current_pos:]}"
            elif cs_op == "*":
                continue
            else:
                raise ValueError("This should not happen") # insertion shouldn't consume reference bases
        else:
            out += f"{cs_op}{cs_arg}"
    if out == "":
        return None
    return out

def slice_cs_right(cs: str, pos: int, start: int = 0) -> str | None:
    '''
    Zero-indexed. Returns the cigar substring entries ending (but excluding) pos. Assuming input cigar string starts at reference position `start`. If cigar string doesn't cover the window completely, return None.
    '''
    if pos < start:
        return None
    cs_info = parse_cigar(cs)
    current_pos = start
    out = ""
    for cs_op, cs_arg in cs_info:
        consumed = ref_bp_consumed(cs_op, cs_arg)
        current_pos += consumed
        if current_pos > pos:
            if current_pos - consumed < pos:
                # need to break it up
                if cs_op == ":":
                    out += f":{pos - current_pos + consumed}"
                elif cs_op == "-":
                    out += f"-{cs_arg[:pos - current_pos + consumed]}"
                elif cs_op == "*":
                    continue
                else:
                    raise ValueError("This should not happen")
            break
        else:
            out += f"{cs_op}{cs_arg}"
    if current_pos < pos:
        return None
    if out == "":
        return None
    return out

def slice_cs(cs: str, window: slice, start: int = 0) -> str | None:
    '''
    Zero-indexed. Returns the cigar substring entries in the window. Assuming input cigar string starts at reference position `start`. If cigar string doesn't cover the window completely, return None.
    '''
    out = slice_cs_right(cs, window.stop, start)
    if out is None:
        return None
    # slice right first so that start doesn't change
    return slice_cs_left(out, window.start, start)

def clean_cs(cs: str, cutsite: int) -> str:
    '''
    Only keep cs differences that are adjacent to the cutsite.
    '''
    right_border = cutsite
    while True:
        right_border += 1
        if (sofar := slice_cs_right(cs, right_border)) is None or parse_cigar(sofar)[-1][0] == ":":
            right_border -= 1
            break
    left_border = cutsite
    while True:
        left_border -= 1
        if (sofar := slice_cs_left(cs, left_border)) is None or parse_cigar(sofar)[0][0] == ":":
            left_border += 1
            break
    total_len = sum(ref_bp_consumed(cs_op, cs_arg) for cs_op, cs_arg in parse_cigar(cs))
    if right_border == cutsite and left_border == cutsite:
        return f":{total_len}"
    out = ""
    if left_border > 0:
        out += f":{left_border}"
    out += slice_cs(cs, slice(left_border, right_border))
    if right_border < total_len:
        out += f":{total_len - right_border}"
    return out

abe_ratio_thresh = 2

def find_editable_pos(ref_seq: str, guides: pd.DataFrame, editing_window: tuple[int, int] = (-2, 12)) -> set[int]:
    '''
    editing window is zero-indexed, inclusive. Canonical window would be 3, 7.
    '''
    # Find sgRNA positions in the target locus
    editable_pos = []
    for i, c in enumerate(ref_seq):
        if c == 'A':
            _guides = guides['cutsite'].loc[guides['strand']]
            _abe_window = (editing_window[0] - 17, editing_window[1] - 17)
        elif c == 'T':
            _guides = guides['cutsite'].loc[~guides['strand']]
            _abe_window = (17 - editing_window[1], 17 - editing_window[0])
        else:
            continue
        if any((_guides + _abe_window[0] <= i) & (i <= _guides + _abe_window[1])):
            editable_pos.append(i)
    return set(editable_pos)

def filter_abe_edits(num_edits: pd.DataFrame, abe: scipy.sparse.sparray, window: list[slice], editable_pos: set[int], mask_outside_window: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    _edits = num_edits.copy()
    abe_edits = abe.tolil().rows
    for i, _abe_edits in tqdm(enumerate(abe_edits), total=abe_edits.shape[0], desc="Filtering ABE edits"):
        abe_edits[i] = [x for x in _abe_edits if any(_window.start <= x < _window.stop for _window in window) and x in editable_pos]
        if len(abe_edits[i]) == 0 and len(_abe_edits) > 0:
            abe_edits[i] = [-1]
    _edits['abe_edits'] = list(map(tuple, abe_edits))
    if mask_outside_window:
        # mask snps, TODO: should actually be more precise about this
        _edits['abe_edits'] = _edits['abe_edits'].map(lambda x: x if x != (-1,) else tuple())
    scores_agg = _edits.value_counts('abe_edits').to_frame()
    scores_agg['freq'] = scores_agg['count'] / scores_agg['count'].sum()

    return _edits, scores_agg

def aggregate_reads_with_abe_genotype(edits: scipy.sparse.sparray, abe: scipy.sparse.sparray, genotypes_of_interest: list[tuple[int]], editable_pos: list[int], ddda_thresh: int | None = None, dedup_thresh: int | None = None, ddda_data: ddda_dataset = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Order of genotypes_of_interest matters because that's the order of the rows in the heatmap.
    Order of editable_pos doesn't matter.
    Exclude reads with less than or equal to ddda_thresh DddA edits. If None, keep all reads.
    Run dedup if dedup_thresh is not None or 0. Will set to None if 0.
    ddda_data is only required if dedup_thresh is not None.
    """
    assert edits.shape == abe.shape
    assert isinstance(editable_pos, list)
    if not isinstance(edits, scipy.sparse.csr_matrix):
        edits = edits.tocsr()
    if not isinstance(abe, scipy.sparse.csc_matrix):
        abe = abe.tocsc()
    if dedup_thresh == 0:
        dedup_thresh = None

    heatmap = np.full((len(genotypes_of_interest), edits.shape[1]), np.nan)
    counts = np.zeros(len(genotypes_of_interest), dtype=int)

    # exclude reads with low DddA editing
    if ddda_thresh is not None:
        ddda_high = edits.sum(axis=1) > ddda_thresh
    else:
        ddda_high = np.ones(edits.shape[0], dtype=bool)
    
    # Calculate wildtype
    wt_mask = ddda_high & (abe[:, editable_pos].sum(axis=1) == 0)
    if dedup_thresh is None:
        wt = edits[wt_mask].mean(axis=0)
    else:
        assert ddda_data is not None
        if len(ddda_data.region_dict) != 1:
            raise NotImplementedError("Only one region is supported for now")
        locus = list(ddda_data.ref_seq_dict.keys())[0]
        wt = np.zeros(edits.shape[1])
        wt_reads_index = wt_mask.nonzero()[0]
    
    # Go through genotypes
    for i, genotype in tqdm(enumerate(genotypes_of_interest), total=len(genotypes_of_interest), desc="Aggregating reads by genotype"):
        # DddA editing is high
        mask = ddda_high.copy()
        # has desired ABE edit(s)
        mask &= abe[:, genotype].sum(axis=1) == len(genotype)
        # all other editable positions are unedited
        mask &= abe[:, editable_pos].sum(axis=1) == len(genotype)
        counts[i] = mask.sum()
        if counts[i] == 0:
            continue

        # aggregate reads
        if dedup_thresh is None:
            heatmap[i, :] = edits[mask].mean(axis=0)
        else:
            ref_seq = ddda_data.ref_seq_dict[locus]
            CG_mask = np.arange(len(ref_seq))[ddda_data.start_gap_threshold:-ddda_data.end_gap_threshold]
            C_pos = np.array([i for i in CG_mask if ref_seq[i] == "C"])
            G_pos = np.array([i for i in CG_mask if ref_seq[i] == "G"])
            dedup_thresh_factor = dedup_thresh / (len(C_pos) + len(G_pos))

            A_pos_results = diff_DddA_helper(
                ddda_data,
                mask.nonzero()[0],
                wt_reads_index,
                locus,
                down_sample_n=5000,
                dedup_thresh_factor=dedup_thresh_factor,
            )
            if A_pos_results is None:
                continue
            # Calculate effect of sgRNA on DddA edits
            CT_ABE_edited_DddA_edits = A_pos_results["C_to_T"]["fg_DddA_edits"]
            CT_ABE_unedited_DddA_edits = A_pos_results["C_to_T"]["bg_DddA_edits"]
            GA_ABE_edited_DddA_edits = A_pos_results["G_to_A"]["fg_DddA_edits"]
            GA_ABE_unedited_DddA_edits = A_pos_results["G_to_A"]["bg_DddA_edits"]
            heatmap[i, :] = (CT_ABE_edited_DddA_edits - CT_ABE_unedited_DddA_edits + GA_ABE_edited_DddA_edits - GA_ABE_unedited_DddA_edits) / 2

    return heatmap, wt, counts, ddda_high.sum()

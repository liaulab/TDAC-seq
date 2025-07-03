import numpy as np
from ..utils import masked_smooth
from tqdm import tqdm

def calculate_guide_effects(ref_seq, sgRNA_results_sub, smooth=False, is_abe: bool = False):
    read_count = np.zeros(len(sgRNA_results_sub), dtype=int)
    effect_mtx = np.empty((len(sgRNA_results_sub), len(ref_seq)), dtype=float)
    agg_mtx = np.empty((len(sgRNA_results_sub), len(ref_seq)), dtype=float)
    agg_mtx_wt = np.empty((len(sgRNA_results_sub), len(ref_seq)), dtype=float)
    cut_sites = []
    for sgRNA_ind, (sgRNA_pos, sgRNA_results_sub_sub) in enumerate(tqdm(sgRNA_results_sub.items(), total=len(sgRNA_results_sub), desc="Calculating guide effects")):

        # Calculate effect of sgRNA on DddA edits
        CT_del_edits = sgRNA_results_sub_sub["C_to_T"]["del_edits"]
        CT_undel_edits = sgRNA_results_sub_sub["C_to_T"]["undel_edits"]
        GA_del_edits = sgRNA_results_sub_sub["G_to_A"]["del_edits"]
        GA_undel_edits = sgRNA_results_sub_sub["G_to_A"]["undel_edits"]
        diff_edits = (CT_del_edits - CT_undel_edits + GA_del_edits - GA_undel_edits) / 2
        del_edits = (CT_del_edits + GA_del_edits) / 2
        undel_edits = (CT_undel_edits + GA_undel_edits) / 2

        # Smoothing while masking deleted regions
        mask = np.zeros(len(diff_edits), dtype=bool)
        if not is_abe:
            if isinstance(sgRNA_pos, int):
                mask[max(0, sgRNA_pos - 20):min(sgRNA_pos + 20, len(mask))] = True
            elif isinstance(sgRNA_pos, tuple) and len(sgRNA_pos) == 2:
                mask[sgRNA_pos[0]:sgRNA_pos[0] + sgRNA_pos[1]] = True
        if smooth:
            diff_edits = masked_smooth(x=diff_edits, mask=mask, radius=50)
            del_edits = masked_smooth(x=del_edits, mask=mask, radius=50)
            undel_edits = masked_smooth(x=undel_edits, mask=mask, radius=50)
        #diff_edits = np.convolve(diff_edits, np.ones(100), mode="same") / 100
        effect_mtx[sgRNA_ind] = diff_edits
        agg_mtx[sgRNA_ind] = del_edits
        agg_mtx_wt[sgRNA_ind] = undel_edits

        # Also record read depth for the current sgRNA (save the smaller value between the replicates)
        n_reads = sgRNA_results_sub_sub["C_to_T"]["n_reads"] + \
            sgRNA_results_sub_sub["G_to_A"]["n_reads"]
        read_count[sgRNA_ind] = n_reads

        cut_sites.append(sgRNA_pos)

    return effect_mtx, agg_mtx, agg_mtx_wt, read_count, cut_sites

def calculate_impact_score(effect_mtx):
    return np.sqrt(np.power(effect_mtx, 2).sum(axis=1))

import numpy as np
from tqdm import tqdm
import multiprocessing as mp
import tempfile
from umi_tools import UMIClusterer
import logging
import subprocess
import scipy.sparse

def dedup_read_inds(edits: np.ndarray, read_inds: np.ndarray[int], threshold: int, batch_size: int | None = None, dup_rate_tolerance: float = 0.05) -> np.ndarray[int]:
    '''
    If batch_size is not None, divide and conquer to dedup, stopping when the dedup rate is within the tolerance. So it's an approximate dedup.
    '''
    if batch_size is not None and len(read_inds) > batch_size:
        # divide
        np.random.seed(2020)
        split = np.random.rand(len(read_inds)) > 0.5
        batch_1 = read_inds[split]
        batch_2 = read_inds[~split]
        # conquer
        batch_1_kept = dedup_read_inds(edits, batch_1, threshold, batch_size, dup_rate_tolerance)
        batch_2_kept = dedup_read_inds(edits, batch_2, threshold, batch_size, dup_rate_tolerance)
        # merge
        merged_inds = np.concatenate([batch_1_kept, batch_2_kept])
        if len(batch_1_kept) / len(batch_1) > 1 - dup_rate_tolerance and len(batch_2_kept) / len(batch_2) > 1 - dup_rate_tolerance:
            return merged_inds
        else:
            return dedup_read_inds(edits, merged_inds, threshold, batch_size, dup_rate_tolerance)

    with tempfile.NamedTemporaryFile("r+t") as f_grouped, tempfile.NamedTemporaryFile("r+t") as f_grouped_sorted:

        umi_groups = UmiIterator(edits, read_inds, threshold)

        for read_ind, umi_group in umi_groups:
            if read_ind is None:
                break
            f_grouped.write(f"{read_ind}\t{umi_group}\n")
        f_grouped.flush()
        f_grouped.seek(0)

        # Sort by umi group
        logging.info(f"Sorting reads by UMI group")
        subprocess.run(["sort", "-k2,2", "-o", f_grouped_sorted.name, f_grouped.name], check=True)

        # Read sorted file and keep the read with the highest umi count in each group
        logging.info("Deduping UMI groups")
        group_kept_read_ind = {}
        group_max_umi_count = {} # Maintains the largest umi count for each group
        for line in tqdm(f_grouped_sorted):
            read_ind, group = line.strip().split("\t")
            read_ind = int(read_ind)
            group = int(group)
            read_umi_count = umi_groups.umi_hist[umi_groups.umi_dict[read_ind]]
            if group not in group_max_umi_count or group_max_umi_count[group] < read_umi_count:
                group_max_umi_count[group] = read_umi_count
                group_kept_read_ind[group] = read_ind
        return np.array(list(group_kept_read_ind.values()))

class UmiIterator:
    """
    Yields the read index and the UMI group id for each read. After all reads are processed, yields (None, None) as stop signal. Then yields the umi histogram and umi dictionary.
    """

    def __init__(self, edits: scipy.sparse.sparray, read_inds: np.ndarray[int], threshold: int):
        self.f = tempfile.NamedTemporaryFile("r+t")
        self.umi_hist: dict[bytes, int] = {}
        self.umi_dict = {}

        # During dedup, only use bases with > 0 edits
        edit_count = np.sum(edits, axis=0)
        mask = edit_count > 0

        # Extract umis and build histogram
        logging.info("Building UMI histogram")
        for read_ind in tqdm(read_inds):
            umi = bytes([49 if edited else 48 for edited in edits[[read_ind], mask]]) # ASCII code for "1" and "0" respectively
            self.f.write(f"{read_ind}\t{umi.decode()}\n")
            self.umi_dict[read_ind] = umi
            self.umi_hist[umi] = self.umi_hist.get(umi, 0) + 1
        self.f.flush()
        self.f.seek(0)

        # Call umi groups
        logging.info("Calling UMI groups")
        clusterer = UMIClusterer()
        clusters = clusterer(self.umi_hist, threshold=threshold)
        logging.info(f"Found {len(clusters)} unique reads out of {len(read_inds)}.")
        self.cluster_dict = {raw_umi: i for i, cluster in enumerate(clusters) for raw_umi in cluster}
        logging.info("Done calling UMI groups")
    
    def __iter__(self):
        return self
    
    def __next__(self):
        # Assign umi group id to each read
        line = self.f.readline()
        if not line:
            self.f.close()
            raise StopIteration()

        read_ind, umi = line.strip().split("\t")
        return int(read_ind), self.cluster_dict[umi.encode()]

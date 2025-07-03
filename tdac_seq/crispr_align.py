import pandas as pd
import itertools
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
import numpy as np
import matplotlib.pyplot as plt
plt.set_loglevel("WARNING")
from tqdm import tqdm
import mappy
import multiprocessing as mp
import itertools
import tempfile
from tdac_seq.utils import get_fastq_iter, parse_cigar, ref_bp_consumed
from typing import Iterable
import re
from collections.abc import Container

def align_helper(args):
    return _align_helper(*args)

def _align_helper(reads_batch: Iterable[tuple[str, str, str]], ref_seq: str, start_gap_threshold: int, end_gap_threshold: int, ref_fasta_fname: str, peaks: list[tuple[int, int]], cutsites: Container[int]):

    aligner = mappy.Aligner(ref_fasta_fname, preset="map-ont")

    out = ""

    for read_id, read_seq, _ in reads_batch:

        # Align the current read
        alignments = [a for a in aligner.map(read_seq, cs=True)]
        
        # Remove multi-mappers and unmapped reads
        if len(alignments) == 0 or len(alignments) > 1:
            continue
        else:
            alignment = alignments[0]

        # Remove reads with large gaps at the beginning or end
        if alignment.r_st >= start_gap_threshold:
            continue
        if alignment.r_en <= len(ref_seq) - end_gap_threshold:
            continue
        
        cur_pos = alignment.r_st
        del_sofar = []
        num_ct_edits = [0] * len(peaks)
        num_ga_edits = [0] * len(peaks)
        for cs_op, cs_arg in parse_cigar(alignment.cs):
            if cs_op == "-" and len(cs_arg) >= 5:
                del_sofar.append((cur_pos, len(cs_arg)))
            elif cs_op == "*":
                edit_from, edit_to = cs_arg
                if edit_from == "c" and edit_to == "t":
                    for i, (start, end) in enumerate(peaks):
                        if cur_pos >= start and cur_pos < end:
                            num_ct_edits[i] += 1
                elif edit_from == "g" and edit_to == "a":
                    for i, (start, end) in enumerate(peaks):
                        if cur_pos >= start and cur_pos < end:
                            num_ga_edits[i] += 1
            cur_pos += ref_bp_consumed(cs_op, cs_arg)
        if len(del_sofar) == 0:
            del_start = 0
            del_len = 0
        elif len(del_sofar) > 1:
            continue
        else:
            del_start, del_len = del_sofar[0]
        
        edits = itertools.chain.from_iterable(zip(num_ct_edits, num_ga_edits))

        out += f"{read_id}\t{del_start}\t{del_len}\t{'\t'.join(map(str, edits))}\n"

    return out

def _write_ref_fasta(locus, ref_seq, guides, ref_fasta):
    ref_fasta.write(f">{locus}\n{ref_seq}\n")
    ref_fasta.flush()

def align_reads(aln_fname, locus, ref_seq, fastq_file, total_reads, start_gap_threshold, end_gap_threshold, peaks: list[tuple[int, int]], guides: pd.DataFrame):
    with open(aln_fname, "wt") as aln_file, mp.Pool(mp.cpu_count()) as pool, tempfile.NamedTemporaryFile("w+t") as ref_fasta:
        _write_ref_fasta(locus, ref_seq, guides, ref_fasta)

        edits = "\t".join(f"num_ct_edits_{peak_start}-{peak_end}\tnum_ga_edits_{peak_start}-{peak_end}" for peak_start, peak_end in peaks)
        aln_file.write(f"read_id\tdel_start\tdel_len\t{edits}\n")

        for out in pool.imap(
                align_helper,
                ((reads_batch, ref_seq, start_gap_threshold, end_gap_threshold, ref_fasta.name, peaks, guides['cutsite']) for reads_batch in itertools.batched(tqdm(get_fastq_iter(fastq_file), total=total_reads), 10000)),
                ):
            if out is not None:
                aln_file.write(out)

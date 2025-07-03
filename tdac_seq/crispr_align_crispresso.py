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
from tdac_seq.utils import get_fastq_iter, parse_cigar, ref_bp_consumed, cigar_to_seq
from tdac_seq.crispr import slice_cs
from typing import Iterable
import re
from collections.abc import Container
import subprocess
import gzip
import os

def align_helper(args):
    return _align_helper(*args)

def _align_helper(reads_batch: Iterable[tuple[str, str, str]], ref_seq: str, start_gap_threshold: int, end_gap_threshold: int, peaks: list[tuple[int, int]], window: slice) -> tuple[dict[str, tuple], str]:

    aligner = mappy.Aligner(seq=ref_seq, preset="map-ont")

    out = dict() # read_id -> (num_ct_edits, num_ga_edits) for each peak
    out2 = ""

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
        num_ct_edits = [0] * len(peaks)
        num_ga_edits = [0] * len(peaks)
        for cs_op, cs_arg in parse_cigar(alignment.cs):
            if cs_op == "*":
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

        out[read_id] = tuple(itertools.chain.from_iterable(zip(num_ct_edits, num_ga_edits)))

        # Write the read to the sliced fastq file
        cs = slice_cs(alignment.cs, window=window, start=alignment.r_st)
        if cs is None:
            continue
        query_seq = cigar_to_seq(cs, ref_seq[window])
        if len(query_seq) == 0:
            continue
        out2 += f"@{read_id}\n{query_seq}\n+\n{'F' * len(query_seq)}\n"

    return out, out2

def align_reads(aln_fname, locus, ref_seq, fastq_file, total_reads, start_gap_threshold, end_gap_threshold, peaks: list[tuple[int, int]], guides: pd.DataFrame):
    hybrid_start = 4136
    hybrid_end = 9064
    buffer = 100
    window = slice(hybrid_start - buffer, hybrid_start + buffer)
    _ref_seq = ref_seq[:hybrid_start] + ref_seq[hybrid_end:]

    with mp.Pool(mp.cpu_count()) as pool, tempfile.NamedTemporaryFile("w+t") as fastq_for_crispresso, tempfile.TemporaryDirectory() as crispresso_dir:

        ddda_edits = dict()

        for out, out2 in pool.imap(
                align_helper,
                ((reads_batch, _ref_seq, start_gap_threshold, end_gap_threshold, peaks, window) for reads_batch in itertools.batched(tqdm(get_fastq_iter(fastq_file), total=total_reads), 10000)),
                ):
            fastq_for_crispresso.write(out2)
            ddda_edits.update(out)
        fastq_for_crispresso.flush()

        # Run CRISPResso
        subprocess.call([
            "CRISPResso",
            "--fastq_r1", fastq_for_crispresso.name,
            "--amplicon_seq", _ref_seq[window],
            "--amplicon_name", locus,
            "--guide_seq", ",".join(guides['spacer']),
            "--guide_name", ",".join(guides.index),
            "--suppress_report",
            "--suppress_plots",
            "--fastq_output",
            "--output_folder", crispresso_dir,
            ])

        # Parse CRISPResso output
        with open(aln_fname, "wt") as aln_file, gzip.open(os.path.join(crispresso_dir, f"CRISPResso_on_{os.path.basename(fastq_for_crispresso.name)}", "CRISPResso_output.fastq.gz"), "rt") as f:

            edits = "\t".join(f"num_ct_edits_{peak_start}-{peak_end}\tnum_ga_edits_{peak_start}-{peak_end}" for peak_start, peak_end in peaks)
            aln_file.write(f"read_id\tdel_start\tdel_len\t{edits}\n")

            for read_id, _, annotation, _ in itertools.batched(f, 4):
                if annotation.startswith('+ ALN=NA '):
                    continue
                if m := re.search(r" DEL=(\d+)\((\d+)\) ", annotation):
                    del_start = int(m.group(1)) + hybrid_start - buffer
                    del_len = int(m.group(2))
                elif " MODS=D0;" in annotation:
                    del_start = hybrid_start
                    del_len = 0
                else:
                    raise ValueError(annotation)
                if del_start <= hybrid_start and del_start + del_len >= hybrid_start:
                    del_len += hybrid_end - hybrid_start

                read_id = read_id.rstrip().lstrip("@>")

                aln_file.write(f'@{read_id}\t{del_start}\t{del_len}\t' + '\t'.join(map(str, ddda_edits[read_id])) + '\n')

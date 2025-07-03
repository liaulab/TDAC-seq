import pandas as pd
from tdac_seq.utils import get_fastq_iter, count_reads_in_fastq, cigar_to_seq
from tdac_seq.crispr import find_cutsite, find_guide_strand, slice_cs
from Bio import SeqIO
from tqdm import tqdm
import tempfile
import mappy
import os
import logging
import subprocess

guides = pd.read_csv('data/HJR244_sgRNA.txt', sep='\t', header=None, index_col=0)
guides.rename_axis(index='guide', inplace=True)
guides.rename(columns={1: 'spacer'}, inplace=True)
guides['spacer'] = guides['spacer'].str.upper()

ref_record = next(record for record in SeqIO.parse('data/ref.fa', 'fasta') if record.name == 'LCR_new')
ref = str(ref_record.seq.upper())

guides['cutsite'] = guides['spacer'].apply(lambda x: find_cutsite(ref, x))
guides['strand'] = pd.Categorical.from_codes(codes=guides['spacer'].apply(lambda x: find_guide_strand(ref, x)).astype(int), categories=['-', '+'])

window = slice(1322 - 100, 1652 + 100)
for data_id in ("HJR244_12_NoDddA", "HJR244_34_NoDddA"):
    fastq_fname = f'/n/holystore01/LABS/liau_lab/Users/heejinroh/HJR244/{data_id}/{data_id}_merged.fastq.gz'
    # Cas9, No DddA: /n/holystore01/LABS/liau_lab/Users/heejinroh/HJR244/HJR244_12_NoDddA/HJR244_12_NoDddA_merged.fastq.gz
    # ABE, No DddA: /n/holystore01/LABS/liau_lab/Users/heejinroh/HJR244/HJR244_34_NoDddA/HJR244_34_NoDddA_merged.fastq.gz

    working_dir = f"results/HJR244/{data_id}"
    os.makedirs(working_dir, exist_ok=True)
    sliced_fastq_fname = os.path.join(working_dir, f"{data_id}_sliced.fastq")
    if os.path.exists(sliced_fastq_fname):
        logging.info(f"Skipping minimap aligning {data_id} because {sliced_fastq_fname} already exists.")
    else:
        fastq_iter = get_fastq_iter(fastq_fname)
        with tempfile.NamedTemporaryFile('r+t') as ref_file, open(sliced_fastq_fname, 'wt') as sliced_fastq_file:
            SeqIO.write(ref_record, ref_file, 'fasta')
            ref_file.flush()
            aligner = mappy.Aligner(ref_file.name, preset='map-ont')
            stats = {'not aligned': 0, 'multi aligned': 0, 'aligned': 0, 'insufficient coverage': 0, 'huge deletion': 0}
            TOTAL_READS = count_reads_in_fastq(fastq_fname)
            for read_id, read_seq, read_qual in tqdm(fastq_iter, desc='Processing reads', total=TOTAL_READS):
                # Align the current read
                alignments = [a for a in aligner.map(read_seq, cs=True)]
                # Remove multi-mappers and unmapped reads
                if len(alignments) == 0:
                    stats['not aligned'] += 1
                    continue
                elif len(alignments) > 1:
                    stats['multi aligned'] += 1
                    continue
                alignment = alignments[0]
                # Retrieve the locus that this read is aligned to
                assert alignment.ctg == ref_record.name
                # Write the read to the sliced fastq file
                cs = slice_cs(alignment.cs, window=window, start=alignment.r_st)
                if cs is None:
                    stats['insufficient coverage'] += 1
                    continue
                query_seq = cigar_to_seq(cs, ref[window])
                if len(query_seq) == 0:
                    stats['huge deletion'] += 1
                    continue
                stats['aligned'] += 1
                sliced_fastq_file.write(f"@{read_id}\n{query_seq}\n+\n{'F' * len(query_seq)}\n")

    # Run CRISPResso
    if data_id == 'HJR244_12_NoDddA':
        args = []
    elif data_id == 'HJR244_34_NoDddA':
        args = ['--base_editor_output', '--quantification_window_size', '10', '--quantification_window_center', '-10', '--conversion_nuc_from', 'A', '--conversion_nuc_to', 'G']
    subprocess.call([
        'CRISPResso',
        '--fastq_r1', sliced_fastq_fname,
        '--amplicon_seq', ref[window],
        '--amplicon_name', f'LCR_new({window.start}-{window.stop})',
        '--guide_seq', ','.join(guides['spacer']),
        '--guide_name', ','.join(guides.index),
        '--place_report_in_output_folder',
        '--output_folder', working_dir,
        *args,
        ])

cores=16
genome=hg38
experiment=HJR269
picardPATH=/home/yanhu/utils/picard/picard.jar
cd /home/yanhu/data/DddA/HJR288/single_sgRNA_validation/

if [ ! -d fragments ]; then
  mkdir fragments
fi

cd bams

bam_files=*_bowtie2_rmdups.bam
for bam in $bam_files
do

    fname=`echo $bam | sed 's/_bowtie2_rmdups.bam//g'`
    echo "Processing $fname"

    if [ ! -f $fname.n_sort.rmdup.flt.bam ]; then
        echo "Name sorting bam file"
        samtools sort -n -@ $cores -o $fname.n_sort.rmdup.flt.bam ${fname}_bowtie2_rmdups.bam
    fi

    if [ ! -f ../fragments/$fname.frags.gz ]; then
        echo "Converting bam file to fragments file"
        bedtools bamtobed -i $fname.n_sort.rmdup.flt.bam -bedpe | \
        sed 's/_/\t/g' | \
        awk -v OFS="\t" -v ID=$fname '{if($9=="+"){print $1,$2+4,$6-5,ID}}' |\
        sort --parallel=$cores -S 40G  -k4,4 -k1,1 -k2,2n -k3,3n | \
        uniq -c | \
        awk -v OFS="\t" '{print $2, $3, $4, $5, $1}' | \
        gzip > $fname.frags.gz
        mv $fname.frags.gz ../fragments
    fi

done

# Peak Calling
if [ ! -f ../peaks.bed ]; then

    # Call peaks using MACS2
    mkdir peakCalling
    macs2 callpeak \
        -t *_bowtie2_rmdups.bam \
        -f BAM \
        -n $experiment \
        --outdir peakCalling \
        --keep-dup all \
        --nolambda --nomodel \
        --call-summits

    mv peakCalling ../
    cd ../

  # Filter and resize peaks
  Rscript /home/yanhu/DddA/DddA/tdac_seq/filterPeaks.R $genome

fi

# Merge fragments files
cd fragments
zcat *frags.gz | gzip > all.frags.tsv.gz
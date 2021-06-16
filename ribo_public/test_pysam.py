import pysam
import os

test_bam_path = "/data/users/wergillius/UTR_VAE/public_ribo/GSE52809/ribomap_out/alignment/SRR1039860_1_trimmed_transcript_Aligned.out.bam"

os.path.exists(test_bam_path)
bam = pysam.AlignmentFile(test_bam_path,"rb")

bam.head()
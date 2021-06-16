import os
import sys
import pysam
import re
import numpy as np
import json
from Bio import SeqIO
from tqdm import tqdm
from multiprocessing import Pool


def gtf_line_2_json(gtf_line):
    """from the last one extract info"""
    json = {}
    json['chromosome'] = gtf_line[0]
    json['hierarchy'] = gtf_line[2]
    json['start'] = int(gtf_line[3])
    json['end'] = int(gtf_line[4])
    last_field= gtf_line[-1]
    entry_ls=last_field.split(';')[:-1] # the last one is empty
    for entry in entry_ls:
        key,item = entry.strip().split(' ')
        json[key]=item.replace('"','')
    return json


# ======== read sam file =============
ribo_priasbl_sam = sys.argv[1]
assert os.path.exists(ribo_priasbl_sam)
assert ribo_priasbl_sam.endswith(".bam") , "BAM format required"
# the path to which output dict will save
ouputjson_path = ribo_priasbl_sam.replace(".bam","_ExonStart.json")
print("\nBAM file used : %s\n"%ribo_priasbl_sam)

# ======== read gtf file =============
if len(sys.argv) == 2:
    # default gtf : gencode primary assemble one
    gtf_path='/data/users/wergillius/reference/gencode/gencode.v37.annotation.gtf'
else:
    gtf_path = sys.argv[2]
print("GTF file used : %s\n"%gtf_path.split("/")[-1])

## this is to extract the exon lacation (chromosome.start,end) and id 
with open(gtf_path,'r') as f:
    gtf = []
    for line in tqdm(f,"parsing gtf..."):
        if 'exon' in line:
            # convert into json
            split_line = gtf_line_2_json(line.strip().split("\t"))
            gtf.append(split_line)
    f.close()

# ========== split whole gtf into 5 part ============
intercept=(len(gtf)//5)
sub_gtf_ls = [gtf[i*intercept:(i+1)*intercept] for i in range(4)]
sub_gtf_ls += gtf[4*intercept:]     # ensure complete spliting
del gtf  # not used later 


# =========== fecting reads start location that mapped to exon ==============

# prepare for multiprocessing
def exon_fetch(gtf):
    """
    function to parallerize
    for each exon in gtf, we use this loaction to fetch mapped reads and save their mapped loci
    """
    # read in  sub process specific bam
    ribo_sam = pysam.AlignmentFile(ribo_priasbl_sam,'rb',require_index=True)
    # output 
    exon_reads_dict = {}
    print("start sub process")
    for anno in gtf:
        if anno['hierarchy'] == 'exon':
            exon_id = anno['exon_id']
            # fecth read mapped in this region
            exon_iter  = ribo_sam.fetch(anno['chromosome'],int(anno['start']),int(anno['end']))
            # record all the 5' position of reads
            # align a list of position to the dict, index  by exon id
            exon_reads_dict[exon_id] = [read.reference_start for read in exon_iter]  

    return exon_reads_dict

# multi-processing 
print("\nstart fecthing .......")
# pool = Pool(5)
# result = pool.map(exon_fetch,sub_gtf_ls)

# aggerate subprocess out
out_dict={}
for process_out in result:
    out_dict.update(process_out)

# write to file
with open(ouputjson_path,'w') as f:
    json.dump(exon_reads_dict,f)
    f.close()
print("result saved to :  %s"%ouputjson_path)

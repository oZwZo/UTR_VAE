import os
import sys
from Bio import SeqIO
from tqdm import tqdm
import re

def reg_epx_transcript_id(last_field):
    matcher = r".*transcript_id \"(\S*)\"\;.*"
    tag_matcher = r".*tag \"(\S*)\"\;.*"
    t_id = re.match(matcher,last_field).group(1)
    try:
        tag = re.match(tag_matcher,last_field).group(1)
    except:
        tag = None
    return t_id,tag

def gtf_line_2_json(gtf_line):
    """from the last one extract info"""
    json = {}
    json['level'] = gtf_line[2]
    json['start'] = gtf_line[3]
    json['end'] = gtf_line[4]
    last_field= gtf_line[-1]
    entry_ls=last_field.split('";')[:-1] # the last one is empty
    for entry in entry_ls:
        key,item = entry.split(' "')
        json[key]=item
    return json

def gtf_2_df(gtf_path):
    """for a gtf file, convert to DF"""
    with open(gtf_path,'r') as f:
        gtf = [line.strip().split('\t') for line in f]
        f.close()
    
    gtf_json = [gtf_line_2_json(line) for line in gtf]
    
    return pd.json_normaliz(gtf_json)

# 1. load in gtf and full cdna fasta
homo_dir = '/data/users/wergillius/reference/homo_sapiens'
fa_path = os.path.join(homo_dir,'Homo_sapiens.GRCh38.cdna.all.fa')
assert os.path.join(fa_path)
gtf_path = os.path.join(homo_dir,'Homo_sapiens.GRCh38_protein_cdna.96.gtf')
assert os.path.join(gtf_path)
rrna_path = os.path.join(homo_dir,'Homo_sapiens.GRCh38.rRNA.gtf')
assert os.path.join(rrna_path)


# read in 
print("reading fasta file .... \n")
all_fa = list(SeqIO.parse(fa_path,'fasta'))

print("reading protein coding gene gtf file .... \n")
with open(gtf_path,'r') as f:
    Pcg_gtf = [line.strip().split('\t') for line in f]
    f.close()

print('constructing homo sapiens protein coding RNA...\n')
gtf_json = [gtf_line_2_json(line) for line in Pcg_gtf]
gtf_df = pd.normalize(gtf_json)
gtf_df.to_csv(os.path.join(homo_dir,'Homo_sapiens.GRCh38_protein_cdna.96gtf.csv'),index=False)

# 2. take out line that define transcript

transcripts = []
for gtf in tqdm(Pcg_gtf,desc='extracting transcript id'):
    if (gtf[2] == 'transcript'):
        last_field = gtf[-1] # contain a lot of annotaion
        transcript_id,tag = reg_epx_transcript_id(last_field)
        if tag =='basic':
            transcripts.append(transcript_id)
    
# 3. take protein coding SeqRecord 
keep_dict = {tid:1 for tid in transcripts}
def keep_transcript(tid):
    try:
        return keep_dict[tid]
    except:
        return 0

protein_coding_fa = []
for SeqRecord in tqdm(all_fa,desc='putting protein coding'):
    shortid = SeqRecord.id.split('.')[0]
    if keep_transcript(shortid):
        protein_coding_fa.append(SeqRecord)

SeqIO.write(protein_coding_fa,os.path.join(homo_dir,'Homo_sapiens.GRCh38_protein_cdna.96.fasta'),'fasta')
print("protein coding fasta saved to :%s"%os.path.join(homo_dir,'Homo_sapiens.GRCh38_protein_cdna.96.fasta'))        



# =====================================================
# 4. extract rRNA transcript id
print("reading rRNA gtf file .... \n")
with open(rrna_path,'r') as f:
    rrna_gtf = [line.strip().split('\t') for line in f]

rRNA_transcripts = []
for gtf in tqdm(rrna_gtf,desc='extracting rRNA id'):
    if gtf[2] == 'transcript':
        last_field = gtf[-1] # contain a lot of annotaion
        transcript_id = reg_epx_transcript_id(last_field)
        rRNA_transcripts.append(transcript_id)

keeprRNA_dict = {tid:1 for tid in rRNA_transcripts}
def keep_rRNA(tid):
    try:
        return keeprRNA_dict[tid]
    except:
        return 0

rRNA_fa = []
for SeqRecord in tqdm(all_fa,desc='putting protein coding'):
    shortid = SeqRecord.id.split('.')[0]
    if keep_rRNA(shortid):
        rRNA_fa.append(SeqRecord)

SeqIO.write(protein_coding_fa,os.path.join(homo_dir,'Homo_sapiens.GRCh38.rRNA.fasta'),'fasta')
print("\n protein coding fasta saved to :%s\n"%os.path.join(homo_dir,'Homo_sapiens.GRCh38.rRNA.fasta'))        


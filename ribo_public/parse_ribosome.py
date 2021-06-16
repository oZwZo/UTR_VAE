import os
import sys
import PATH
import pysam
import re
import numpy as np
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
# from models import reader
from sklearn import linear_model
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt
from torch.utils.data import random_split
from matplotlib import cm
import matplotlib
import seaborn as sns
import RNA

global P_M_order
global nt_order
global possible_codon

possible_codon = []
for x in ['A','G','C','T']:
    for y in ['A','G','C','T']:
        for z in ['A','G','C','T']:
            possible_codon.append(x+y+z)
                
P_M_order= ['Pyrimidine','Purine']
nt_order = ['T','C','A','G']

get_value = lambda x : x.split(':')[-1].strip()

def read_salmon_sf(path):
    DF = pd.read_table(path,sep='\t')[['Name','TPM','NumReads']]
    return DF

def read_ribomap_entry(entry):
    """
    process a ribomap output : stats , a typical entry of stats is:
         'refID: 1',
         'tid: NM_000015.3',
         'rabd: 2.13338',
         'tabd: 2.83927e-07',
         'te: 7.51385e+06',
    """
    ref_ID = get_value(entry[0])
    t_ID = get_value(entry[1])
    ribo_count = float(get_value(entry[2]))
    t_rabd = float(get_value(entry[3]))
    TE_score = float(get_value(entry[4]))
    
    attr_dict = {'REF_ID':ref_ID,'Transcipt_ID':t_ID,'Ribo_count':ribo_count,'Transcript_Rabd':t_rabd,'TE-score':TE_score}
    
    return attr_dict

def read_ribomap_stats(path):
    """
    read in a ribomap output stats file, convert to DF
    """
    
    # read the raw file
    with open(path,'r') as f:
        plain_txt_ls = f.readlines()
        f.close()
    # strip
    plain_txt_ls = [line.strip() for line in plain_txt_ls]
    
    #### process each entry ####
     
    structured_entry_ls=[]    # a dict list
    for i in range(0,len(plain_txt_ls),5):
        # each entry consists of 5 lines
        entry = plain_txt_ls[i:i+5]    
        
        # process with predefine funcion
        entry_dict = read_ribomap_entry(entry)
        structured_entry_ls.append(entry_dict)
    
    DF = pd.json_normalize(structured_entry_ls) #convert to DF
    
    return DF

def read_gb(SeqRecord):
    """
    read gbff term , extract info of interest
    ...input : `SeqRecord` object from Bio.SeqIO.parse("gbff")
    """
    cds_index = None
    for i in range(len(SeqRecord.features)):
        if SeqRecord.features[i].type == 'CDS':
            cds_index = i
    # ignore terms that have no CDS annotation
    if cds_index is None:
        raise AssertionError("Not CDS !!")
        
    gb_dict={}
    
    # the term to extract
    gb_dict["ID"] = SeqRecord.id
    gb_dict['Gene'] = SeqRecord.features[cds_index].qualifiers['gene'][0]
    
    # there are several term
    for xref in SeqRecord.features[cds_index].qualifiers['db_xref']:
        if 'GeneID:' in xref:
            with_id = xref 
    gb_dict['GeneID'] = with_id.split('GeneID:')[-1]
    
    gb_dict['protein_id'] = SeqRecord.features[cds_index].qualifiers['protein_id'][0]
    
    gb_dict['seq'] = SeqRecord.seq
    gb_dict['start'] = SeqRecord.features[cds_index].location.nofuzzy_start
    gb_dict['end'] = SeqRecord.features[cds_index].location.nofuzzy_end
    
    gb_dict['utr'] = SeqRecord.seq[:gb_dict['start']]
    
    
    return gb_dict

def gbff_DF(gbff_path):
    """
    read NCBI genome annotation `gbff` file, extract CDS info, and return in DF
    """
    gbff_gnrt = SeqIO.parse(gbff_path,'gb') 
    gbff_json = []
    
    print("\nGBFF file loaded, extracting Transcript info ..... \n\nThis will take around 2 min\n\n")
    
    for SeqRecord in tqdm(gbff_gnrt):
        try:
            # extract info
            gb_dict = read_gb(SeqRecord) 
            gbff_json.append(gb_dict)
        except AssertionError:
            # skip entry with Assertion : NO CDS
            continue
    
    gbff_df = pd.json_normalize(gbff_json)
    
    gbff_df.utr = gbff_df.utr.apply(str)
    gbff_df.seq = gbff_df.seq.apply(str)
    gbff_df.loc[:,'utr_len']=gbff_df.utr.apply(len)
    gbff_df.loc[:,'seq_len']=gbff_df.seq.apply(len)
    gbff_df.loc[:,'cds_len'] = gbff_df.end.values - gbff_df.start.values
    gbff_df.loc[:,'with_uAUG'] = gbff_df.utr.apply(lambda x: 'ATG' in x)
    return gbff_df

def kozak_consensus_test(seq,start,loose_criteria=False):
    """
    detect whether given sequencing meet kozak consensus
    ...seq : sequence to test
    ...start : location of start codon
    ...loose_criteira : if True, only 'G' at +4 position is allowed. \
                        if False, not "C" is considered as consense.
    """
    n_3 = seq[start-3]
    p_4 = seq[start+3]

    if loose_criteria:
        # whatever not 'C'
        p4_meet= (p_4 != 'C')
    else:
        # only 'G'
        p4_meet= (p_4 == 'G')

    # is -3 site a Purine ?
    n3isP = n_3 in ['A','G']

    return n3isP & p4_meet
    
def detect_uAUG_context(utr,loose_criteria=False):
    """
    detect uAUG , if there is , then we pipe to kozak context analysis
    ...input : the sequence to test
    ...loose_criteira : if True, only 'G' at +4 position is allowed. \
                        if False, not "C" is considered as consense.
        return: 'no_uAUG' , 'weak_uAUG' , 'strong_uAUG'
    """
    if "ATG" in utr:
        # location of ATG
        uAUG_site = utr.index("ATG")
        # we use utr sequence and give location of uAUG,
        is_kzk_con = kozak_consensus_test(utr,uAUG_site,loose_criteria)  
        
        uAUG_context = 'strong_uAUG' if (is_kzk_con==True) else 'weak_uAUG'
        return uAUG_context
    else:
        return "no_uAUG"
    
    

def seq_chunk_N_oh(seq,pad_to,trunc_len=50):
    """
    truncate the sequence and encode in one hot
    """
    if len(seq) > trunc_len:
        seq = seq[-1*trunc_len:]
    
    X = reader.one_hot(seq)
    X = torch.tensor(X)

    X_padded = reader.pad_zeros(X,pad_to)
    
    return X_padded.float()

def scatter_linearreg_plot(quanty,y,ax=None):
    linear_mod = linear_model.LinearRegression().fit(quanty.reshape(-1,1),y.reshape(-1,1))
    line_x = np.array([quanty.min(),quanty.max()])
    line_y = linear_mod.predict(line_x.reshape(-1,1))
    y_pred = linear_mod.predict(quanty.reshape(-1,1))
    r2 = r2_score(y,y_pred)
    
    fit_line="y=%.2fx+%.2f"%(linear_mod.coef_,linear_mod.intercept_)
    
    if ax is None:
        fig = plt.figure(figsize=(6,4))
        ax = fig.gca()
    
    ax.scatter(quanty,y,s=5,alpha=0.2,label=r'$R^2$=%.3f'%r2)
    ax.plot(line_x,line_x,'-.',color='orange',alpha=0.5,label='y=x')
    ax.plot(line_x,line_y,'--',color='black',alpha=0.5,label=fit_line)
    ax.legend()


class GSE65778_dataset(Dataset):
    
    def __init__(self,DF,pad_to,trunc_len=50,seq_col='utr',value_col='TE_count'):
        """
        Dataset to trancate sequence and return in one-hot encoding way
        `dataset(DF,pad_to,trunc_len=50,seq_col='utr')`
        ...DF: the dataframe contain sequence and its meta-info
        ...pad_to: final size of the output tensor
        ...trunc_len: maximum sequence to retain. number of nt preceding AUG
        ...seq_col : which col of the DF contain sequence to convert
        """
        self.df = DF
        self.pad_to = pad_to
        self.trunc_len =trunc_len
        
        # X and Y
        self.seqs = DF.loc[:,seq_col].values
        self.Y = DF.loc[:,value_col].values
        
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self,i):
        seq = self.seqs[i]
        x_padded = seq_chunk_N_oh(seq,self.pad_to,self.trunc_len)
        y = self.Y[i]
        return x_padded,i
        
def zwz_cor_m(matrix,axis):
    """
    calculate the covariance matrix 
    where the axis parameter denote which axis is the feature axis 
    """
    
    m,n = np.shape(matrix)
    if axis == 0:
        mean_M = np.mean(matrix,1-axis).reshape(m,1)
        norm_M = matrix-mean_M
        COV_M = np.dot(norm_M,np.transpose(norm_M))/n
        std_M = np.std(matrix,1-axis).reshape(m,1)
        db_std_M = std_M@std_M.T
        COR_M = COV_M / db_std_M
    else:
        mean_M = np.mean(matrix,1-axis).reshape(1,n)
        norm_M = matrix-mean_M
        COV_M = np.dot(np.transpose(norm_M),norm_M)/m
        std_M = np.std(matrix,1-axis).reshape(1,n)
        db_std_M = std_M.T@std_M
        COR_M = COV_M / db_std_M
        
    return COR_M

def cor_M_heatmap(corr_M,label):
    color_norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    mapperable = cm.ScalarMappable(color_norm,cmap=cm.RdBu_r)

    from matplotlib.ticker import StrMethodFormatter

    fig = plt.figure(figsize=(11,9))
    ax = fig.gca()
    heatmap=ax.imshow(corr_M,aspect='auto',cmap=cm.RdBu_r)
    plt.colorbar(heatmap)

    valfmt = StrMethodFormatter('{x:.4f}')
    for i in range(corr_M.shape[0]):
        for j in range(corr_M.shape[1]):
            if j>i:
                continue
            font_color = ['black','w'][(np.tril(corr_M,k=0)[i,j] <0.96)|(np.tril(corr_M,k=0)[i,j] > 0.98)]
            ax.text(j-0.3,i+0.08,valfmt(np.tril(corr_M,k=0)[i,j],None),color=font_color,size=12);
    ax.set_yticks(range(9))
    ax.set_yticklabels(label,size=14);

    return fig

def uAUG_kozak_cooccur(with_uAUG,kozak_filter):
    start_factor_ls=[]
    # 0: no uAUG  , non kozak
    # 1: either is 1
    # 2: uAUG , kozak
    raw_factor = with_uAUG+kozak_filter
    for i,raw in enumerate(raw_factor):
        if raw == 0:
            factor='wo_uAUG_no_kozak'
        elif raw == 1:
            if with_uAUG[i] == 0:
                factor = 'wo_uAUG_kozak'
            else:
                factor = 'uAUG_no_kozak'
        else:
            factor = "uAUG_kozak"
            
        start_factor_ls.append(factor)
        
    return start_factor_ls


def process_pseudoalignment_bam(bam_path,ref:list,seq_include):
    """
    from kallisto pseudo alignment bam, extract refernce start and read length
    bam path : absolute path
    ref : list of reference ,`list(SeqIO.parse(ref_path,'fasta'))`
    seq_include : the isoform to include  , `df.target_id.values`
    """
    ribo_bam = pysam.AlignmentFile(bam_path,'rb')
    
    # construct a dict to map o
    tidx_2_tname = {}
    for i,SeqRecord in tqdm(enumerate(ref)):
        if SeqRecord.id in seq_include:
            tidx_2_tname[i] = SeqRecord.id

    # this is a super fast function to detect whtether a read falls in our interest group
    def to_stay_(enquery):
        try:
            return tidx_2_tname[enquery]
        except:
            return 0 
    
    # take read / segment that we want
    Segment_list =[]
    for Segment in tqdm(ribo_bam):
        if to_stay_(Segment.rname) != 0:
            Segment_list.append(Segment)

    start_dict={}
    len_dict={}
    # to initiate
    for idx in tidx_2_tname.keys():
        start_dict[idx]=[]
        len_dict[idx] = []
        

    for Segment in tqdm(Segment_list):
        start_dict[Segment.rname].append(Segment.reference_start)  # the 5' most of position 
        len_dict[Segment.rname].append(Segment.reference_length)   # read length
    
    return Segment_list, start_dict,len_dict


def kallisto_most_expr_isoform(input_df,tpm_col='tpm'):
    """
    Identify most expressive isoform from kallisto output
    
    args:
        input_df : kallisto output abundance tsv 
        tpm_col : the column to rank isoform within genes
    """
    if 'Gene' not in input_df.columns:
        if 'dedup_gbff_df' in globals().keys():
            input_df = input_df.merge(dedup_gbff_df,left_on=['target_id'],right_on=['ID'])
        else:
            globals()['dedup_gbff_df'] =  pd.read_csv('/data/users/wergillius/reference/GRCh38_p13_rna/2021_04_21_dedup_gbff_df.csv')
            input_df = input_df.merge(dedup_gbff_df,left_on=['target_id'],right_on=['ID'])

    input_df.sort_values(by=['Gene',tpm_col],ascending=False)
    df = input_df.drop_duplicates(['Gene'],keep='first')

    return df

def data_quality_plot(df,mRNA_col='tpm',ribo_col='ribo_tpm',get_data=False,fig=None,axs=None):
    
    print("nubmer of orf : \n",df.shape[0])
    
    x=np.log10(df[mRNA_col].values)
    y=np.log10(df[ribo_col].values)
    
    if fig is None:
        fig,axs=plt.subplots(1,2,figsize=(10,4),dpi=300)        
    
    # scatter
    scatter_linearreg_plot(x,y,ax=axs[0])
    axs[0].set_xlabel("log10 mRNA RPKM",fontsize=12)
    axs[0].set_ylabel("log10 RPF RPKM",fontsize=12)
    
    # TE distribution
    TE_d = np.log2(np.divide(df[ribo_col].values,df[mRNA_col].values))
    sns.kdeplot(TE_d,ax=axs[1])
    axs[1].set_xlabel("log2 TE",fontsize=12)
    
    if get_data:
        return x,y,TE_d
    
def gtf_line_2_json(gtf_line):
    """from the last one extract info"""
    json = {}
    json['chromosome'] = gtf_line[0]
    json['hierarchy'] = gtf_line[2]
    json['start'] = int(gtf_line[3])
    json['end'] = int(gtf_line[4])
    json['strand'] = gtf_line[6]
    last_field= gtf_line[-1]
    entry_ls=last_field.split(';')[:-1] # the last one is empty
    for entry in entry_ls:
        key,item = entry.strip().split(' ')
        json[key]=item.replace('"','')
    return json

def gtf_2_json(gtf_path,condition='_type "protein_coding"'):
    """for a gtf file, fectch protein coding line and convert to a list of dict"""
    gtf =[]
    with open(gtf_path,'r') as f:
        for i in range(5):
            head=f.readline()
        for line in tqdm(f):
            if condition in line:
                # convert into json
                split_line = gtf_line_2_json(line.strip().split("\t"))
                gtf.append(split_line)
        f.close
    return gtf

def build_GTE_graph(gtf):
    """
    construct Gene-Transcript-Exon relationship graph form gtf
        input: gtf : [dict] or str, can either input a processed gtf json or give a abs path 
    can look at notes:
        https://www.notion.so/GTF-Structure-at-Exon-level-6743fcfeab8f4aa9afcad92a5333b5df
    """ 
    
    if type(gtf) == str:
        # input is a file path
        gtf = gtf_2_json(gtf)
        
    gene_dict = {"_START_":{"location":[0]}}
    gene = "_START_"
    transcript_num = 0
    for i,anno in tqdm(enumerate(gtf)):
        if anno['hierarchy'] == 'gene':
            # previous gene_id
            gene_dict[gene]["location"].append(i) 
            # renew gene_id
            gene = anno["gene_id"]
            gene_dict[gene] = {'n_transcript':0,"location":[i]} # initiate a dict for gene
            transcript_num = 0

        elif anno['hierarchy'] == 'transcript':
            gene_dict[gene]['n_transcript'] += 1
            transcript = anno['transcript_id']
            gene_dict[gene][transcript] = []

        elif anno['hierarchy'] == 'exon':
            gene_dict[gene][transcript].append(anno['exon_id'])

    gene_dict[gene]["location"].append(i) 
    gene_dict.pop("_START_");

    return gene_dict

def build_exon_parent(gtf):
    """
    construct a Exon-Transcript relationship graph form gtf
        input: gtf : [dict] or str, can either input a processed gtf json or give a abs path 
    can look at notes:
        https://www.notion.so/GTF-Structure-at-Exon-level-6743fcfeab8f4aa9afcad92a5333b5df
    """ 
    if type(gtf) == str:
        # input is a file path
        gtf = gtf_2_json(gtf)

    exon_parent = {}
    for anno in gtf:
        if anno['hierarchy'] == 'exon':
            # initiate a {key:[],key:[]}
            exon_parent[anno['exon_id']] = []

    for anno in gtf:
        if anno['hierarchy'] == 'exon':
            exon_parent[anno['exon_id']].append(anno['transcript_id'])

    return exon_parent

def sam_line_cigar(test_entry):
    """
    from a line in sam or bam ,we process the line and return unmatched end
    """
    
    test_cigar = test_entry.__str__().split('\t')[5]
    test_seq = test_entry.seq

    left_S,right_S = re.match(r"(\d{,2})S{,1}[\d,S,M]*(\d{,2})S{,1}",test_cigar).groups()

    assert (left_S != '')|(right_S != '')
    
    if left_S == '':
        right_int = int(right_S)
        return test_entry.reference_start,test_seq[:-1*right_int]
    elif right_S == '':
        # only left side
        left_int = max(min(12,int(left_S)),5)
        return test_entry.reference_start + left_int , test_seq[left_int:]
    else:
        right_int = int(right_S)
        left_int = int(left_S)
        if left_int > right_int:
            left_int = max(min(12,left_S),5)
            return test_entry.reference_start + left_int , test_seq[left_int:]
        else:        
            return test_entry.reference_start , test_seq[:-1*right_side]

def compute_offset_matrix(start_codon_coordinate,reads_list,strategy='annotated start site'):
    """
    Compute P site offset matrix , read length (20,40) to probability of offset (-30,30). The transcripts / Exon to use can be specified by `start_codon_coordinate` and `reads_list`
    
    Args:
    ...start_codon_coordinate : list of tuple, [(chromosome,start,end),..] 
    ...reads_list : list [pysam.Segment, ...]. Shoulb be the same length as `start_codon_coordinate`
    ...strategy : str, 'annotated start site' or 'ATG_index'; the first one, with compute offset using distance from start codon coordinate to the distance of read left side location; the second one use sequence of the reads, and find the position of ATG in it.
    
    Returns:
    ...offset_M : offset matrix M, m_{i,j} i: read length, j: distance from start / positionn in the string depending on the strategy used. 
    """
    
    assert len(start_codon_coordinate) == len(reads_list) , "these two list should of the same length"
    
    ######################################
    ## reads stratified by read length  ##
    ######################################
    len_distribution ={i:[] for i in range(20,40)} 
    
    for i in tqdm(range(len(start_codon_coordinate))):
        # iter over start codon
        if len(reads_list[i]) > 100:
            anno_start_site =  start_codon_coordinate[i][1]

            if strategy == 'ATG_index':
                try:
                    left_site = np.array([read.seq.index("ATG") for read in reads_list[i]])
                except ValueError:
                    raise ValueError("the wrong read list is used, please use the one every reads with ATG")
            else:
                left_site = np.array([read.reference_start for read in reads_list[i]]) - anno_start_site

            seq_len = [len(read.seq) for read in reads_list[i]]

            # iter over reads 
            for j in range(len(seq_len)):
                len_distribution[seq_len[j]].append(left_site[j])
                
    #############################
    ## compute offset matrixs  ##
    #############################
    offset_M = []
    for len_seq in range(20,40):
        len_dist=np.array(len_distribution[len_seq])
        offset_of_len = [np.sum(len_dist==distance) for distance in range(-30,31)]
        # stack
        offset_M.append(offset_of_len)

    offset_M = np.stack(offset_M)
    print(offset_M.shape)

    return offset_M        
        
def compute_codon_abundance(bam_path,P_site_offset,faid_2_transcript,is_transcripts_picked,A_site=False,cigar_specified='32M',total_read_n=None):
    """
    with P site offset, the function will open indexed bam file, summarise the codon abundance of reads of specified length.
    Args
    ..bam_path : the absolution path of bam file, bam must be sorted and indexed
    ...P_site_offset : int, the P site offset of `length_specified`, i.e P_site_offset=13 for length = 32
    ...faid_2_transcript : arguments for kallisto pseudo-alignment, the chr in reads is fa id 
    ...length_specified : int, the argument coupled with P_site offset
    ...A_site : boolen, the codon to summarize swithch to A site if `True`
    Returns
    ...P_codon_stats : dict, the codon : relative abundance  
    """
    
    if A_site:
        p_offset = P_site_offset +3
    else:
        p_offset = P_site_offset
    
    # to know the total number of reads to iter
    ribo_sam = pysam.AlignmentFile(bam_path,'rb')
    
    print('\ncalculating total number of reads...')
    if total_read_n is None:
        for i,_ in enumerate(ribo_sam):
            continue
        total_read_n = i + 1
    print(total_read_n,"\nstart codon summarizing..")
    
    ######################################
    ##  summarize the codon appearance  ## 
    ######################################
    
    P_codon_stats = {codon:0 for codon in possible_codon}
    
    ribo_sam = pysam.AlignmentFile(bam_path,'rb')
    # iterate all the reads
    for i in tqdm(range(total_read_n)):
        read = next(ribo_sam)
        
        # only cleanly mapped reads , default 32M
        if read.cigarstring != cigar_specified:
            continue

        # only consider read of len 32 for the offset of it is the most accurate
        tid = faid_2_transcript[int(read.__str__().split()[2])]
        if is_transcripts_picked(tid) != False:
            codon = read.seq[p_offset:p_offset+3]
            P_codon_stats[codon] += is_transcripts_picked(tid) # + 1 / tpm
            
    ###############################
    ## normalize codon abundance ##
    ###############################
    # summation
    total_reads = 0
    for codon,summation in P_codon_stats.items():
        total_reads += summation
    
    # relative abundance
    for codon,summation in P_codon_stats.items():
        P_codon_stats[codon] = summation/total_reads
    
    return P_codon_stats

def compute_transcript_k_el(cds_seq,P_site_codon_stats,just_freq=False):
    """
    elongaion rate is computed by the inner product of decoded probability and codon frequency in cds sequence
    k_el = [ P_AAA ,... , P_CTG ]@ [F_AAA, ..., F_CTG] 
    Arguments:
    ... cds_seq : str, the cds sequence used to compute 
    ... P_site_codon_stats : dict, keys == possible_codon
    Returns:
    ... elongation rate : float
    """
    cds_codon_stats = {codon:0 for codon in possible_codon}
    assert len(cds_seq)%3 == 0
    protin_len = len(cds_seq)//3
    
    for i in range(0,len(cds_seq),3):
        codon = cds_seq[i:i+3]
        cds_codon_stats[codon] += 1
    
    # relative abundance
    for codon,summation in cds_codon_stats.items():
        cds_codon_stats[codon] = summation/protin_len
    cds_codon_freq = np.array([freq for codon,freq in cds_codon_stats.items()])
    
    if just_freq:
        return list(cds_codon_freq)
    else:
        dwelling_time = np.array([1/summation for codon,summation in P_site_codon_stats.items()])

        return 1/np.inner(dwelling_time,cds_codon_freq)
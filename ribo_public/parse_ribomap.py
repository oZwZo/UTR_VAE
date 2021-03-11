import os
import PATH
import numpy as np
import pandas as pd
from Bio import SeqIO
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from models import reader
from sklearn import linear_model
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt
from torch.utils.data import random_split
global P_M_order
global nt_order
P_M_order= ['Pyrimidine','Purine']
nt_order = ['T','C','A','G']

get_value = lambda x : x.split(':')[-1].strip()



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
    
    print("y = %.3f x + %.3f"%(linear_mod.coef_,linear_mod.intercept_))
    
    if ax is None:
        fig = plt.figure(figsize=(6,4))
        ax = fig.gca()
    
    ax.scatter(quanty,y,s=5,alpha=0.2)
    ax.plot(line_x,line_x,'-.',color='orange',alpha=0.5,label='y=x')
    ax.plot(line_x,line_y,'--',color='black',alpha=0.5,label=r'$R^2$=%.3f'%r2)
    plt.legend()


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
    
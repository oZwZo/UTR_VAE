import os
import sys
import re
import pysam
from tqdm import tqdm

def process_sam_line(test_entry):
    """
    from a line in sam or bam ,we process the line and return unmatched end
    """
    test_cigar = test_entry.__str__().split('\t')[5]
    test_seq = test_entry.__str__().split('\t')[9]

    left_S,right_S = re.match(r"(\d{,2})S{,1}[\d,S,M]*(\d{,2})S{,1}",test_cigar).groups()

    assert (left_S != '')|(right_S != '')

    return int_of(left_S,right_S,test_seq)

def int_of(left_S,right_S,seq):
    """ from CIGAR i.e. 8S32M , we identy the part not matched and extract sequence of unmatched end"""
    try:
        left_int = int(left_S)
    except:
        # only right side
        right_int = max(min(int(right_S),9),8)*-1  # reversed index 
        return seq[right_int:]
    
    try:
        right_int = int(right_S)
    except:
        # only left side
        left_int = max(min(9,int(left_S)),8)
        return seq[:left_int]
    
    # both side have matched number 
    if left_int > right_int:
        left_int = max(min(9,left_S),8)
        return seq[:left_int]
    else:
        right_int = max(min(right_S,9),8)*-1  # reversed index 
        return seq[right_int:]

def main():
    input_sam = sys.argv[1]
    output_adaptor = sys.argv[2]
    # read in bam or sam
    Alignment_file = pysam.AlignmentFile(input_sam,'rb') if input_sam.endswith('bam') else pysam.AlignmentFile(input_sam,'r')
    print('read in %s \n'%input_sam)
    print('start analysing adaptor.. this will take about 10 min')
    # process each line and write the adaptor
    with open(output_adaptor,'w') as f:
        
        for Segment in tqdm(Alignment_file,'writing adaptor ....'):
            try:
                adaptor = process_sam_line(Segment)
            except AssertionError:
                # both left and right are matched
                continue
            f.write(adaptor+'\n')
        
        print('\n finished')

if __name__ == '__main__':
    main()
        
        
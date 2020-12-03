import os
import sys
import numpy as np
import pandas as pd

sys.path.append("/home/wergillius/tool/RNAlib/lib/python3.8/site-packages")
sys.path.append("/home/wergillius/tool/RNAlib/lib/python3.8/")

import RNA



def parsing_secondary_structure(ss,return_all=False):
    """
    the all in once integrated function:
        1. convert ss to ptv
        2. seperate block to block_ls
        3. take out loops only
        4. detect the closing pair loc of each continous block 
        5. find out which blocks share common closing pair
        6. merge blocks that form a bigger loop
        7. identify loop type of the merge blocks 
    ...arguments:
        ...return_all :multiloop_blocks,loops_loc,closing_pair_ls,co_pair_ls,merged_co_pair_ls,loop_type_ls,cps_bigloop_ls
    
    
    ...input : dot-bracket text
    ...output: the three output with matched length
        ...loop_type_ls     : loop type of each merge loop
        ...merged_co_pair_ls: index of block that form merge loop
        ...cps_bigloop_ls   : all closing pair of merge loop
        
    """
    # step 1. convert ss to ptv
    
    if not "(" in ss:
        return ['bad_seq']
    
    ptv=list(RNA.ptable(ss))[1:]
    
    # step 2. seperate block to block_ls
    multiloop_blocks = seperate_block(ptv)[0]
    
    # step 3. take out loops
    loops_loc = get_loops_loc(multiloop_blocks) 
        
    # step 4. detect the closing pair loc of each continous block 
    closing_pair_ls = [get_closing_pair_of_loop(multiloop_blocks,loops_loc,i) for i in range(len(loops_loc))]
    
    # step 5. find out which blocks share common closing pair
    co_pair_ls = find_co_closing_pair(closing_pair_ls)
    
    # step 6. merge blocks that form a bigger loop
    merged_co_pair_ls = merge_potential_multi_loop(co_pair_ls)
    
    # step 7. identify loop type of the merge blocks
    loop_type_ls,cps_bigloop_ls = identify_loop_type(merged_co_pair_ls,closing_pair_ls)
    
    result = loop_type_ls,merged_co_pair_ls,cps_bigloop_ls
    
    if return_all:
        result = multiloop_blocks,loops_loc,closing_pair_ls,co_pair_ls,merged_co_pair_ls,loop_type_ls,cps_bigloop_ls
    
    return result

def detect_part(ptv):
    """
    detect how many part can the ptv devided to 
    """
    end_ls = []                                       # a list to store ending loc of the part

    ptv_is_not_0 = np.array(ptv) != 0
                                                      # end and start of the first part
    end_ls.append([loc for loc in ptv if loc != 0][0])#  starting of the first    
                                                      # `last_end` is the ending loc of the last part
    last_end = end_ls[0]                              # initialize
    
    while (1 in ptv_is_not_0[last_end:]):             # with remaining part still have matching
       
        end = [loc for loc in ptv[last_end:] if loc != 0][0]
        end_ls.append(end)
        last_end = end
        
    start_ls = [ptv.index(loc) for loc in end_ls]
    
    return [(i,j) for i,j in zip(start_ls,end_ls)]

def seperate_block(ptv,verbose=False):
    """
    from a ptable, detect continous block and record the location of closing pair 
    closing pair location is the left side 
    """
    continous_ls = []
    closing_pair_loc = []
    last = ptv[0]
    local_block = [last]
    for loc,current in enumerate(ptv[1:]):
        if verbose:
            print("the last one",last)
            print("the current one",current)

        ## is actually a logical XOR
        if (current == last)|((last != 0 )& (current !=0)):
            # 0 & 0         |   !0.  & !0
            local_block.append(current)       # enlong the local_block 

        else:
            # one of them is 0 and one of it is not
            # the starting of 
            closing_pair_loc.append(loc)
            continous_ls.append(local_block)
            local_block = [current] # since `current` is 0

        last = current

        if verbose:
            print(local_block)
            print(continous_ls)
    continous_ls.append(local_block)
    return continous_ls,closing_pair_loc

def label_location_for_ptable(ptv):
    """
    for ptable, note its location
    so that it can 
    """
    numeric_blocks=[]                  # the new ptable
    i = 1                              # count
    
    for block in ptv:     # loop against different block
        n_local_block=[]
        for value in block:            # loop within block
            n_local_block.append((value,i)) # the value together with location
            i += 1
        numeric_blocks.append((n_local_block))
        
    return numeric_blocks if len(numeric_blocks) > 1 else numeric_blocks[0]  


def get_loops_loc(multiloop_blocks):

    loops_loc = np.where([np.sum(block) == 0 for block in multiloop_blocks])[0]
    if loops_loc[0] == 0:
    # the last [0,..0] also the last block , it will not form a loop
        loops_loc = loops_loc[1:]
    if loops_loc[-1] == len(multiloop_blocks)-1:
    # the last [0,..0] also the last block , it will not form a loop
        loops_loc = loops_loc[:-1] 
    return loops_loc

def get_closing_pair_of_loop(multiloop_blocks,loops_loc,index):
    """
    given seperated continous blocks , return the closing pair of the i^th loop
    """
    
    # label the blocks so that we can have pair loc later
    n_multiloop_blocks=label_location_for_ptable(multiloop_blocks)

    #  n_multiloop_blocks[loops_loc[i]] :  the i^th loop block
    
    # [0] : the head of the loop , [1] is the location , -1 to get the location of last closing pair
    head_closing_pair = n_multiloop_blocks[loops_loc[index]-1][-1]   
    # [0] : the tail of the loop , [1] is the location , +1 to get the location of next closing pair
    tail_closing_pair = n_multiloop_blocks[loops_loc[index]+1][0]   
    
    return head_closing_pair,tail_closing_pair

def detect_hairpin(closing_pairs):
    head_closing_pair,tail_closing_pair = closing_pairs
    
    # ((34, 29), (29, 34))
    if (head_closing_pair == tail_closing_pair) | (head_closing_pair == tail_closing_pair[::-1]):
        is_hairpin = True
    else:
        is_hairpin = False
    
    return is_hairpin

def find_co_closing_pair(closing_pairs_ls):
    """
    find out the loops block that with shared closing pair
    """
    hairpin_loc = [detect_hairpin(cps) for cps in  closing_pairs_ls]   # cps: closing pairs ((xx,ww),(zz,ee))
    
    co_closing_index_ls= []
    for i,cps in enumerate(closing_pairs_ls):
        co_closing_of_i = [i]
            
        for j,matching_cps in enumerate(closing_pairs_ls):
            
            if i == j:
                continue
                
            head_i , tail_i = matching_cps
            if (head_i in cps) | (head_i[::-1] in cps) | (tail_i in cps) | (tail_i[::-1] in cps):
                    co_closing_of_i.append(j)
        
        co_closing_index_ls.append(co_closing_of_i)
    return co_closing_index_ls

def merge_potential_multi_loop(co_pair_ls):
    """
    this is designed to 1. merge multi-loop co_pair location ; 2. deduplicate
    ... input : co_pair_ls , list[list] , ie.[[0, 1, 3], [1, 0, 2], [2, 1, 3], [3, 0, 2]]
    ...output : to_merge   , list[list] , with each element contain all the co_pair that together form a big loop
    ... ie.  [0, 1, 2, 3]
    """
    
    # ----- part.1 merge -----
    to_merge = []
    
    for i,co_pair_i in enumerate(co_pair_ls):            # co_pair_i : list , ie. [0, 1, 3]
        i_share = [i]
        for j , co_pair_j in enumerate(co_pair_ls):
            
            if i == j:                                   # avoid merging itself
                continue

            if len(np.intersect1d(co_pair_i, co_pair_j)) != 0:   # if they share any co_pair
                i_share.append(j)
                co_pair_i += co_pair_j
                
        to_merge.append(i_share)
    
    # ---- part.2 de-duplicate ----
    to_merge = np.array([sorted(i_share) for i_share in to_merge],dtype=object)  # sorted every co_pair
    
    
    to_merge = list(np.unique(to_merge)) 
    
    # to fix some problem met when only 1 merging loop 
    if type(to_merge[0]) == int:
        to_merge = [to_merge]
    return to_merge

def identify_loop_type(merged_co_pair_loop_ls,closing_pair_ls):
    """
    for merged loops, detect the extact number of closing pairs, and thus determine the loop type
    """
    loop_type_ls = []
    cps_bigloop_ls = []
    
    for loc_big_loop in merged_co_pair_loop_ls:         # the index of consisting closing pair
        
        all_closing_pair = []
        for iid in loc_big_loop:
            all_closing_pair.append(closing_pair_ls[iid][0])
            all_closing_pair.append(closing_pair_ls[iid][1])
        all_closing_pair = np.unique([sorted(closing_pair) for closing_pair in all_closing_pair],axis=0)
        
        num_closing_pair = len(all_closing_pair)
        
        if len(loc_big_loop) == 1:                      # bulge or hairpin
            
            if num_closing_pair ==1:
                loop_type_ls.append('Hairpin')
            elif num_closing_pair ==2:
                loop_type_ls.append("Bulge")
            else:
                raise AssertionError("There are other un-considered case when len(loc_big-loop)==1")
            
            
        elif len(loc_big_loop) == 2:                    # internal-loop
            loop_type_ls.append('Internal_loop')
        elif len(loc_big_loop) > 2:                     # multi-loop
            loop_type_ls.append('Multi_loop')
        else:
            raise AssertionError("There are other un-considered case with len(loc_big-loop)")
        
        cps_bigloop_ls.append(all_closing_pair)
        
    return loop_type_ls,cps_bigloop_ls

    
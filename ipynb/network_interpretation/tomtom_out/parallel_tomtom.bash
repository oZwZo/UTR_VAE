echo "please enter motif txt file : xxx_cv"
read Query

motif_DIR=/ssd/users/wergillius/Project/UTR_VAE/ipynb/network_interpretation/save_motif
for i in {7..7} ;do 
    # TOMTOM 
    bash run_tomtom.sh ${motif_DIR}/${Query}_cv$i.txt ${motif_DIR}/AllCh_MerCelLine_1c_Nov2.txt > ${Query}_ribo_cv${i}.tsv &
    bash run_tomtom.sh ${motif_DIR}/${Query}_cv$i.txt ${motif_DIR}/ystsct_allCh_1c_Nov2.txt > ${Query}_yeast_cv${i}.tsv &
    pid=$!
    # SEA
    bash ../sea_out/run_sea.sh ${motif_DIR}/${Query}_cv$i.txt > ../sea_out/${Query}_enrich_cv$i.tsv &
    
    done
    
wait $pid

for i in {7..7} ;do 
    # merge 2 TOMTOM result
    python merge_2_tomtomout.py ${Query} $i &
    done
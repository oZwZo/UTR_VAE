
echo "please enter relative directory path"
read log_dir
for filename in $log_dir/*.ini; do
    out_name=$(echo $filename | cut -d "." -f 1)
    
    nohup python script/iter_train.py  --config_file $filename --cuda 1 --kfold_index 0 > ${out_name}.out &
    echo $out_name $!
done 


echo "please enter relative directory path"
read log_dir
for filename in $log_dir/*.ini; do
    out_name=$(echo $filename | cut -d "." -f 1)
    
     nohup python script/main_train.py  --config_file $filename > ${out_name}.out &
done 

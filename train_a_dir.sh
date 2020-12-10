
echo "please enter relative directory path"
read log_dir
for filename in $log_dir/*.ini; do
    nohup python script/main_train.py  --config_file $filename > $filename.out &
done 

CONFIG_FILE="/home/wergillius/Project/UTR_VAE/log/LSTM_AE/setting4_AE/setting4_AE.ini"
CUDA_VISIBLE_DEVICES=1
nohup python script/main_train.py --config_file $CONFIG_FILE > traning1.out &
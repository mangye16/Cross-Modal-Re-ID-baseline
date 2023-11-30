CUDA_VISIBLE_DEVICES=0 python train_cap.py --target '1' --data_dir '/data0/ReIDData/RegDB/slice-cam2' --logs_dir  '/data1/log/regdb/1' 


# target: trial for regdb, ('0', '1', ..., '10')
# data_dir: dataset path
# logs_dir: train log path
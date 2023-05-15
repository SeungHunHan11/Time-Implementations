wandb=$1
subdataset='A1 A2 A3 A4'
Lambda='3'

for sub in $subdataset
do
    for lambda in $Lambda
    do
        Exp_name="Yahoo-$sub-Lambda_$lambda"
        echo $Exp_name
        python main.py \
            --use_wandb $wandb \
            --train_mode True \
            --use_scheduler True \
            --end_to_end False \
            --eval_per_epoch True \
            --savedir './saved_models/' \
            --checkpoint_name 'best_model' \
            --dataset_name yahoo \
            --datadir ./data/Anomaly_Detection/yahoo_S5 \
            --use_rawdata True \
            --scaler MinMaxScaler \
            --window_size 100 \
            --step_size 1 \
            --d_model 512 \
            --n_head 8 \
            --num_layers 3 \
            --ffnn_dim 512 \
            --dropout 0.4 \
            --activation 'relu' \
            --norm_type LayerNorm \
            --emb_type Conv1d \
            --pos_type encoding \
            --drop_pos False \
            --batch_size 32 \
            --num_workers 12 \
            --lr 1e-4 \
            --epochs 10 \
            --Lambda $lambda \
            --temperature 50 \
            --anomaly_ratio 0.1 \
            --point_adjustment False \
            --run_name $Exp_name \
            --seed 1998\
            --project_name 'Anomaly-Transformer-yahoo'\
            --subdataset $sub
    done
done


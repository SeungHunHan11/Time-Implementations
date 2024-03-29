use_wandb=$1
pos_type='learnable encoding'
scalers='StandardScaler MinMaxScaler'
norm_type='BatchNorm1d LayerNorm'
emb_type='Linear Conv1d'
Lambda='1 2 3'
data='SMAP'

for pos in $pos_type
do
    for scaler in $scalers
    do
        for norm in $norm_type
        do
            for lambda in $Lambda
            do
                for emb in $emb_type
                do
                    echo "pos: $pos, scaler: $scaler, Norm_type: $norm, Embedding: $emb, Lambda: $lambda"
                    Exp_name="$data-pos_$pos-scaler_$scaler-Norm_$norm-Emb_$emb-Lambda_$lambda"

                    python main.py \
                        --use_wandb $use_wandb \
                        --train_mode True \
                        --use_scheduler True \
                        --end_to_end False \
                        --eval_per_epoch True \
                        --savedir './saved_models/' \
                        --checkpoint_name 'best_model' \
                        --dataset_name $data \
                        --datadir ./data \
                        --scaler $scaler \
                        --window_size 100 \
                        --step_size 1 \
                        --d_model 512 \
                        --n_head 8 \
                        --num_layers 3 \
                        --ffnn_dim 512 \
                        --dropout 0.3 \
                        --activation 'relu' \
                        --norm_type $norm \
                        --emb_type $emb \
                        --pos_type $pos \
                        --drop_pos False \
                        --batch_size 32 \
                        --num_workers 12 \
                        --lr 1e-4 \
                        --epochs 10 \
                        --Lambda $lambda \
                        --point_adjustment False \
                        --temperature 50 \
                        --anomaly_ratio 1.00 \
                        --run_name $Exp_name \
                        --seed 1998
                done
            done
        done
    done
done

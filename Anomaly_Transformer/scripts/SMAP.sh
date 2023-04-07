batch_size='128 256'
pos_type='learnable encoding'
scalers='StandardScaler MinMaxScaler'
norm_type='BatchNorm1d LayerNorm'
emb_type='Linear Conv1d'
Lambda='3 4 5'
data='SMAP'

for bs in $batch_size
do
    for pos in $pos_type
    do
        for scaler in $scalers
        do
            for norm in $norm_type
            do
                for emb in $emb_type
                do
                    for lambda in $Lambda
                    do
                        echo "BS: $bs, pos: $pos, scaler: $scaler, Norm_type: $norm, Embedding: $emb, Lambda: $lambda"
                        Exp_name="$data-batch_$bs-pos_$pos-scaler_$scaler-Norm_$norm-Emb_$emb-Lambda_$lambda"

                        python main.py \
                            --train_mode True \
                            --use_scheduler True \
                            --end_to_end False \
                            --eval_per_epoch True \
                            --savedir './saved_models/' \
                            --checkpoint_name 'latest_model.pt' \
                            --dataset_name $data \
                            --datadir ./data \
                            --scaler $scaler \
                            --window_size 100 \
                            --step_size 1 \
                            --d_model 512 \
                            --n_head 6 \
                            --num_layers 3 \
                            --ffnn_dim 512 \
                            --dropout 0.3 \
                            --activation 'relu' \
                            --norm_type $norm \
                            --emb_type $emb \
                            --pos_type $pos \
                            --drop_pos False \
                            --batch_size $bs \
                            --num_workers 12 \
                            --lr 1e-4 \
                            --epochs 10 \
                            --Lambda $lambda \
                            --temperature 50 \
                            --anomaly_ratio 4.00 \
                            --run_name $Exp_name \
                            --seed 1998

                    done
                done
            done
        done
    done
done
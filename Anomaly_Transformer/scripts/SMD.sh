use_wandb=$1
pos_type='encoding'
scalers='StandardScaler'
norm_type='LayerNorm'
emb_type='Conv1d'
Lambda='3'
data='SMD'

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
                        --use_rawdata False \
                        --scaler $scaler \
                        --window_size 100 \
                        --step_size 100 \
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
                        --temperature 50 \
                        --anomaly_ratio 0.50 \
                        --point_adjustment False \
                        --run_name $Exp_name \
                        --project_name 'Anomaly-Transformer'\
                        --seed 1998
                done
            done
        done
    done
done

python main.py \
    --use_wandb False \
    --train_mode True \
    --use_scheduler False \
    --end_to_end True \
    --eval_per_epoch True \
    --savedir './saved_models/' \
    --checkpoint_name 'best_model' \
    --dataset_name 'SMD' \
    --datadir ./data \
    --use_rawdata False \
    --scaler 'StandardScaler' \
    --window_size 100 \
    --step_size 100 \
    --d_model 512 \
    --n_head 8 \
    --num_layers 3 \
    --ffnn_dim 512 \
    --dropout 0.3 \
    --activation 'relu' \
    --norm_type 'BatchNorm1d' \
    --emb_type 'Conv1d' \
    --pos_type 'encoding' \
    --drop_pos False \
    --batch_size 32 \
    --num_workers 8 \
    --lr 1e-4 \
    --epochs 10 \
    --Lambda 3 \
    --temperature 50 \
    --anomaly_ratio 1.00 \
    --point_adjustment False \
    --run_name 'without_PA_1998' \
    --project_name 'Anomaly-Transformer'\
    --seed 1998
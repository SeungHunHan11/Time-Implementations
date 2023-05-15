SEED='1998 1105 1905'
PA='True False'

for seed in $SEED
do
    for pa in $PA
    do
        # Check if the number is greater than 10
        if [ "$pa" = "True" ]
        then
            Exp_name="with_PA_$seed"
        else
            Exp_name="without_PA_$seed"
        fi

        echo "$Exp_name-MSL"

        python main.py \
                --use_wandb False \
                --train_mode True \
                --use_scheduler False \
                --end_to_end True \
                --eval_per_epoch True \
                --savedir './saved_models/' \
                --checkpoint_name 'best_model' \
                --dataset_name 'MSL' \
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
                --point_adjustment $pa \
                --run_name $Exp_name \
                --project_name 'Anomaly-Transformer'\
                --seed $seed
    done
done
docker build -t anomaly_transformer . 

nvidia-docker run -it -h Ano_trans \
        -p 4444:4444 \
        --ipc=host \
        --name Ano_trans \
        -v /home/seunghun/바탕화면/Time_Series_exp/Anomaly_Transformer:/directory \
        anomaly_transformer bash
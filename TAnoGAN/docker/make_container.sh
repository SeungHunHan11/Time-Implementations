docker build -t ts_exp . 

nvidia-docker run -it -h ts \
        -p 1936:1936 \
        --ipc=host \
        --name ts \
        -v /home/seunghun/바탕화면/Time_Series_exp:/workspace \
        ts_exp bash
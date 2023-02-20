nvidia-docker run -it -h dagmm \
        -p 1905:1905 \
        --ipc=host \
        --name dagmm \
        -v /home/seunghun/바탕화면/DAGMM:/DAGMM \
        nvcr.io/nvidia/pytorch:20.11-py3 bash
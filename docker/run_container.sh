docker rm -f bundlesdf
DIR=$(pwd)/../
# xhost +  && docker run --gpus all --env NVIDIA_DISABLE_REQUIRE=1 -it --network=host --name bundlesdf  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined  -v /home:/home -v /tmp:/tmp -v /mnt:/mnt -v $DIR:$DIR  --ipc=host -e DISPLAY=${DISPLAY} -e GIT_INDEX_FILE nvcr.io/nvidian/bundlesdf:latest bash
xhost +  && \
docker run --gpus all \
  --env NVIDIA_DISABLE_REQUIRE=1 \
  -it \
  --network=host \
  --name bundlesdf \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  -v /home:/home \
  -v /tmp:/tmp \
  -v /mnt:/mnt \
  -v $DIR:$DIR \
  --ipc=host \
  -e DISPLAY=${DISPLAY} \
  nvcr.io/nvidian/bundlesdf:latest \
  bash


# python run_custom.py --mode run_video --video_dir /home/justin/code/Manipulator-Software/data/object_mesh/mouse_box/  --out_folder /home/justin/code/Manipulator-Software/data/object_mesh/mouse_box/ --use_segmenter 1 --use_gui 1 --debug_level 2
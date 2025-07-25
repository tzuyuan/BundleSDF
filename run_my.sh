# python3 run_custom.py \
#     --mode run_video\
#     --video_dir /home/justin/code/Manipulator-Software/data/object_mesh/mouse_box3/ \
#     --out_folder /home/justin/code/Manipulator-Software/data/object_mesh/mouse_box3_out/ \
#     --use_segmenter 1 \
#     --use_gui 1 \
#     --debug_level 4

python3 run_custom.py \
    --mode global_refine \
    --video_dir /home/justin/code/Manipulator-Software/data/object_mesh/mouse_box3/ \
    --out_folder /home/justin/code/Manipulator-Software/data/object_mesh/mouse_box3_out/ \
    --use_segmenter 1 \
    --use_gui 1 \
    --debug_level 4

# 3) (Optional) If you want to draw the oriented bounding box to visualize the pose, similar to our demo
python3 run_custom.py \
    --mode draw_pose \
    --out_folder /home/justin/code/Manipulator-Software/data/object_mesh/mouse_box3_out/

# python3 run_custom.py \
#     --mode run_video\
#     --video_dir /home/justin/code/Manipulator-Software/data/sam_ycb_output/ \
#     --out_folder /home/justin/code/Manipulator-Software/data/sam_ycb_output_out/ \
#     --use_segmenter 0 \
#     --use_gui 1 \
#     --debug_level 4

# python3 run_custom.py \
#     --mode run_video\
#     --video_dir /home/justin/code/BundleSDF/data/2022-11-18-15-10-24_milk \
#     --out_folder /home/justin/code/BundleSDF/data/2022-11-18-15-10-24_milk_out \
#     --use_segmenter 0 \
#     --use_gui 0 \
#     --debug_level 0

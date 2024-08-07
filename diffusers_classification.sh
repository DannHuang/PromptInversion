accelerate launch --num_processes 1 --gpu_ids 0 cls_token_optimization.py \
    --output_dir /share2/huangrenyuan/logs/ground_diffusion \
    --pretrained_model_name_or_path /share2/huangrenyuan/model_zoo/stable-diffusion-xl-base-1.0 \
    --train_data_dir /share2/huangrenyuan/dataset/imagenet512.zip \
    --train_batch_size 1 \
    --gradient_checkpointing \
    --class_token "<cls>" \
    --mixed_precision no \
    --learning_rate 1e-2 \
    --seed 42 \
    --test_num 1000 \
    --resolution 1024 \
    --test_timestep 200 \
    --test_rabbit
    --attn_map
    --initializer_token object \

# 50, 60, 70, 80, 90, 140
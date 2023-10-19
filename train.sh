accelerate launch examples/text_to_image/train_text_to_image_lora_sdxl.py \
  --pretrained_model_name_or_path "stabilityai/stable-diffusion-xl-base-1.0" \
  --train_data_dir "dataset/flat_color/r1" --caption_column "text"\
  --resolution 1024 \
  --train_batch_size 8 \
  --num_train_epochs 2000 --checkpointing_steps 100 \
  --learning_rate 1e-04 --lr_scheduler "constant" --lr_warmup_steps 0 \
  --seed 0 \
  --output_dir "saved_weights/flat_color_mode0"  \
  --rank 4 \
  --PRF_mode 1 \
  --Train_mode 0 #\
  #--pretrained_model_path "saved_weights/ink_painting_new1" \
  #--pretrained_model_iter 1500
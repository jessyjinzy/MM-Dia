accelerate launch --config_file accl_config.yaml train_higgs.py \
    --train_data_path ../output_data/mm_dia_splits/train\
    --val_data_path   ../output_data/mm_dia_splits/eval\
    --batch_size 3 \
    --lr 1e-5 \
    --accumulation_steps 8 \
    --instruction \
    --freeze_text_encoder \
    --semantic_amplification 4.0 \
    --epochs 25\
    --output_dir ../exp/test
    #--resume_from_checkpoint \


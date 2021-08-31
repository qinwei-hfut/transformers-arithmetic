import os
import pdb
import json


def run_exp():
    the_command = 'python main.py \
        --output_dir=. \
        --model_name_or_path=t5-base \
        --operation=addition \
        --orthography=10ebased \
        --balance_train \
        --balance_val \
        --train_size=100000 \
        --val_size=10000 \
        --test_size=10000 \
        --min_digits_train=2 \
        --max_digits_train=15 \
        --min_digits_test=2 \
        --max_digits_test=15 \
        --base_number=10 \
        --seed=1 \
        --train_batch_size=4 \
        --accumulate_grad_batches=32 \
        --val_batch_size=32 \
        --max_seq_length=512 \
        --num_workers=4 \
        --gpus=1 \
        --optimizer=AdamW \
        --lr=3e-4 \
        --weight_decay=5e-5 \
        --scheduler=StepLR \
        --t_0=2 \
        --t_mult=2 \
        --gamma=1.0 \
        --step_size=1000 \
        --max_epochs=20 \
        --check_val_every_n_epoch=2 \
        --amp_level=O0 \
        --precision=32  \
        --dataset=NumPrediction_B_T5 \
        --gradient_clip_val=1.0'

    os.system(the_command)



# output_dir = ''

run_exp()

- [bhagavad gita data source: archive.org](https://archive.org/stream/bhagwat-gita-in-nepali/bhagwat%20gita%20in%20NEPALI_djvu.txt)

# modification in previous_chapters.py: create_dataloader_v2()

- replace data*files from:
  `data_files = {"train": base_url + "nepberta*" + str(context_length) + ".parquet"}`
- to:
  `data_files = {"train": base_url + "iriisnepal_u_nepberta_512.parquet"}`

## Run Debug code

!python 3_pretrain.py \
 --n_epochs 1 \
 --batch_size 2 \
 --output_dir model_checkpoints \
 --eval_freq 100 \
 --save_ckpt_freq_steps 200 \
 --debug True \

## Files required for stand-alone-debug training

- [3_pretrain.py](./3_pretrain.py)
- [previous_chapters.py](./previous_chapters.py)
- [functions.py](./functions.py)
- [tokenizer.json](./tokenizer.json)
- [debug_dataloaders.py](./debug_dataloaders.py)
- [cleaned_bhagwat_gita.txt](./cleaned_bhagwat_gita.txt)

# Run actual train-data

## Run Debug code

```
# 300M
# =======
# # # # Download latest model checkpoint from hub
# # # # <replace checkpoint_name with latest checkpoint name>

!rm -rf parameters_300m/*

# Download latest checkpoint from parameters_300m to /kaggle/working/parameters_300m
!python3 4.5_download_checkpoint_from_hub.py \
  --checkpoint_path parameters_300m \
  --destination /kaggle/working

!ls -l --block-size=M parameters_300m

!python 3_pretrain.py \
 --n_epochs 2 \
 --batch_size 18 \
 --output_dir "parameters_300m" \
 --eval_freq 1000 \
 --save_ckpt_freq_steps 1000 \
 --debug False \
 --resume_from_previous_training True \
 --push_to_hub_hours 11.6 \
 --initial_lr 6e-5 \
 --min_lr 6e-5  \
 --peak_lr 6e-4 \
 --warmup_steps .01 \
 --cplr_steps 0.0 \
 --num_parameters 300 \
 --min_lr_steps 0 # min. lr at the end
```

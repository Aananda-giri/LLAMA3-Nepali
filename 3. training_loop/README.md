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

## Files required for stand-alone training

- [3_pretrain.py](./3_pretrain.py)
- [previous_chapters.py](./previous_chapters.py)
- [functions.py](./functions.py)
- [tokenizer.json](./tokenizer.json)
- [debug_dataloaders.py](./debug_dataloaders.py)
- [cleaned_bhagwat_gita.txt](./cleaned_bhagwat_gita.txt)

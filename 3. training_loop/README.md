- [bhagavad gita data source: archive.org](https://archive.org/stream/bhagwat-gita-in-nepali/bhagwat%20gita%20in%20NEPALI_djvu.txt)

# modification in previous_chapters.py: create_dataloader_v2()

- replace data*files from:
  `data_files = {"train": base_url + "nepberta*" + str(context_length) + ".parquet"}`
- to:
  `data_files = {"train": base_url + "iriisnepal_u_nepberta_512.parquet"}`

#%%
from pathlib import Path

import umap
import zarr
import pandas as pd
#%%
ds_path = Path("/Users/alexander.hillsley/Documents/CZBiohub/projects/cfib/data/analysis/2-dataset/pilot_1_dataset.zarr")
ds_store = zarr.open(ds_path)
position_dict = ds_store.attrs['labels']

label_dict = {}
for _, val in position_dict.items():
    if ds_path.name[:7] == 'pilot_1':
        if val.split('/')[1] == '1' or val.split('/')[1] == '6':
            label_dict[val] = '1-no_stimulus'
        if val.split('/')[1] == '2' or val.split('/')[1] == '7':
            label_dict[val] = '2-5ngul_tgfb'
        if val.split('/')[1] == '3' or val.split('/')[1] == '8':
            label_dict[val] = '3-10ngul_tgfb'
        if val.split('/')[1] == '4' or val.split('/')[1] == '9':
            label_dict[val] = '4-20ngul_tgfb'
        if val.split('/')[1] == '5' or val.split('/')[1] == '10':
            label_dict[val] = '5-Ang_II'
# %%
def generate_labels(
        dataset_path: str,
)->None:
    ds_store = zarr.open(ds_path)
    position_dict = ds_store.attrs['labels']

    label_dict = {}
    for _, val in position_dict.items():
        if ds_path.name[:7] == 'pilot_1':
            if val.split('/')[1] == '1' or val.split('/')[1] == '6':
                label_dict[val] = '1-no_stimulus'
            if val.split('/')[1] == '2' or val.split('/')[1] == '7':
                label_dict[val] = '2-5ngul_tgfb'
            if val.split('/')[1] == '3' or val.split('/')[1] == '8':
                label_dict[val] = '3-10ngul_tgfb'
            if val.split('/')[1] == '4' or val.split('/')[1] == '9':
                label_dict[val] = '4-20ngul_tgfb'
            if val.split('/')[1] == '5' or val.split('/')[1] == '10':
                label_dict[val] = '5-Ang_II'
    
    label_df = pd.DataFrame(label_dict, index=['label']).T

    return
# %%

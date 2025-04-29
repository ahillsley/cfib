from pathlib import Path
import glob
import tifffile
import zarr
import numpy as np
from tqdm import tqdm

from iohub import open_ome_zarr


def create_zarr(
        dir_path: str,
        output_path,
) -> None:

    tif_path_list = glob.glob(dir_path + "/*")

    position_dict = {}
    for t in tif_path_list:
        t = Path(t)
        if t.suffix != '.tif':
            continue
        row = t.name[0]
        col = t.name[1]
        fov = t.name.split("--")[2][1:]
        channel = t.name.split("--")[-1].split(' ')[0]
        pos_name = f'{row}/{col}/{fov}'

        if pos_name not in position_dict:
            position_dict[pos_name] = {}
        
        position_dict[pos_name][channel] = t

    channel_names = list(position_dict[pos_name].keys())

    output_store = open_ome_zarr(
        output_path, layout='hcs', mode='w', channel_names=channel_names
        )
    
    for key, val in tqdm(position_dict.items()):
        array_list = []
        for c in channel_names:
            tif_path = val[c]
            tif_array = tifffile.imread(tif_path)
            array_list.append(np.expand_dims(tif_array, 0))
        tif_array_3c = np.concatenate(array_list, axis=0)
        tif_array_ome = np.expand_dims(tif_array_3c, axis=(0,2))

        output_pos = output_store.create_position(*Path(key).parts)
        output_pos.create_image(
            name='0',
            data=tif_array_ome,
            chunks = (1,1,1, 1024, 1344)
        )

    return

if __name__ == "__main__":
    dir_path = '/Users/alexander.hillsley/Documents/CZBiohub/projects/cfib/data/pilot_2'
    output_path = '/Users/alexander.hillsley/Documents/CZBiohub/projects/cfib/data/analysis/pilot_2.zarr'
    create_zarr(dir_path, output_path)

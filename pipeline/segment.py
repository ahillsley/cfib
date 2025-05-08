# %%
import cellpose
from cellpose import models
from tqdm import tqdm
from pathlib import Path
import zarr
import torch
import numpy as np
import matplotlib.pyplot as plt
from iohub import open_ome_zarr
from instanseg import InstanSeg

# TODO: segmentations are not optimal, some cells are being missed


def segment_nuclei(
    store_path: str,
    seg_path: str,
    model_type: str = "instanseg",
) -> None:

    store = open_ome_zarr(store_path)
    pos_list = [a[0] for a in store.positions()]

    seg_store = open_ome_zarr(
        seg_path, layout="hcs", mode="w", channel_names=["Segmentation"]
    )
    if model_type == "cellpose":
        model_cellpose = models.CellposeModel(
            gpu=True,
            model_type="nuclei",
        )
    if model_type == "instanseg":
        model_instanseg = InstanSeg("fluorescence_nuclei_and_cells", device="cpu")

    for pos in tqdm(pos_list):
        nuclei = store[pos].data[0, 1, 0, :, :]
        if model_type == "cellpose":
            out, a, b = model_cellpose.eval(
                nuclei > 50, flow_threshold=0.9, diameter=80, min_size=50
            )
            out = np.expand_dims(out, axis=(0, 1, 2))
        if model_type == "instanseg":
            out = model_instanseg.eval_small_image(
                nuclei, normalise=True, pixel_size=0.3, return_image_tensor=False
            )
            out = np.expand_dims(out.numpy()[0, 0, :, :], axis=(0, 1, 2))
        seg_pos = seg_store.create_position(*Path(pos).parts)

        # TODO: segs are being stored as float32, should be int16 or even int8

        seg_pos.create_image(name="0", data=out, chunks=(1, 1, 1, 1024, 1344))


if __name__ == "__main__":
    store_path = "/Users/alexander.hillsley/Documents/CZBiohub/projects/cfib/data/analysis/pilot_1.zarr"
    seg_path = "/Users/alexander.hillsley/Documents/CZBiohub/projects/cfib/data/analysis/1-segmentation/pilot_1_seg.zarr"

    segment_nuclei(store_path, seg_path, model_type="instanseg")

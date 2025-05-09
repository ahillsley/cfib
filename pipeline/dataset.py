# %%
import numpy as np
from skimage.measure import regionprops
from iohub import open_ome_zarr
import matplotlib.pyplot as plt
import zarr
from tqdm import tqdm
import pandas as pd


def create_dataset(
    store_path: str,
    seg_path: str,
    dataset_path: str,
    window: int = 64,
) -> None:
    """
    window represents half the final image size i.e. window=64 -> final crop 128x128
    """

    fov_store = open_ome_zarr(store_path)
    seg_store = open_ome_zarr(seg_path)
    pos_list = [a[0] for a in fov_store.positions()]

    cell_count = 0
    cell_props = {}
    fov_normalization_params = {}

    output_zarr = zarr.DirectoryStore(dataset_path)
    output_store = zarr.group(
        store=output_zarr, overwrite=True
    )  # will not ammend, will create a new zarr
    zattrs_dict = {}
    labels = {}

    for pos in tqdm(pos_list):

        output_store.create_group(pos)

        segmentation = seg_store[pos].data[0, 0, 0, :, :].astype(np.uint16)
        fov = fov_store[pos].data[0, :, 0, :, :]

        # normalize each image between 0 and 1
        fov_norm = (fov - np.expand_dims(np.min(fov, axis=(1, 2)), (1, 2))) / (
            np.expand_dims(np.max(fov, axis=(1, 2)) - np.min(fov, axis=(1, 2)), (1, 2))
        )
        channel_means = np.mean(fov_norm, axis=(1, 2))
        channel_stds = np.std(fov_norm, axis=(1, 2))
        mean_intensity = {
            0: channel_means[0],
            1: channel_means[1],
            2: channel_means[2],
        }
        std_intensity = {
            0: channel_stds[0],
            1: channel_stds[1],
            2: channel_stds[2],
        }
        fov_normalization_params[pos] = {"mean": mean_intensity, "std": std_intensity}

        fov_props = regionprops(segmentation)
        for i, obj in enumerate(fov_props):

            croid = [int(a) for a in obj.centroid]
            cell_props[cell_count] = {"fov": pos, "centroid": croid}

            i_min = np.clip(
                croid[0] - window, a_min=0, a_max=fov.shape[-2] - 2 * window
            )
            i_max = i_min + window * 2
            j_min = np.clip(
                croid[1] - window, a_min=0, a_max=fov.shape[-1] - 2 * window
            )
            j_max = j_min + 2 * window

            sc_crop = fov_norm[:, i_min:i_max, j_min:j_max]

            output_store[pos][str(cell_count).zfill(5)] = sc_crop

            labels[cell_count] = f"{pos}/" + str(cell_count).zfill(5)

            cell_count += 1

    zattrs_dict["normalization_params"] = fov_normalization_params
    zattrs_dict["num_cells"] = cell_count - 1
    zattrs_dict["labels"] = labels

    output_store.attrs.update(zattrs_dict)

    return


if __name__ == "__main__":
    store_path = "/Users/alexander.hillsley/Documents/CZBiohub/projects/cfib/data/analysis/pilot_1.zarr"
    seg_path = "/Users/alexander.hillsley/Documents/CZBiohub/projects/cfib/data/analysis/1-segmentation/pilot_1_seg.zarr"
    dataset_path = "/Users/alexander.hillsley/Documents/CZBiohub/projects/cfib/data/analysis/2-dataset/pilot_1_dataset.zarr"

    create_dataset(store_path=store_path, seg_path=seg_path, dataset_path=dataset_path)

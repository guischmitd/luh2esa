import matplotlib.pyplot as plt

from PIL import Image
import seaborn as sns
import numpy as np

from recover.dataset import RasterDataset

def set_plot_defaults():
    sns.set_palette('dark')
    sns.set_style('darkgrid')
    plt.rcParams['figure.figsize'] = (10, 8)

def plot_single_band(img, cmap='cividis'):
    return plt.imshow(img, cmap=cmap)
    
def plot_cell_over_time(ds: RasterDataset, cell_id: int | None = None, bands: list[str] | str | None = None):
    bands = bands or ds.bands
    if isinstance(bands, str):
        bands = [bands]

    cell_id = cell_id or ds.data.sample(1).index[0]
    return (
        ds.data
        .loc[cell_id]
        .set_index(ds.time_col)
        .sort_index()
        [bands]
        .plot(title=f'Cell {cell_id} fractions over time')
    )

def animate_band(ds, band, out_path, cmap='viridis'):
    colormap = plt.colormaps[cmap]
    
    arrays = [
        ds.arr[i, ds.bands.index(band), :, :]
        for i in range(len(ds.time_steps))
    ]

    colorised = [
        np.uint8(colormap(arr) * 255)
        for arr in arrays
    ]

    imgs = [
        Image.fromarray(arr).resize((ds.width * 4, ds.height * 4), resample=Image.Resampling.NEAREST)
        for arr in colorised]

    imgs[0].save(str(out_path), save_all=True, append_images=imgs[1:], duration=50, loop=0)
    return
    
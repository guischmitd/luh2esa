import matplotlib.pyplot as plt
import seaborn as sns

from recover.dataset import RasterDataset

def set_plot_defaults():
    sns.set_palette('dark')
    sns.set_style('darkgrid')
    plt.rcParams['figure.figsize'] = (10, 8)

def plot_single_band(img, cmap='cividis'):
    return plt.imshow(img, cmap=cmap)
    
def plot_cell_over_time(ds: RasterDataset, cell_id: int, bands: list[str] | str | None = None):
    bands = bands or ds.bands
    if isinstance(bands, str):
        bands = [bands]
    return (
        ds.data
        .loc[cell_id]
        .set_index(ds.time_col)
        .sort_index()
        [bands]
        .plot()
    )

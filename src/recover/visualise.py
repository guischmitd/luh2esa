import matplotlib.pyplot as plt

from PIL import Image, ImageDraw, ImageFont
import seaborn as sns
import numpy as np

from recover.dataset import RasterDataset

FONTDIR = '/mnt/c/Windows/Fonts/'

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

def animate_band(ds, band, out_path, cmap='viridis', fps=3.0):
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

    font = ImageFont.truetype(FONTDIR + 'CascadiaMono.ttf', 20)    
    for img, year in zip(imgs, ds.time_steps):
        draw = ImageDraw.Draw(img)
        bbox = font.getbbox(str(int(year)))
        anchor = (img.size[0] - 100, 15)
        rect = (
            bbox[0] + anchor[0], 
            bbox[1] + anchor[1], 
            bbox[2] + anchor[0], 
            bbox[3] + anchor[1], 
        )
        draw.rectangle(rect, fill='black')
        draw.text(anchor, text=str(int(year)), fill='white', font=font)

    imgs[0].save(
        str(out_path), 
        save_all=True, 
        append_images=imgs[1:],
        duration=1000 / fps, 
        loop=0)
    
    return out_path

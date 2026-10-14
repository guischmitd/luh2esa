import matplotlib.pyplot as plt

from recover.dataset import RasterDataset

def plot_single_band(img, cmap='cividis'):
    return plt.imshow(img, cmap=cmap)
    
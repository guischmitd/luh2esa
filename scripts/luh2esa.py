import pandas as pd
import rasterio

from recover.constants import DATA_DIR
from recover.dataset import raster_to_df

def load_data():
    luh_raw = pd.read_parquet(DATA_DIR / 'luh_timeseries.parquet')
    esa_raw = pd.read_parquet(DATA_DIR / 'esa_timeseries.parquet')

    biome_data = raster_to_df(DATA_DIR / 'NTAmericas_efgs_50km.tif')
    biome_classes = pd.read_excel(DATA_DIR / 'ESA-LC_Biome_proportions.xlsx')

    return luh_raw, esa_raw, biome_data, biome_classes

def preprocess():
    
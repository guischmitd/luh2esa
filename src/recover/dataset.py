from pathlib import Path
from pprint import pprint

import pandas as pd
import numpy as np
import rasterio as rio

def cell2idx(cell_id: int, cols: int = 180):
    """
    Given an integer cell_id and a raster width in pixels (cols) 
    return the (i, j) coordinates of the cell.
    """
    j = int(cell_id % cols)
    i = int(cell_id // cols)
    return i, j

def raster_to_df(raster_path: Path):
    """
    Reads a **single-band** raster and converts it to the standard 
    dataframe format with cell_id as the index. 
    """
    # TODO ensure multiband is interpreted correctly
    with rio.open(raster_path, 'r') as raster:
        data = raster.read()
        pprint(raster.meta)
        null_value = raster.meta.get('nodata')

    h, w = data.shape[-2:]
    df = pd.DataFrame(data.ravel(), index=range(1, 1 + h * w), columns=[raster_path.stem])
    df = df.replace(null_value, None).dropna()
    df.index.name = 'cell'
    return df


class RasterDataset:
    def __init__(self, 
                 df: pd.DataFrame, 
                 id_col: str = 'cell', 
                 width: int = 180, 
                 height: int = 157, 
                 time_col = 'year'):
        self.data = df.set_index(id_col) if df.index.name != id_col else df
        self.width = width
        self.id_col = id_col
        self.height = height
        self.time_col = time_col
        self.time_steps = sorted(df[self.time_col].unique().tolist())  
        self.bands = [c for c in df.columns if c not in [time_col, id_col]]
        print(f"Initialising raster dataset with bands {self.bands}")
        
        self.arr = self.__init_array()
        print(f'Finished building array with shape {self.arr.shape}')

    def __init_array(self):
        spine = pd.DataFrame(index=range(1, self.width * self.height + 1))
        
        arr = np.full((len(self.time_steps), len(self.bands), self.height, self.width), 255.0, dtype=np.float32)
        for i, t in enumerate(self.time_steps):
            year_data = spine.join(self.data.query(f"{self.time_col} == {t}"), 
                                    how='left').copy()
            
            for j, b in enumerate(self.bands):
                band_array = (
                    year_data.
                    sort_index()
                    [b].values.reshape((self.height, self.width))
                    )
                arr[i, j, :, :] = band_array

        return arr
    
    def to_file(self, out_path: Path, reference_raster: Path):
        with rio.open(reference_raster, 'r') as ref:
            meta = ref.meta
        with rio.open(out_path, 'w') as out:
            out.write(self.arr)
    
    @property
    def pixel_count(self):
        """Counts the number of non-null pixels"""
        return (~np.isnan(self.arr.mean(axis=0).mean(axis=0))).sum()

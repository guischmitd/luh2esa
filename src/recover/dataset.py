from abc import ABC, abstractmethod

import pandas as pd
import numpy as np


class RasterDataset:
    def __init__(self, df: pd.DataFrame, x_col: str = 'x', y_col: str = 'y', t_col = 'year'):
        self.data = df
        self.x_col = x_col
        self.y_col = y_col
        self.t_col = t_col
        self.time_steps = sorted(df[self.t_col].unique().tolist())  
        self.bands = [c for c in df.columns if c not in [x_col, y_col, t_col]]
        print(f"Initialising raster dataset with bands {self.bands}")
        
        self.__init_image_attrs()
        self.arr = self.__init_array()
        print(f'Finished building array with shape {self.arr.shape}')

    def __init_image_attrs(self):
        self.W, self.H = self.data[self.x_col].nunique(), self.data[self.y_col].nunique()

        unique_x = self.data[self.x_col].drop_duplicates()
        unique_y = self.data[self.y_col].drop_duplicates()

        self.XMAP = unique_x.rank().astype(int) - 1
        self.XMAP.index = unique_x.values

        self.YMAP = unique_y.rank(ascending=False).astype(int) - 1
        self.YMAP.index = unique_y.values
        
        self.unique_x = unique_x.values
        self.unique_y = unique_y.values

    def __init_array(self):
        mesh = np.meshgrid(self.unique_y, self.unique_x)
        spine = pd.DataFrame([arr.flatten() for arr in mesh], index=['y', 'x']).T
        
        arr = np.full((len(self.time_steps), len(self.bands), self.H, self.W), np.nan, dtype=np.float32)
        for i, t in enumerate(self.time_steps):
            year_data = spine.merge(self.data.query(f"{self.t_col} == {t}"), 
                                    on=['x', 'y'],
                                    how='left').copy()
            for j, b in enumerate(self.bands):
                band_array = (
                    year_data.set_index(['y', 'x'])
                    .sort_index(ascending=[False, True])        
                    [b].values.reshape((self.H, self.W))
                    )
                arr[i, j, :, :] = band_array

        return arr
    
    @property
    def pixel_count(self):
        """Counts the number of non-null pixels"""
        return (~np.isnan(self.arr.mean(axis=0).mean(axis=0))).sum()

    def sample_pixels(self):
        return self.data

    def sample_timesteps(self):
        pass

def sample_years(dataset, years: list[int|str]):
    pass

def sample_patch(dataset, bounds):
    pass
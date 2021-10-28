#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8

"""
Copyright © 2021 Daniel Santiago <http://github.com/dspelaez>
Distributed under terms of the GNU/GPL 3.0 license.

@author: Daniel Santiago
@github: http://github.com/dspelaez
@created: 2021-10-14
"""

import numpy as np
import xarray as xr
import pandas as pd
import dask.dataframe as dd
import datetime as dt
import matplotlib.pyplot as plt
import multiprocessing as mp
import requests
import tarfile
import tqdm
import os
from scipy.io import wavfile
from io import BytesIO
from zipfile import ZipFile
plt.ion()


BASE_URL = "https://spiddal.marine.ie/data/hydrophones"
BASE_PATH = os.path.expanduser("~/Workspace/data/smartbay/hydrophone")
CACHE_DIRECTORY = os.path.expanduser("~/Workspace/data/cache")


class SmartBayHydrophone(object):

    def __init__(self, date, hydrophone="SBF1622"):
        """Class constructor"""
        self.date = date
        self.hydrophone = hydrophone
        self._cache_directory = CACHE_DIRECTORY
        

    def get_filenames(self):
        """Return list of available file names for the diven day"""

        base_url = BASE_URL
        zip_url = (
            f"{base_url}/{self.hydrophone}/{self.date:%Y/%m/%d}/"
            f"{self.hydrophone}_{self.date:%Y%m%d}.zip"
        )
        content = requests.get(zip_url)
        zf = ZipFile(BytesIO(content.content))

        return [
            f"{base_url}/{self.hydrophone}/{self.date:%Y/%m/%d}/{item}"
            for item in zf.namelist() if item.endswith(".txt")
        ]

    def read_local_file(self):
        """Return class for a givel local filename"""
        
        # get file names
        base_path = BASE_PATH
        fname = f"{base_path}/{self.hydrophone}_{self.date:%Y%m%d}.tgz"
        with tarfile.open(fname, "r:*") as tar:
            fnames = tar.getnames()
            self.data = xr.concat(
                [
                    self.read_csv(tar.extractfile(fname)) 
                    for fname in tqdm.tqdm(fnames)
                ],
                dim="time"
            )

        return self.data


    def read_csv(self, fname):
        """Return hydrophone pandas data"""

        date_parser = lambda x: self.date.strftime("%Y-%m-%d ") + x
        try:
            # read pandas object
            df = pd.read_csv(
                fname, sep="\t", skiprows=29, index_col="Time",
                parse_dates=True, date_parser=date_parser
            )
            #
            # drop fist six columns
            df.drop(df.columns[:5], axis=1, inplace=True)
            #
            # return object
            return  xr.DataArray(
                data = df.values.astype("float"),
                dims = ["time", "frqs"],
                coords = {
                    "time": df.index.values,
                    "frqs": df.columns.values.astype("float")
                }
            )
            
        except Exception as e:
            print(e)


    def download(self):
        """Download one day of hydrophone data"""

        # check cache
        ncname = f"_{self.hydrophone}_{self.date:%Y%m%d.nc}"
        if os.path.isfile(f"{self._cache_directory}/{ncname}"):
            self.data = xr.open_dataarray(f"{self._cache_directory}/{ncname}")
        else:
            #
            # download data
            fnames = self.get_filenames()
            self.data = xr.concat(
                [self.read_csv(fname) for fname in tqdm.tqdm(fnames)],
                dim="time"
            )
            # save to file
            self.data.to_netcdf(f"{self._cache_directory}/{ncname}")

        # return data object
        return self.data

    
    def plot_spectrogram(
            self, zoom=None, vmin=10, vmax=30, cmap="magma", **kwargs
        ):
        """Plot spectrogram"""

        fig, ax = plt.subplots(figsize=(7,4))
        if zoom is None:
            self.data.plot(
                x="time", vmin=vmin, vmax=vmax, cmap=cmap,
                cbar_kwargs={'label': "dB ref 1V -120"}, **kwargs
            )
        else:
            self.data.sel(time=zoom).plot(
                x="time", vmin=vmin, vmax=vmax, cmap=cmap,
                cbar_kwargs={'label': "dB ref 1V -120"}, **kwargs
            )
        ax.set_ylabel("Frequency [Hz]")
        return fig, ax


    def sound_wave(self, sampling_rate=512000):
        """Return sound wave with zero-phase"""
        
        s = slice(1,-1,4)
        t = self.data.time.values.astype("float")[s]
        t = (t - t[0]) / sampling_rate

        intensity = self.data.values[s,:]
        intensity = intensity - intensity.min()
        intensity = intensity / intensity.max()
        
        sound_wave = np.zeros_like(t)
        for i, frq in enumerate(self.data.frqs.values):
            sound_wave += intensity[:,i] * np.cos(2*np.pi * frq * t)

        wavfile.write("test.wav", int(sampling_rate/1000), sound_wave)



def smartbay_hourly_average(date):
    """Return an hourly dataset containing spectrograms"""
    try:
        outname = f"{BASE_PATH}/hourly/{date:%Y%m%d}.nc"
        if not os.path.exists(outname):
            return (
                SmartBayHydrophone(date, "SBF1622")
                .read_local_file()
                .resample(time="1H")
                .mean("time")
                .to_netcdf(outname)
            )
    except Exception as e:
        print(e)

def paralell_smartbay_hourly():
    """Try to paralellize"""
    
    t_beg = dt.date(2019,10,1)
    t_end = dt.date(2020,8,12)
    numdays = (t_end - t_beg).days + 1
    dates = [t_beg + dt.timedelta(days=n) for n in range(numdays)]

    pool = mp.Pool(mp.cpu_count())
    results = pool.map(smartbay_hourly_average, dates)
    
    pool.close()
        



if __name__ == "__main__":
    paralell_smartbay_hourly()
    
    # Available: SBF1622 or SBF1323
    # date = dt.date(2020, 8, 12)
    # date = dt.date(2021, 10, 1)
    # self = SmartBayHydrophone(date, "SBF1622")
    # self.download()
    # self.plot_spectrogram()


# -eof

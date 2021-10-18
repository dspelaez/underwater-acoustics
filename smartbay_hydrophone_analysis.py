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
import requests
import tqdm
import os
from scipy.io import wavfile
from io import BytesIO
from zipfile import ZipFile
plt.ion()


class SmartBayHydrophone(object):

    def __init__(self, date, hydrophone="SBF1622"):
        """Class constructor"""
        self.date = date
        self.hydrophone = hydrophone
        self._cache_directory = os.path.expanduser("~/Workspace/data/cache")
        

    def get_filenames(self):
        """Return list of available file names for the diven day"""

        base_url = "https://spiddal.marine.ie/data/hydrophones"
        zip_url = (
            f"{base_url}/{self.hydrophone}/{self.date:%Y/%m/%d}/"
            f"{self.hydrophone}_{date:%Y%m%d}.zip"
        )
        content = requests.get(zip_url)
        zf = ZipFile(BytesIO(content.content))

        return [
            f"{base_url}/{self.hydrophone}/{self.date:%Y/%m/%d}/{item}"
            for item in zf.namelist() if item.endswith(".txt")
        ]


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
        """Download one-lenght day of hydrophone data"""

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

    
    def plot_spectrogram(self, vmin=10, vmax=30, cmap="magma"):
        """Plot spectrogram"""
        if hasattr(self, "data"):
            fig, ax = plt.subplots(figsize=(7,4))
            self.data.plot(x="time", vmin=vmin, vmax=vmax, cmap=cmap)
            ax.set_ylabel("Frequency [Hz]")

        else:
            print("No data available: run download first")


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


class SoundFile(object):
    """Handle wav files"""

    def __init__(self, fname):
        """Class constructor"""
        self.fname = fname
        self._read()

    def _read(self):
        """Read sound wave file"""
        self.sampling_rate, self.data = wavfile.read(self.fname)
        if self.data.ndim > 1:
            self.data = self.data[:,0]

    def plot_spectrogram(self, vmin=-4, vmax=4, cmap="magma", nperseg=256):
        """Plot spectrogram"""

        if hasattr(self, "data"):
            self.frqs, self.time, self.power = signal.spectrogram(
                self.data, fs=self.sampling_rate, nperseg=nperseg
            )
            fig, ax = plt.subplots(figsize=(7,4))
            ax.pcolormesh(
                self.time, self.frqs, np.log10(self.power),
                vmin=vmin, vmax=vmax, cmap=cmap
            )
            ax.set_ylabel("Frequency [Hz]")

        else:
            print("No data available")


if __name__ == "__main__":
    
    # Available: SBF1622 or SBF1323
    date = dt.date(2020, 8, 12)
    # date = dt.date(2021, 10, 1)
    self = SmartBayHydrophone(date, "SBF1622")
    self.download()
    self.plot_spectrogram()


# -eof

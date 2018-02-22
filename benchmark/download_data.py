#!/usr/bin/env python

import shutil
import os

import requests


def download_url(url, filename):
    print("Downloading file from {}...".format(url))
    response = requests.get(url, stream=True)
    with open(filename, "wb") as outfile:
        shutil.copyfileobj(response.raw, outfile)


if not os.path.exists("benchmark/data"):
    os.mkdir("benchmark/data")

download_url("http://archive.ics.uci.edu/ml/machine-learning-databases/00265/CASP.csv",
             "benchmark/data/CASP.csv")

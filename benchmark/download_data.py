#!/usr/bin/env python

import re
import zipfile
import os
import glob
import json

import requests
import progressbar


def download_dataset(meta):
    print("Downloading file from {}...".format(meta["url"]))
    response = requests.get(meta["url"], stream=True)
    file_size = int(response.headers.get("content-length"))
    file_name = "benchmark/data/{name}.{extension}".format(**meta)
    with open(file_name, "wb") as outfile:
        downloaded = 0
        bar = progressbar.ProgressBar(max_value=file_size)
        for block in response.iter_content(chunk_size=4096):
            downloaded += len(block)
            outfile.write(block)
            bar.update(downloaded)
    if meta["extension"] == "zip":
        _unzip(meta)
        os.remove(file_name)
    if "train_size" in meta:
        _split(meta)


def _split(meta):
    name = meta["name"]
    main = f"benchmark/data/{name}.csv"
    train = f"benchmark/data/{name}-train.csv"
    test = f"benchmark/data/{name}-test.csv"
    with open(main, "r") as main_file:
        if meta["header"]:
            main_file.readline()
        with open(train, "w") as train_file:
            for _ in range(meta["train_size"]):
                train_file.write(main_file.readline())
        with open(test, "w") as test_file:
            for line in main_file:
                test_file.write(line)
    os.remove(main)


def _match_names(pattern, members):
    pattern_clean = pattern.replace(".", r"\.").replace("*", ".*")
    return [member for member in members if re.match(pattern_clean, member)]


def _unzip(meta):
    name = meta["name"]
    filenames = meta["filenames"]
    with zipfile.ZipFile(f"benchmark/data/{name}.zip") as infile:
        if "all" in filenames:
            names = _match_names(filenames["all"], infile.namelist())
            _unzip_files(infile, names, f"benchmark/data/{name}.csv", meta["header"])
        else:
            train_names = _match_names(filenames["train"], infile.namelist())
            _unzip_files(infile, train_names, f"benchmark/data/{name}-train.csv", meta["header"])
            test_names = _match_names(filenames["test"], infile.namelist())
            _unzip_files(infile, test_names, f"benchmark/data/{name}-test.csv", meta["header"])


def _unzip_files(zip_file, names, dest, header=False):
    with open(dest, "wb") as dest_file:
        first_name = True
        for name in names:
            with zip_file.open(name, "r") as source_file:
                if header:
                    if first_name:
                        first_name = False
                    else:
                        source_file.readline()
                for line in source_file.readlines():
                    dest_file.write(line)


if __name__ == "__main__":
    meta_files = glob.glob("benchmark/data/*-meta.json")
    for filename in meta_files:
        with open(filename, "r") as file:
            metadata = json.load(file)
            download_dataset(metadata)

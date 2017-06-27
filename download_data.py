import urllib.request
import zipfile
import os

SOURCE_DIRECTORY = "http://archive.ics.uci.edu/ml/machine-learning-databases/00203/"
FILENAME = "YearPredictionMSD.txt.zip"
TARGET_DIRECTORY = "data/"

# Create the directory for the data
if not os.path.exists(TARGET_DIRECTORY):
    os.makedirs(TARGET_DIRECTORY)

# Download the data
urllib.request.urlretrieve(SOURCE_DIRECTORY + FILENAME,
                           TARGET_DIRECTORY + FILENAME)

# Extract the data
with zipfile.ZipFile(TARGET_DIRECTORY + FILENAME, "r") as zip_ref:
    zip_ref.extractall(TARGET_DIRECTORY)

# Delete the zipfile
os.remove(TARGET_DIRECTORY + FILENAME)

"""Download MNIST and write the PS4 digit subset and logbook entries to data/.

Run: python3 -I instructor/import_data.py
Only the Python standard library is required. Download the four original gzip
files from the mirror into a temporary folder, never from a local copy: this
machine also holds Fashion-MNIST files with the same names. Check each source
hash, then overwrite the shipped files in data/. Failed checks raise
AssertionError; download and file errors propagate.
"""
from pathlib import Path
import csv
import gzip
import hashlib
import json
import tempfile
import urllib.request

MIRROR = "https://storage.googleapis.com/cvdf-datasets/mnist/"
SOURCES = {
    "train-images-idx3-ubyte.gz": "440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609",
    "train-labels-idx1-ubyte.gz": "3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c",
    "t10k-images-idx3-ubyte.gz": "8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6",
    "t10k-labels-idx1-ubyte.gz": "f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6",
}
TRAINING_IMAGES = 10_000  # the first 10,000 official training images
PIXELS = 28 * 28

# Five hourly logbook rows of four readings; every value has exactly four digits -
QUANTITIES = [("Reactor temperature", "K"), ("Reactor pressure", "kPa"),
              ("Feed flow", "L/h"), ("Coolant temperature", "K")]
READINGS = {
    "06:00": ["342.5", "1013", "1840", "298.4"],
    "07:00": ["345.1", "1027", "1835", "297.9"],
    "08:00": ["348.6", "1046", "1852", "298.7"],
    "09:00": ["351.2", "1069", "1861", "299.3"],
    "10:00": ["347.8", "1054", "1829", "298.1"],
}

root = Path(__file__).resolve().parents[1]
data = root / "data"
data.mkdir(exist_ok=True)


def sha256(path):
    """Return the hex SHA-256 digest of the file at path."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_idx(path, magic):
    """Return (count, body bytes) from an IDX gzip file after checking its header.

    magic is 2051 for images (also checks 28 x 28) or 2049 for labels.
    """
    raw = gzip.open(path).read()
    assert int.from_bytes(raw[0:4], "big") == magic, f"{path.name}: wrong IDX type"
    count = int.from_bytes(raw[4:8], "big")
    offset = 8
    if magic == 2051:
        rows, cols = int.from_bytes(raw[8:12], "big"), int.from_bytes(raw[12:16], "big")
        assert (rows, cols) == (28, 28), f"{path.name}: images are {rows} x {cols}"
        offset = 16
    body = raw[offset:]
    assert len(body) == count * (PIXELS if magic == 2051 else 1), f"{path.name}: wrong length"
    return count, body


# 1. Download and check the original files -
with tempfile.TemporaryDirectory(prefix="ps4-mnist-") as temporary:
    folder = Path(temporary)
    for name, digest in SOURCES.items():
        urllib.request.urlretrieve(MIRROR + name, folder / name)
        assert sha256(folder / name) == digest, f"{name}: hash does not match"
        print(f"{name}: hash OK")
    _, train_images = read_idx(folder / "train-images-idx3-ubyte.gz", 2051)
    _, train_labels = read_idx(folder / "train-labels-idx1-ubyte.gz", 2049)
    test_count, test_images = read_idx(folder / "t10k-images-idx3-ubyte.gz", 2051)
    _, test_labels = read_idx(folder / "t10k-labels-idx1-ubyte.gz", 2049)

# 2. Write the training subset and the full test set as raw bytes -
shipped = {
    "train-images.bin": train_images[:TRAINING_IMAGES * PIXELS],
    "train-labels.bin": train_labels[:TRAINING_IMAGES],
    "test-images.bin": test_images,
    "test-labels.bin": test_labels,
}
assert test_count == 10_000
for name, body in shipped.items():
    (data / name).write_bytes(body)
    print(f"{name}: {len(body):,} bytes")

# 3. Build the logbook page: each digit uses the first unused test image with that label -
used = set()
rows = []
for time, values in READINGS.items():
    for (quantity, unit), value in zip(QUANTITIES, values):
        digits = [int(c) for c in value if c.isdigit()]
        assert len(digits) == 4, f"{value} must have four digits"
        images = []
        for digit in digits:
            number = next(i + 1 for i, label in enumerate(test_labels) if label == digit and i + 1 not in used)
            used.add(number)
            images.append(number)
        rows.append([time, quantity, unit, value, *images])
with open(data / "logbook.csv", "w", newline="") as handle:
    writer = csv.writer(handle, lineterminator="\n")
    writer.writerow(["time", "quantity", "unit", "value", "image_1", "image_2", "image_3", "image_4"])
    writer.writerows(rows)
print(f"logbook.csv: {len(rows)} entries")

# 4. Record where the files came from -
provenance = {
    "dataset": "MNIST database of handwritten digits",
    "dataset_page": "http://yann.lecun.com/exdb/mnist/",
    "creators": ["Yann LeCun", "Corinna Cortes", "Christopher J. C. Burges"],
    "citation": {
        "authors": ["Yann LeCun", "Léon Bottou", "Yoshua Bengio", "Patrick Haffner"],
        "title": "Gradient-based learning applied to document recognition",
        "journal": "Proceedings of the IEEE",
        "year": 1998,
        "volume": 86,
        "issue": 11,
        "pages": "2278-2324",
        "doi": "10.1109/5.726791",
    },
    "license": "Creative Commons Attribution-Share Alike 3.0 (CC BY-SA 3.0), "
               "https://creativecommons.org/licenses/by-sa/3.0/",
    "license_source": "The dataset page has no license statement. The Keras MNIST loader documentation "
                      "(https://github.com/keras-team/keras/blob/master/keras/src/datasets/mnist.py) states: "
                      "Yann LeCun and Corinna Cortes hold the copyright of MNIST dataset, which is a derivative "
                      "work from original NIST datasets. MNIST dataset is made available under the terms of the "
                      "Creative Commons Attribution-Share Alike 3.0 license.",
    "download_mirror": MIRROR,
    "source_sha256": SOURCES,
    "selection": "first 10,000 official training images and labels; all 10,000 official test images and labels",
    "format": "raw bytes; images are 784 bytes each (28 rows of 28 pixels, row by row, 0 = white, 255 = black ink), "
              "stored back to back; labels are one byte (0-9) per image, in the same order",
    "logbook": "data/logbook.csv lists 20 four-digit readings; for each digit, in entry order, the first unused "
               "test image with that label (image numbers start at 1)",
    "shipped_sha256": {name: sha256(data / name) for name in [*shipped, "logbook.csv"]},
}
(data / "provenance.json").write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n")
print("provenance.json written")

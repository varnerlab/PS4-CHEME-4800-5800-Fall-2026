# Handwritten-digit data

Source: the [MNIST database of handwritten digits](http://yann.lecun.com/exdb/mnist/),
created by Yann LeCun, Corinna Cortes, and Christopher J. C. Burges.

Publication: LeCun, Y.; Bottou, L.; Bengio, Y.; Haffner, P. (1998).
**Gradient-based learning applied to document recognition.** *Proceedings of the
IEEE*, **86**(11), 2278–2324.
[DOI: 10.1109/5.726791](https://doi.org/10.1109/5.726791).

| File | Contents |
|---|---|
| `train-images.bin` | The first 10,000 MNIST training images. |
| `train-labels.bin` | The digit (0–9) in each training image, in the same order. |
| `test-images.bin` | All 10,000 MNIST test images. |
| `test-labels.bin` | The digit (0–9) in each test image, in the same order. |
| [logbook.csv](logbook.csv) | The 20 readings on the logbook page and the test images used to write them. |
| [provenance.json](provenance.json) | Where the files came from, with checksums that confirm the original and shipped files are unchanged. |

## File layout

Each image is 28 × 28 pixels, stored as 784 bytes: the first row of 28 pixels,
then the second row, and so on. A byte of 0 is white paper and 255 is black ink.
The images are stored one after another, with nothing before the first image.
Each label is one byte.

You do not need to read these files yourself. `load_digits()` in
[src/Data.jl](../src/Data.jl) reads them, scales each pixel to a number between
0 and 1, and adds the 1 for the bias.

## The logbook page

[logbook.csv](logbook.csv) has one row for each of the 20 readings on the page:
four readings (reactor temperature, reactor pressure, feed flow, and coolant
temperature) each hour from 06:00 to 10:00. Every reading has exactly four
digits. The model reads only the digits; any decimal point is copied as it is.
Columns `image_1` to `image_4` give the row of `data.test.X` used for each digit
(row 1 is the first test image). Going through the page digit by digit, each
digit uses the earliest test image of that digit not already used. The
handwriting on the page is MNIST handwriting, not real operators' handwriting.

## License

The MNIST data are shared under the Creative Commons Attribution-Share Alike 3.0
license. See [THIRD_PARTY_NOTICES.txt](../THIRD_PARTY_NOTICES.txt) for the full
notice and the changes made for PS4. The instructor can rebuild these files
from the original MNIST downloads.

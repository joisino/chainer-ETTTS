## Dataset

This document describes datasets of chainer-ETTTS.

* Each dataset should be placed in one directory.
* Each data in a dataset should consists of three files.
  * The first is the npy dump file of the normalized mel spectrogram of the speech data. The dtype is `np.float32` and the shape is `(80, T)`. The name of this file should end with `_mel.npy`.
  * The second is the npy dump file of the normalized amplitude spectrogram of the speech data. The dtype is `np.float32` and the shape is `(513, 4T)`. The name of this file should end with `_fft.npy`.
  * The second is the npy dump file of the text data. The dtype is `np.int32` and the shape is `(L,)`. The name of this file should end with `_txt.npy`.
  * The name of all of these files should have the same prefix. (e.g. `file001_mel.npy`, `file001_fft.npy`, `file001_txt.npy`.)
  * If you want to train only one of the two networks, `_fft.npy` is not needed for Text2Mel, and `_txt.npy` is not needed for SSRN.
* The directory of the dataset can contain other files such as `README.md`.

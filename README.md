# chainer-ETTTS

This is an implementation of [Efficiently Trainable Text-to-Speech System Based on Deep Convolutional Networks with Guided Attention](https://arxiv.org/abs/1710.08969) with chainer.

It can train a speech synthesis model with single GTX 1080Ti in a day.

`python 3.5.4` + `chainer 3.2.0`

## Demo

* [No Event Good Life (with LJSpeech pretrained model)](https://raw.githubusercontent.com/joisino/chainer-ETTTS/master/demo/noevent.ogg)
* [All your base are belong to us. (with LJSpeech pretrained model)](https://raw.githubusercontent.com/joisino/chainer-ETTTS/master/demo/aybabtu.ogg)
* [風船みたいな頭の子だな。 (with JSUT pretrained model)](https://raw.githubusercontent.com/joisino/chainer-ETTTS/master/demo/fuusen.ogg)
* [やっぱりもう少し考えて喋れ。 (with JSUT pretrained model)](https://raw.githubusercontent.com/joisino/chainer-ETTTS/master/demo/yappari.ogg)

## Installing dependencies

```
$ pip3 install -r requirements.txt
```

## Usage

First, you have to prepare dataset. If you want to use the [LJSpeech](https://keithito.com/LJ-Speech-Dataset/) dataset, you can use the following commands.

```
$ wget http://data.keithito.com/data/speech/LJSpeech-1.0.tar.bz2
$ tar xvf LJSpeech-1.0.tar.bz2
$ python3 ./preprocess_LJSpeech.py ./LJSpeech1.0 ./LJSpeech_data
```

If you want to use the [JSUT](https://sites.google.com/site/shinnosuketakamichi/publication/jsut) dataset, you can use the following commands.

```
$ wget http://ss-takashi.sakura.ne.jp/corpus/jsut_ver1.1.zip
$ unzip jsut_ver1.1
$ python3 ./preprocess_JSUT.py ./jsut_ver1.1/basic5000 ./JSUT_data
```

If you want to use original dataset, you can see `./sample_data`.

### Text2Mel

You can train Text2Mel model with `./Text2Mel/train.py`. If you want to train with the LJSpeech dataset, you can use following commands.

```
$ cd Text2Mel
$ python3 ./train.py --gpu 0 --data ../LJSpeech_data
```

If you want to use another dataset, you can replace `../LJSpeech_data` with the corresponding directory (e.g. `../JSUT_data`, `../sample_data`).

You can also download pretrained models.

```
$ wget https://www.dropbox.com/s/6wfisuis3t1q2wr/LJSpeech_text2mel_model
$ wget https://www.dropbox.com/s/wghw567rzolc6oh/JSUT_text2mel_model
```

The first is the LJSpeech model, which can generate woman English speech. The LJSpeech model is provided under the public domain.

The second is the JSUT model, which can generate woman Japanese speech. The JSUT model is licensed with the [CC-BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/).

### SSRN

You can train SSRN model with `./SSRN/train.py`. If you want to train with LJSpeech dataset, you can use following commands.

```
$ cd SSRN
$ python3 ./train.py --gpu 0 --data ../LJSpeech_data
```

If you want to use another dataset, you can replace `../LJSpeech_data` with the corresponding directory (e.g. `../JSUT_data`, `../sample_data`).

You can also download a pretrained model.

```
$ wget https://www.dropbox.com/s/5kcw54b5l9n5r1q/SSRN_model
```

This is the LJSpeech model, which can synthesis not only English speech but also Japanese speech etc. The LJSpeech model is provided under the public domain.

### Speech synthesis

```
$ cd Text2Mel
$ python3 ./generate.py --text "no event good life" --model ./results/model
```

First, you can make mel spectrogram with Text2Mel model. These commands generates mel spectrogram in `./Text2Mel/results/res.npy`.

The input text should be written in small capital roman alphabet. If you want to use Japanese characters, use `--ja` option.

```
$ cd ../SSRN
$ python3 ./generate.py --mel ../Text2Mel/results/res.npy --model ./results/model
```

Then, you can make wavefile with SSRN model. These commands generates audio file in `./SSRN/res.wav`.

If you want to use pretrained model or other models, you can replace `./results/model` with the corresponding file (e.g. `./LJSpeech_text2mel_model`, `./SSRN_model`)

## Bibliography

[1] Hideyuki Tachibana, Katsuya Uenoyama, Shunsuke Aihara. [Efficiently Trainable Text-to-Speech System Based on Deep Convolutional Networks with Guided Attention](https://arxiv.org/abs/1710.08969). In ICASSP 2018.

the original paper

[2] Keith Ito. [The LJ Speech Dataset](https://keithito.com/LJ-Speech-Dataset/).

the English speech dataset

[3] Ryosuke Sonobe, Shinnosuke Takamichi and Hiroshi Saruwatari, "[JSUT corpus: free large-scale Japanese speech corpus for end-to-end speech synthesis](https://arxiv.org/abs/1711.00354)," arXiv preprint, 1711.00354, 2017.

the Japanese speech dataset

[4] http://r9y9.github.io/blog/2017/11/23/dctts/

the blog post about the paper (Japanese)

[5] http://joisino.hatenablog.com/entry/2017/12/24/000000

the blog post related to this repository (Japanese)

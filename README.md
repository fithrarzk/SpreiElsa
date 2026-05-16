# SpreiElsa

Repository ini berisi implementasi Tugas Besar 2 IF3270 Pembelajaran Mesin. Fokus utama proyek adalah implementasi dan pengujian model deep learning untuk dua task:

- CNN untuk klasifikasi citra pada dataset Intel Image Classification.
- RNN/LSTM untuk image captioning pada dataset Flickr8k.

Implementasi dilakukan dengan dua pendekatan. Keras digunakan untuk training dan eksperimen utama, sedangkan implementasi from scratch berbasis NumPy digunakan untuk memahami dan memvalidasi proses forward propagation, batch inference, dan sebagian backward propagation.

## Struktur Repository

```text
SpreiElsa/
  doc/                  # laporan akhir
  src/
    cnn/                # modul CNN Keras dan CNN from scratch
    notebook/           # notebook eksperimen dan analisis
    rnn-lstm/           # modul RNN/LSTM Keras dan from scratch
    utils/              # image utility, text utility, feature extraction
    outputs/            # hasil run notebook
  README.md
  requirements-dev.txt
```

Dataset full tidak disimpan di repository. Dataset dapat diletakkan di luar repository untuk eksekusi lokal, atau digunakan melalui Kaggle Dataset ketika menjalankan notebook di Kaggle.

## Setup

Gunakan Python 3.10 atau lebih baru. Dari root repository `SpreiElsa/`, install dependency dengan:

```bash
pip install -r requirements-dev.txt
```

Dependency utama yang digunakan adalah TensorFlow/Keras, NumPy, Pillow, Matplotlib, scikit-learn, dan NLTK.

## Dataset

### Intel Image Classification

Pipeline CNN menerima dataset dalam format:

```text
train/<class>/*.jpg
val/<class>/*.jpg
test/<class>/*.jpg
```

Jika menggunakan dataset Intel dari Kaggle, format aslinya biasanya:

```text
seg_train/seg_train/<class>/*.jpg
seg_test/seg_test/<class>/*.jpg
```

Notebook CNN akan menyiapkan ulang dataset tersebut menjadi format `train/val/test` di folder working directory.


### Flickr8k

Pipeline RNN/LSTM menggunakan dataset Flickr8k yang berisi gambar, file caption, dan split train/validation/test. Pada Kaggle, notebook akan mencoba mendeteksi lokasi dataset secara otomatis. Jika diperlukan, path dataset dapat diatur melalui environment variable seperti `FLICKR8K_ROOT`, `FLICKR8K_IMAGES`, dan `FLICKR8K_CAPTIONS`.

## Cara Menjalankan Program

### CNN

Notebook utama CNN:

```bash
jupyter notebook src/notebook/cnn.ipynb
```

Notebook demo CNN untuk sanity check cepat:

```bash
jupyter notebook src/notebook/cnn_demo.ipynb
```


Output utama CNN disimpan dalam folder `outputs/cnn` atau `src/outputs/cnn`, tergantung lokasi eksekusi. Artefak yang dihasilkan mencakup `summary.csv`, `metrics.json`, `history.json`, `loss.png`, `scratch_compare.json`, dan `shared_vs_non_shared.csv`.

### RNN/LSTM

Notebook utama RNN/LSTM:

```bash
jupyter notebook src/notebook/rnn_lstm.ipynb
```

Notebook demo RNN/LSTM untuk sanity check cepat:

```bash
jupyter notebook src/notebook/rnn_lstm_demo.ipynb
```

Pipeline RNN/LSTM mencakup preprocessing caption, ekstraksi fitur gambar menggunakan CNN encoder pretrained, training decoder RNN dan LSTM, evaluasi BLEU-4 dan METEOR, serta perbandingan Keras dengan implementasi from scratch.

## Menjalankan di Kaggle

Untuk menjalankan notebook dari Kaggle, copy folder repository ke `/kaggle/working`, lalu eksekusi notebook menggunakan `nbconvert`.

Contoh menjalankan CNN full:

```bash
RUN_MODE=full jupyter nbconvert \
  --to notebook \
  --execute /kaggle/working/SpreiElsa/src/notebook/cnn.ipynb \
  --output /kaggle/working/executed_cnn.ipynb \
  --ExecutePreprocessor.timeout=-1 \
  --ExecutePreprocessor.kernel_name=python3
```

Contoh menjalankan CNN demo:

```bash
RUN_MODE=demo jupyter nbconvert \
  --to notebook \
  --execute /kaggle/working/SpreiElsa/src/notebook/cnn_demo.ipynb \
  --output /kaggle/working/executed_cnn_demo.ipynb \
  --ExecutePreprocessor.timeout=-1 \
  --ExecutePreprocessor.kernel_name=python3
```

Untuk RNN/LSTM, gunakan notebook `rnn_lstm.ipynb` atau `rnn_lstm_demo.ipynb` dengan pola yang sama.

## Ringkasan Modul

- `src/utils/image_utils.py`: image loader, batch loader, normalisasi, scanner dataset, dan helper save/load feature.
- `src/utils/text_utils.py`: preprocessing caption, tokenisasi, vocabulary, encode/decode caption.
- `src/cnn/`: implementasi CNN Keras, CNN from scratch, training, evaluasi, plotting, dan comparison helper.
- `src/rnn-lstm/`: implementasi RNN/LSTM Keras, RNN/LSTM from scratch, caption model, preprocessing Flickr8k, training decoder, dan evaluasi captioning.
- `src/notebook/`: notebook eksperimen utama, notebook demo, dan notebook analisis.

## Pembagian Tugas

| Nama Lengkap | NIM | Tugas |
| --- | --- | --- |
| Indah Novita Tangdililing | 13523047 | Membuat caption preprocessing, feature extraction Flickr8k, membuat RNN from scratch, membuat LSTM from scratch, membuat training decoder RNN dan LSTM, melakukan evaluasi captioning dengan BLEU-4 dan METEOR, membuat qualitative analysis hasil caption. |
| Muhammad Fithra Rizki | 13523049 | Menyusun infrastruktur project, melakukan evaluasi framework, membuat plotting dan visualisasi hasil eksperimen, mengintegrasi pipeline CNN dan RNN/LSTM, membuat batch inference pipeline, mendokumentasi README dan laporan, implementasi bonus visualisasi fitur CNN dan Grad-CAM, implementasi bonus init-inject, implementasi bonus beam search decoder, implementasi bonus batch inference, implementasi bonus backward propagation from scratch. |
| Muhammad Timur Kanigara | 13523055 | Membuat utility image processing, membuat CNN from scratch, melakukan CNN Keras training, melakukan eksperimen hyperparameter CNN, membuat evaluasi CNN, melakukan perbandingan shared vs non-shared parameter. |

## Catatan

File model dan dataset berukuran besar tidak perlu dimasukkan ke repository. Artefak hasil eksperimen dapat disimpan di folder `outputs/` atau diunduh dari Kaggle sesuai kebutuhan laporan.

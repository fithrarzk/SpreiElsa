# SpreiElsa

Implementasi IF3270 Tubes 2 untuk eksperimen neural network, dengan fokus saat ini pada CNN untuk Intel Image Classification.

## Struktur

```text
SpreiElsa/
  doc/
  src/
    cnn/                # modul CNN from scratch dan pipeline Keras
    dataset/            # script dataset + subset kecil lokal train/val/test
    notebook/
    utils/              # image loader, preprocessing, helper umum
```

Dataset full berada di luar repo ini:

```text
../dataset/
```

Jangan pindahkan dataset full ke dalam `SpreiElsa/`.

## Workflow Lokal

Development lokal ditujukan untuk sanity check kecil di CPU:

- cek loader gambar
- cek bentuk input/output model
- debug training mini
- cek evaluator macro F1-score
- cek save/load weights

Generate subset kecil dari root `SpreiElsa/`:

```powershell
python src/dataset/create_local_subset.py --source ..\dataset --output src\dataset --train-per-class 20 --val-per-class 5 --test-per-class 5
```

Subset lokal akan dibuat dalam format:

```text
src/dataset/
  train/<class>/*.jpg
  val/<class>/*.jpg
  test/<class>/*.jpg
```

## Setup

```powershell
pip install -r requirements-dev.txt
```

## Sanity Check Loader

```powershell
python -m src.utils.sanity_image_utils --split src/dataset/train --image-size 64
```

Expected shape untuk subset default:

```text
(120, 64, 64, 3) (120,) ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']
```

## CNN Anggota 1

Implementasi CNN utama ada di `src/cnn/`.

- `src/utils/image_utils.py`: public image loader, batch loader, scanner class-folder dataset, save/load `.npy`, dan helper feature extraction dengan frozen Keras encoder.
- `src/cnn/cnn_scratch/`: forward propagation NumPy-only untuk Conv2D shared, LocallyConnected2D non-shared, pooling, global pooling, flatten, dense, ReLU, softmax.
- `src/cnn/keras_models.py`: builder Keras untuk shared Conv2D dan non-shared LocallyConnected2D.
- `src/cnn/train_cnn.py`: CLI training satu eksperimen atau 16 eksperimen.
- `src/cnn/evaluate_cnn.py`: evaluator macro F1-score, confusion matrix, dan predictions CSV.
- `src/cnn/plotting.py`: plot loss dan summary CSV eksperimen.
- `src/cnn/scratch_compare.py`: perbandingan prediksi Keras vs forward scratch.
- `src/cnn/compare_shared_non_shared.py`: CSV/JSON ringkas untuk analisis shared vs non-shared.
- `src/cnn/kaggle_run_cnn.py`: runner full untuk Kaggle GPU.

Sanity check scratch:

```powershell
python -m src.cnn.tests_sanity
```

Mini training lokal satu eksperimen:

```powershell
python -m src.cnn.train_cnn --data-root src/dataset --output-dir outputs/cnn --experiment-id d1_f16_k3_max --epochs 1 --batch-size 16
```

Evaluate model mini:

```powershell
python -m src.cnn.evaluate_cnn --model-path outputs/cnn/shared_d1_f16_k3_max/model.keras --data-root src/dataset --output-dir outputs/cnn/shared_d1_f16_k3_max/eval
```

Compare Keras vs scratch:

```powershell
python -m src.cnn.scratch_compare --model-path outputs/cnn/shared_d1_f16_k3_max/model.keras --split-root src/dataset/test --output-path outputs/cnn/shared_d1_f16_k3_max/scratch_compare.json --max-samples 30
```

Format artefak untuk integrasi:

- `history.json`: loss dan validation loss per epoch.
- `metrics.json`: `macro_f1`, `test_accuracy`, `param_count`, dan class names.
- `predictions.csv`: label benar, label prediksi, dan probabilitas per kelas.
- `.npy` feature vector dari `image_utils.save_features/load_features`: array NumPy dengan baris sejajar urutan path input.

## Catatan Kaggle

Training besar nantinya dijalankan di Kaggle GPU dengan dataset full. Script training sebaiknya menerima path dataset dari argumen CLI atau environment variable, bukan hardcode path lokal.

Contoh runner Kaggle:

```bash
python -m src.cnn.kaggle_run_cnn --data-root /kaggle/input/intel-image-classification --output-dir /kaggle/working/outputs/cnn --epochs 10 --batch-size 32
```

`--data-root` boleh berisi format siap pakai `train/val/test`, atau format Intel Kaggle asli:

```text
seg_train/seg_train/<class>/*.jpg
seg_test/seg_test/<class>/*.jpg
```

Kalau formatnya masih Intel asli, runner akan menyiapkan copy `train/val/test` otomatis di `/kaggle/working/intel_prepared`.

# CNN Module Plan

Target CNN project:

- `src/utils/image_loader.py`: loader kecil untuk subset lokal, resize, normalisasi, dan batch/debug cepat.
- `src/dataset/create_local_subset.py`: generator subset lokal dari dataset full di luar repo ke `src/dataset/train`, `src/dataset/val`, dan `src/dataset/test`.
- `src/cnn/`: tempat implementasi CNN modular from scratch seperti `Conv2D`, `LocallyConnected2D`, pooling, flatten, activation, dan softmax.
- `src/cnn/keras_pipeline.py`: pipeline Keras portable untuk training besar di Kaggle GPU.
- `src/cnn/evaluator.py`: evaluasi model, terutama macro F1-score.
- `src/cnn/kaggle_run_cnn.py`: runner full 16 eksperimen untuk Kaggle GPU.

Prinsip path:

- Jangan hardcode path absolut lokal.
- Default lokal boleh mengarah ke `src/dataset/` di root project.
- Pipeline Kaggle sebaiknya menerima path dari argumen CLI atau environment variable.

Prinsip development:

- Local CPU hanya untuk sanity check kecil.
- Training final memakai dataset full di Kaggle.
- Simpan eksperimen berat dan output model besar di folder yang di-ignore git.

Command penting:

```powershell
python -m src.cnn.tests_sanity
python -m src.cnn.train_cnn --data-root src/dataset --output-dir outputs/cnn --experiment-id d1_f16_k3_max --epochs 1 --batch-size 16
python -m src.cnn.plotting --output-root outputs/cnn --summary-path outputs/cnn/summary.csv
python -m src.cnn.scratch_compare --model-path outputs/cnn/shared_d1_f16_k3_max/model.keras --split-root src/dataset/test --output-path outputs/cnn/shared_d1_f16_k3_max/scratch_compare.json
```

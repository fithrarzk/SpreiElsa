# Local Dataset Subset

Folder ini menyimpan subset kecil Intel Image Classification untuk sanity check lokal.

Dataset full tetap berada di `../../dataset/` relatif dari root `SpreiElsa/`, atau `../dataset/` jika command dijalankan dari root project.

Generate ulang subset dari root `SpreiElsa/`:

```powershell
python src/dataset/create_local_subset.py --source ..\dataset --output src\dataset --train-per-class 20 --val-per-class 5 --test-per-class 5
```

Output:

```text
src/dataset/
  train/<class>/*.jpg
  val/<class>/*.jpg
  test/<class>/*.jpg
```

Folder `train`, `val`, dan `test` di-ignore git karena berisi file gambar lokal.

Untuk Kaggle/full dataset dengan layout Intel asli:

```powershell
python src/dataset/prepare_intel_dataset.py --source ..\dataset --output outputs\intel_prepared --max-train-per-class 20 --max-val-per-class 5 --max-test-per-class 5
```

Di Kaggle, tidak perlu batas `--max-*`:

```bash
python -m src.dataset.prepare_intel_dataset --source /kaggle/input/intel-image-classification --output /kaggle/working/intel_prepared
```

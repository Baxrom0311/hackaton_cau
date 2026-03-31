# AI Healthcare Hackathon 2026

Bu repo hackathon uchun tayyorlangan classification + segmentation pipeline, Streamlit web UI, submission build va checker skriptlarini bir joyga yig'adi.

## 1. Nima bor

- `app.py`: Streamlit web interfeys
- `classify.py`: classification inference, Excel chiqadi
- `segment.py`: segmentation inference, PNG maskalar chiqadi
- `prepare_submission.py`: guide-compliant official submission papka tayyorlaydi
- `checker_files/`: local checker va evaluator skriptlari
- `kaggle_*_v5.py`, `colab_*_v5.py`: training scriptlar
- `models/classification/best_model.pth`: classification checkpoint
- `models/segmentation/best_model.pth`: segmentation checkpoint
- `deploy/`: nginx + systemd deploy fayllari

## 2. Kutiladigan papka tuzilmasi

Repo ichida ishlatiladigan asosiy dataset pathlar:

```text
dataset/
  classification/
    train/
      0/ ... 11/
    test/
  Segmentation/
    training/
      masks/
    validation/
      masks/
    testing/
      images/
```

Eslatma:

- Hozirgi local repo ichida `dataset/Segmentation/training/images` va `dataset/Segmentation/validation/images` yo'q.
- Shu sabab segmentation training/evaluation skriptlarining ayrim qismlari faqat to'liq dataset bo'lsa ishlaydi.

## 3. O'rnatish

### Python venv

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

### CLI va training uchun

```bash
pip install -r requirements.txt
```

### Web UI uchun

```bash
pip install -r requirements_ui.txt
```

## 4. Tez start

### Web UI

```bash
streamlit run app.py
```

UI default bo'yicha quyidagi model pathlarni ishlatadi:

- `models/classification/best_model.pth`
- `models/segmentation/best_model.pth`

### Classification inference

```bash
python classify.py \
  --test_dir dataset/classification/test \
  --model_path models/classification/best_model.pth \
  --team OxDEAD
```

Natija:

- `OxDEAD test_ground_truth.xlsx`

### Segmentation inference

```bash
python segment.py \
  --test_dir dataset/Segmentation/testing/images \
  --model_path models/segmentation/best_model.pth \
  --team OxDEAD
```

Natija:

- `OxDEAD/` nomli papka
- ichida `200` ta `.png` maska

## 5. Official submission build

Avval classification Excel va segmentation maskalarni generatsiya qiling, keyin:

```bash
python prepare_submission.py \
  --team OxDEAD \
  --cls_model models/classification/best_model.pth \
  --seg_model models/segmentation/best_model.pth \
  --excel_path "OxDEAD test_ground_truth.xlsx" \
  --masks_dir OxDEAD
```

Natija:

- `OxDEAD_OfficialBuild/OxDEAD/`

Ichida:

- `OxDEAD test_ground_truth.xlsx`
- `OxDEAD/` maska papkasi
- `models/classification/`
- `models/segmentation/`

Muhim:

- segmentation checkpoint ichida `best_threshold` bo'lishi shart
- classification Excel `1276` qator bo'lishi shart
- segmentation maskalar soni `200` bo'lishi shart
- maskalar original image size bilan bir xil va binary bo'lishi shart

## 6. Checker va evaluatorlar

### Submission checker

```bash
python checker_files/check_submission.py \
  --team OxDEAD \
  --submission_dir OxDEAD_OfficialBuild
```

### Classification submission-time metric

```bash
python checker_files/evaluate_classification_submission_metric.py \
  --model_path models/classification/best_model.pth \
  --train_dir dataset/classification/train \
  --split val
```

Agar hard manifest ishlatmoqchi bo'lsangiz:

```bash
python checker_files/evaluate_classification_submission_metric.py \
  --model_path models/classification/best_model.pth \
  --train_dir dataset/classification/train \
  --manifest_path classification_hard_split_manifest.json \
  --split val
```

### Segmentation submission-time metric

```bash
python checker_files/evaluate_segmentation_submission_metric.py \
  --model_path models/segmentation/best_model.pth \
  --img_dir dataset/Segmentation/validation/images \
  --mask_dir dataset/Segmentation/validation/masks
```

Thresholdni checkpointga qayta yozish:

```bash
python checker_files/evaluate_segmentation_submission_metric.py \
  --model_path models/segmentation/best_model.pth \
  --img_dir dataset/Segmentation/validation/images \
  --mask_dir dataset/Segmentation/validation/masks \
  --write-threshold
```

### Dataset audit

```bash
python checker_files/analyze_classification_train_dataset.py \
  --train_dir dataset/classification/train \
  --write_hard_manifest classification_hard_split_manifest.json
```

## 7. Training scriptlar

Classification:

- `kaggle_classification_v5.py`
- `colab_classification_ultra_v5.py`

Segmentation:

- `kaggle_segmentation_v5.py`
- `colab_segmentation_ultra_v5.py`

Muhim note:

- classification trainerlar hard manifestni topsa ishlatadi
- segmentation trainer real validation splitga tayanadi
- segmentation inference `best_threshold` bo'lmasa default `0.50` ishlatadi

## 8. Server deploy

Deploy uchun helper fayllar:

- `deploy/nafisa-streamlit.service`
- `deploy/nafisa.boos.uz.nginx`

Minimal deploy flow:

```bash
rsync -av \
  app.py \
  requirements_ui.txt \
  models/ \
  deploy/ \
  ubuntu@SERVER_IP:~/hackaton-ui/
```

Server ichida:

```bash
cd ~/hackaton-ui
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements_ui.txt
sudo cp nafisa-streamlit.service /etc/systemd/system/nafisa-streamlit.service
sudo cp nafisa.boos.uz.nginx /etc/nginx/sites-available/nafisa.boos.uz
sudo ln -sf /etc/nginx/sites-available/nafisa.boos.uz /etc/nginx/sites-enabled/nafisa.boos.uz
sudo nginx -t
sudo systemctl daemon-reload
sudo systemctl enable --now nafisa-streamlit
sudo systemctl restart nginx
```

Server buyruqlari:

```bash
sudo systemctl status nafisa-streamlit
sudo journalctl -u nafisa-streamlit -f
sudo systemctl restart nafisa-streamlit
sudo nginx -t && sudo systemctl reload nginx
```

Muhim tavsiya:

- shared web serverda AI inference yuritish xavfli
- ayniqsa `streamlit + torch + existing apps` kombinatsiyasi RAM/CPU bosimini oshiradi
- production/demo uchun AI'ni alohida serverga yoki external inference service'ga ajratish yaxshiroq

## 9. Tailscale bilan test

### Private test

Testerlar sizning tailnet ichida bo'lsa:

```bash
tailscale serve 8501
```

### Public test

Testerlar Tailscale ishlatmasa:

```bash
tailscale funnel 8501
tailscale funnel status
```

Eslatma:

- bu `*.ts.net` link beradi
- custom domain emas
- Funnel access beradi, lekin compute muammosini hal qilmaydi

## 10. Troubleshooting

### `best_threshold checkpointda yo'q`

Segmentation checkpointni evaluator bilan qayta threshold yozing:

```bash
python checker_files/evaluate_segmentation_submission_metric.py \
  --model_path models/segmentation/best_model.pth \
  --img_dir dataset/Segmentation/validation/images \
  --mask_dir dataset/Segmentation/validation/masks \
  --write-threshold
```

### Server crash yoki hang

Agar server javob bermay qolsa:

1. oldin `nafisa-streamlit`ni to'xtating
2. kerak bo'lsa `ghouse` backendni ham to'xtating
3. `free -h`, `uptime`, `ss -tulpn` bilan tekshiring
4. agar SSH kirmasa, cloud console orqali reboot qiling

Emergency commandlar:

```bash
sudo systemctl stop nafisa-streamlit
sudo systemctl disable nafisa-streamlit
sudo pkill -f 'streamlit run /home/ubuntu/hackaton-ui/app.py' || true

sudo systemctl stop smartgreenhouse-backend || \
sudo systemctl stop greenhouse || \
sudo pkill -f 'gunicorn.*127.0.0.1:8001' || true
```

## 11. Tavsiya etiladigan ish tartibi

1. `requirements*.txt` o'rnating
2. `classify.py` va `segment.py` bilan natija chiqaring
3. `prepare_submission.py` bilan official papka build qiling
4. `checker_files/check_submission.py` bilan verify qiling
5. kerak bo'lsa `app.py` orqali demo qiling
6. production yoki testerlar uchun shared serverdan ko'ra alohida node ishlating

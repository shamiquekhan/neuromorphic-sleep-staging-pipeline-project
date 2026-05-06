@echo off
REM Training script for seq_len=60 teacher
cd /d C:\Project\CNN-ECG

python -u -m sleep_staging.cli train-teacher ^
  --mode real ^
  --manifest data/manifests/sleep_edf_full.csv ^
  --epochs 80 ^
  --batch-size 16 ^
  --teacher-ckpt artifacts/teacher_seq60_v1.pt ^
  --cache-dir data/cache ^
  --patience 15

echo.
echo Training completed!
echo Check artifacts/teacher_seq60_v1.pt for the saved model
pause

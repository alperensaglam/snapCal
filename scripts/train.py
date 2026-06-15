import sys
from pathlib import Path

# Make the repo root importable when run as `python scripts/train.py`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from ultralytics import YOLO

from lokma.config import AppConfig

def train_model():
    # Donanım hızlandırma kontrolü
    if torch.cuda.is_available():
        device = 'cuda'
        print("Cuda destekli GPU bulundu. Eğitim GPU üzerinde gerçekleştirilecek.")
    elif torch.backends.mps.is_available():
        device = 'mps'
        print("MPS destekli cihaz bulundu. Eğitim MPS üzerinde gerçekleştirilecek.")
    else:
        device = 'cpu'
        print("GPU veya MPS destekli cihaz bulunamadı. Eğitim CPU üzerinde gerçekleştirilecek.")

    # Model yükleme
    config = AppConfig()
    model = YOLO(str(config.pretrained_seg_model))

    # Eğitimi başlat
    model.train(data=str(config.yolo_dataset_dir / "data.yaml"),
            epochs=10, # Segmentasyon daha detaylı olduğu için epoch sayısını artırmak iyidir
            imgsz=640,
            device=device,
            name="snapCal_v1_seg",
            project="runs/train",
            batch=16,
            workers=4,
            cache=True,
            amp=True, # Karışık hassasiyetli eğitim ile hızlandırma
            simplify=True
            )
    
    print("Eğitim tamamlandı.")

if __name__ == "__main__":
    train_model()
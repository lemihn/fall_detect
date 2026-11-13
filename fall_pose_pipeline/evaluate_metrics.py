import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

DATA_DIR = r"fall_pose_pipeline"
MODEL_PATH = os.path.join(DATA_DIR, "fall_lstm_model.pth")
CLASS_NAMES = ["Normal", "Fall", "Warning"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FallDetectionLSTM(nn.Module):
    def __init__(self, input_size=36, hidden_size=64, num_layers=2, num_classes=3, dropout=0.3):
        super(FallDetectionLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.classifier(lstm_out[:, -1, :])

def main():
    print("="*60)
    print(">>> KIEM TRA DANH GIA MO HINH (VAL EVALUATION)")
    print("="*60)
    
    X_val = np.load(os.path.join(DATA_DIR, "X_val.npy"))
    y_val = np.load(os.path.join(DATA_DIR, "y_val.npy"))
    
    ckpt = torch.load(MODEL_PATH, map_location=DEVICE)
    model = FallDetectionLSTM(
        input_size=ckpt["input_size"],
        hidden_size=ckpt["hidden_size"],
        num_layers=ckpt["num_layers"],
        num_classes=ckpt["num_classes"]
    ).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32), 
                                          torch.tensor(y_val, dtype=torch.long)), 
                            batch_size=32, shuffle=False)
                            
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for bx, by in val_loader:
            bx = bx.to(DEVICE)
            logits = model(bx)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(by.numpy())
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # 1. Confusion Matrix
    cm = np.zeros((3, 3), dtype=int)
    for t, p in zip(all_targets, all_preds):
        cm[t, p] += 1
        
    print("\n1. MA TRAN NHAM LAN (CONFUSION MATRIX):")
    print(f"{'Target \\ Pred':<15} {'Normal':<10} {'Fall':<10} {'Warning':<10}")
    for i, name in enumerate(CLASS_NAMES):
        print(f"{name:<15} {cm[i, 0]:<10} {cm[i, 1]:<10} {cm[i, 2]:<10}")
        
    # 2. Precision, Recall, F1
    print("\n2. CHI SO DANH GIA TUNG LOP (PER-CLASS METRICS):")
    print(f"{'Class':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Total':<10}")
    
    f1_list = []
    for i, name in enumerate(CLASS_NAMES):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        support = cm[i, :].sum()
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_list.append(f1)
        
        print(f"{name:<12} {prec*100:<11.2f}% {rec*100:<11.2f}% {f1*100:<11.2f}% {support:<10}")
        
    acc = (all_preds == all_targets).mean() * 100
    macro_f1 = np.mean(f1_list) * 100
    print("-" * 58)
    print(f"Overall Accuracy: {acc:.2f}% | Macro F1-Score: {macro_f1:.2f}%")
    
    # 3. Test toc do suy luan (Latency)
    dummy_input = torch.tensor(X_val[:1], dtype=torch.float32).to(DEVICE)
    start = time.time()
    for _ in range(500):
        with torch.no_grad():
            _ = model(dummy_input)
    elap = (time.time() - start) / 500
    print(f"\n3. TOC DO SUY LUAN TRUONG HOP REAL-TIME:")
    print(f"Latency: {elap*1000:.2f} ms tren 1 frame ({1/elap:.1f} FPS)")
    print("="*60)

if __name__ == "__main__":
    main()

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ==========================================
# CAU HINH (DYNAMIC PATHS)
# ==========================================
DATA_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE_PATH = os.path.join(DATA_DIR, "fall_lstm_model.pth")

BATCH_SIZE = 32
HIDDEN_SIZE = 64
NUM_LAYERS = 2
EPOCHS = 40
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CLASS_NAMES = ["Normal", "Fall", "Warning"]


# ==========================================
# PYTORCH DATASET
# ==========================================
class PoseSequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ==========================================
# KIEN TRUC MANG LSTM
# ==========================================
class FallDetectionLSTM(nn.Module):
    def __init__(self, input_size=36, hidden_size=64, num_layers=2, num_classes=3, dropout=0.3):
        super(FallDetectionLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=False
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        # x: [Batch, Seq_Len, Features] -> [Batch, Hidden] -> [Batch, Classes]
        lstm_out, _ = self.lstm(x)
        last_out = lstm_out[:, -1, :]
        logits = self.classifier(last_out)
        return logits


# ==========================================
# DANH GIA DA CHIEU (CONFUSION MATRIX, PRECISION, RECALL, F1)
# ==========================================
def compute_metrics(y_true, y_pred, num_classes=3):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        if 0 <= t < num_classes and 0 <= p < num_classes:
            cm[t, p] += 1

    precisions, recalls, f1s = [], [], []
    for c in range(num_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)

    macro_f1 = float(np.mean(f1s))
    return cm, precisions, recalls, f1s, macro_f1


# ==========================================
# HAM HUAN LUYEN
# ==========================================
def main():
    print(f">>> Thiet bi huan luyen: {DEVICE}")

    x_train_path = os.path.join(DATA_DIR, "X_train.npy")
    y_train_path = os.path.join(DATA_DIR, "y_train.npy")
    x_val_path = os.path.join(DATA_DIR, "X_val.npy")
    y_val_path = os.path.join(DATA_DIR, "y_val.npy")

    if not (os.path.exists(x_train_path) and os.path.exists(y_train_path)):
        print(f"Loi: Chua tim thay du lieu da trich xuat tai '{DATA_DIR}'.")
        print("Vui long chay script '1_extract_dataset.py' truoc!")
        return

    X_train = np.load(x_train_path)
    y_train = np.load(y_train_path)
    X_val = np.load(x_val_path)
    y_val = np.load(y_val_path)

    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"X_val: {X_val.shape}, y_val: {y_val.shape}")

    # Tinh toan Class Weights de can bang giua Normal, Fall, Warning
    classes, counts = np.unique(y_train, return_counts=True)
    total_samples = len(y_train)
    num_classes = len(CLASS_NAMES)

    weights = np.ones(num_classes, dtype=np.float32)
    for cls, count in zip(classes, counts):
        weights[int(cls)] = total_samples / (num_classes * count)
    class_weights_tensor = torch.tensor(weights, dtype=torch.float32, device=DEVICE)
    print(f"Class Weights de can bang loss: {weights}")

    train_loader = DataLoader(PoseSequenceDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(PoseSequenceDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

    input_size = X_train.shape[2]
    model = FallDetectionLSTM(
        input_size=input_size,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        num_classes=num_classes
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)

    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_macro_f1 = 0.0
    best_metrics_report = None

    print("\n>>> Bat dau qua trinh huan luyen:")
    for epoch in range(1, EPOCHS + 1):
        # TRAIN
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * len(batch_y)
            _, preds = torch.max(outputs, 1)
            train_correct += (preds == batch_y).sum().item()
            train_total += len(batch_y)

        train_loss /= max(train_total, 1)
        train_acc = train_correct / max(train_total, 1)

        # VAL
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(DEVICE)
                batch_y = batch_y.to(DEVICE)

                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                val_loss += loss.item() * len(batch_y)
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == batch_y).sum().item()
                val_total += len(batch_y)

                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(batch_y.cpu().numpy())

        val_loss /= max(val_total, 1)
        val_acc = val_correct / max(val_total, 1)
        cm, precisions, recalls, f1s, macro_f1 = compute_metrics(all_targets, all_preds, num_classes)
        scheduler.step(val_loss)

        # Luu model tot nhat
        saved_mark = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_macro_f1 = macro_f1
            best_metrics_report = (cm, precisions, recalls, f1s)
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "input_size": input_size,
                "hidden_size": HIDDEN_SIZE,
                "num_layers": NUM_LAYERS,
                "num_classes": num_classes,
                "classes": CLASS_NAMES,
                "val_acc": val_acc,
                "val_loss": val_loss,
                "val_macro_f1": macro_f1
            }, MODEL_SAVE_PATH)
            saved_mark = " [*BEST SAVED*]"

        if epoch % 5 == 0 or epoch == 1 or saved_mark:
            print(
                f"Epoch [{epoch:02d}/{EPOCHS:02d}] "
                f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:.1f}% || "
                f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.1f}% | Macro F1: {macro_f1*100:.1f}%{saved_mark}"
            )

    print("\n" + "=" * 50)
    print("HOAN THANH HUAN LUYEN!")
    print(f"Model tot nhat da luu tai: {MODEL_SAVE_PATH}")
    print(f"Best Val Loss: {best_val_loss:.4f} | Best Val Acc: {best_val_acc*100:.2f}% | Best Macro F1: {best_macro_f1*100:.2f}%")

    if best_metrics_report is not None:
        cm, precisions, recalls, f1s = best_metrics_report
        print("\n--- BAO CAO CHI TIET TUNG LOP (BEST MODEL) ---")
        for c, name in enumerate(CLASS_NAMES):
            print(f"  + {name:8s}: Precision={precisions[c]*100:5.1f}% | Recall={recalls[c]*100:5.1f}% | F1-Score={f1s[c]*100:5.1f}%")
        print("\nConfusion Matrix (Rows=True, Cols=Pred [Normal, Fall, Warning]):")
        print(cm)


if __name__ == "__main__":
    main()

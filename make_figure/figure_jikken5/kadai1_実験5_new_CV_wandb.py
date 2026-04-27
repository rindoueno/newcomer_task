import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import KFold
import numpy as np
import random
import wandb

# --- 1. シード固定関数 ---
def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# --- 3. モデル定義（Batch Normalizationを追加） ---
class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_stack = nn.Sequential(
            # 第1ブロック: Conv -> BatchNorm -> ReLU -> Pool
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), # チャンネルごとの統計量を正規化
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 第2ブロック
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.flatten = nn.Flatten()
        self.fc_stack = nn.Sequential(
            nn.Linear(64 * 7 * 7, 512),
            nn.BatchNorm1d(512), # 全結合層用BN
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.conv_stack(x)
        x = self.flatten(x)
        logits = self.fc_stack(x)
        return logits

# --- 学習・評価関数（変更なし） ---
def train_one_epoch(dataloader, model, loss_fn, optimizer, device):
    model.train(); total_loss = 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        pred = model(X); loss = loss_fn(pred, y)
        loss.backward(); optimizer.step(); optimizer.zero_grad()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def evaluate(dataloader, model, loss_fn, device):
    model.eval(); loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X); loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    return loss / len(dataloader), correct / len(dataloader.dataset)

# --- 4. メイン処理 ---
def main():
    seed_value = 42
    fix_seed(seed_value)
    
    project_name = "fashion-mnist-no5-experiment"
    group_name = "CNN-BN-Augm-ErrorAnalysis"
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
    ])
    val_transform = transforms.Compose([transforms.ToTensor()])

    train_dataset_full = datasets.FashionMNIST(root="data", train=True, download=True, transform=train_transform)
    val_dataset_full = datasets.FashionMNIST(root="data", train=True, download=True, transform=val_transform)
    test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=val_transform)
    classes = train_dataset_full.classes
    
    k_folds = 5
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed_value)
    fold_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(train_dataset_full)))):
        print(f"\n>>> Fold {fold+1}/{k_folds}")
        
        wandb.init(
            project=project_name, group=group_name, name=f"fold-{fold+1}",
            config={"lr":0.000399, "batch_size": 64, "epochs": 100, "patience": 5, "architecture": "CNN+BN+Augm"}
        )

        train_loader = DataLoader(Subset(train_dataset_full, train_idx), batch_size=64, shuffle=True)
        val_loader = DataLoader(Subset(val_dataset_full, val_idx), batch_size=64, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

        model = ConvolutionalNeuralNetwork().to(device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.lr)

        best_val_loss = float('inf')
        counter = 0

        # --- 学習ループ ---
        for epoch in range(wandb.config.epochs):
            tr_loss = train_one_epoch(train_loader, model, loss_fn, optimizer, device)
            val_loss, val_acc = evaluate(val_loader, model, loss_fn, device)
            wandb.log({"epoch": epoch + 1, "train_loss": tr_loss, "val_loss": val_loss, "val_acc": val_acc})

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f"best_model_fold{fold+1}.pth")
                counter = 0
            else:
                counter += 1
                if counter >= wandb.config.patience: break

        # --- 最終評価 ＆ 誤分類調査 ---
        model.load_state_dict(torch.load(f"best_model_fold{fold+1}.pth"))
        test_loss, test_acc = evaluate(test_loader, model, loss_fn, device)
        wandb.run.summary["test_accuracy"] = test_acc

        all_preds, all_labels = [], []
        error_table = wandb.Table(columns=["Image", "Predicted", "Actual", "Confidence"])
        
        model.eval()
        error_count = 0
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                probs = torch.softmax(outputs, dim=1)
                preds = outputs.argmax(1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())

                # 誤分類画像を特定して wandb Table に追加 (最初の30件程度)
                if error_count < 30:
                    for i in range(len(y)):
                        if preds[i] != y[i] and error_count < 30:
                            img = X[i].cpu().squeeze().numpy()
                            error_table.add_data(
                                wandb.Image(img), 
                                classes[preds[i]], 
                                classes[y[i]], 
                                probs[i][preds[i]].item()
                            )
                            error_count += 1

        wandb.log({
            "confusion_matrix": wandb.plot.confusion_matrix(y_true=all_labels, preds=all_preds, class_names=classes),
            "error_examples": error_table # 誤分類リストをアップロード
        })
        
        fold_accuracies.append(test_acc)
        wandb.finish()

    print(f"\nFinal CV Accuracy: {np.mean(fold_accuracies)*100:.2f}% ± {np.std(fold_accuracies)*100:.2f}%")

if __name__ == "__main__":
    main()
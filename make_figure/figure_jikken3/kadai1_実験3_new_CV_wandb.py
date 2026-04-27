import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
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

# --- 2. データの準備 ---
transform = transforms.ToTensor()
full_train_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=transform)
test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=transform)
classes = full_train_data.classes

# --- 3. モデル・学習・評価関数の定義 ---
class ConvolutionalNeuralNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            # 畳み込み層の定義
            self.conv_stack = nn.Sequential(
                # 第1ブロック: 1ch入力 -> 32ch出力
                nn.Conv2d(1, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2), # 28x28 -> 14x14
                
                # 第2ブロック: 32ch入力 -> 64ch出力
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2), # 14x14 -> 7x7
            )
            # 全結合層（分類用）
            self.flatten = nn.Flatten()
            self.fc_stack = nn.Sequential(
                nn.Linear(64 * 7 * 7, 512),
                nn.ReLU(),
                nn.Linear(512, 10)
            )

        def forward(self, x):
            x = self.conv_stack(x)
            x = self.flatten(x)
            logits = self.fc_stack(x)
            return logits
        
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
    
    project_name = "fashion-mnist-kfold-modelcnn"
    group_name = "MLP-5-jikken3-modelcnn"
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    k_folds = 5
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=seed_value)
    
    fold_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(full_train_data)):
        print(f"\n>>> Fold {fold+1}/{k_folds}")
        
        wandb.init(
            project=project_name,
            group=group_name,
            name=f"fold-{fold+1}",
            config={
                "fold": fold + 1,
                "learning_rate":0.000919,
                "batch_size": 64,
                "epochs": 100,
                "patience": 5
            }
        )
        config = wandb.config

        train_loader = DataLoader(Subset(full_train_data, train_idx), batch_size=config.batch_size, shuffle=True)
        val_loader = DataLoader(Subset(full_train_data, val_idx), batch_size=config.batch_size, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=config.batch_size, shuffle=False)

        model = ConvolutionalNeuralNetwork().to(device)
        loss_fn = nn.CrossEntropyLoss(); optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

        best_val_loss = float('inf')
        counter = 0

        # 学習
        for epoch in range(config.epochs):
            tr_loss = train_one_epoch(train_loader, model, loss_fn, optimizer, device)
            val_loss, val_acc = evaluate(val_loader, model, loss_fn, device)
            wandb.log({"epoch": epoch + 1, "train_loss": tr_loss, "val_loss": val_loss, "val_accuracy": val_acc})

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), f"best_model_fold{fold+1}.pth")
                counter = 0
            else:
                counter += 1
                if counter >= config.patience: break

        # 【追加】最終評価と混同行列の作成
        model.load_state_dict(torch.load(f"best_model_fold{fold+1}.pth"))
        test_loss, test_acc = evaluate(test_loader, model, loss_fn, device)
        
        # テストセットでの予測を全回収
        all_preds, all_labels = [], []
        model.eval()
        with torch.no_grad():
            for X, y in test_loader:
                X, y = X.to(device), y.to(device)
                all_preds.extend(model(X).argmax(1).cpu().numpy())
                all_labels.extend(y.cpu().numpy())

        # wandbにテスト精度と混同行列を送信
        wandb.run.summary["test_accuracy"] = test_acc
        wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(
            probs=None, y_true=all_labels, preds=all_preds, class_names=classes
        )})
        
        fold_accuracies.append(test_acc)
        wandb.finish()

    print(f"\nFinal CV Accuracy: {np.mean(fold_accuracies)*100:.2f}% ± {np.std(fold_accuracies)*100:.2f}%")

if __name__ == "__main__":
    main()
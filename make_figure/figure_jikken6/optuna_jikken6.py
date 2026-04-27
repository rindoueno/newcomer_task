import torch
from torch import nn
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms, models
import optuna

# --- 1. EfficientNetモデル作成関数 ---
def create_efficientnet(device):
    # weights=DEFAULT で学習済み重みを使用
    model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
    # 最終層を10クラス用に書き換え
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 10)
    return model.to(device)

# --- 2. Optuna用の目的関数 ---
def objective(trial):
    # ハイパーパラメータの探索範囲（転移学習なので低めからスタート）
    lr = trial.suggest_float("lr", 1e-6, 1e-4, log=True)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # EfficientNet用の最小限のTransform（リサイズと3ch化は必須）
    transform = transforms.Compose([
        transforms.Resize(128),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # データの準備（高速化のため50k/10kに分割）
    full_ds = datasets.FashionMNIST(root="data", train=True, download=True, transform=transform)
    train_sub, val_sub = random_split(full_ds, [50000, 10000])
    
    # 試行を速めるため、バッチサイズは少し大きめの64に設定
    t_loader = DataLoader(train_sub, batch_size=64, shuffle=True)
    v_loader = DataLoader(val_sub, batch_size=64, shuffle=False)

    model = create_efficientnet(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    # 3エポックだけ回して学習率のポテンシャルを見る
    model.train()
    for epoch in range(3):
        for X, y in t_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X); loss = loss_fn(pred, y)
            loss.backward(); optimizer.step(); optimizer.zero_grad()

    # 検証精度で評価
    model.eval()
    correct = 0
    with torch.no_grad():
        for X, y in v_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    
    accuracy = correct / len(v_loader.dataset)
    return accuracy # 精度を最大化する

# --- 3. 実行部分 ---
if __name__ == "__main__":
    study = optuna.create_study(direction="maximize") # 精度最大化を目指す
    # 時間短縮のため試行回数は少なめに設定（例: 5〜10回）
    study.optimize(objective, n_trials=5)

    print(f"EfficientNetにおける最高の学習率: {study.best_params['lr']:.8f}")
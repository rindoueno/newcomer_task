import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from sklearn.model_selection import KFold
import wandb
import numpy as np
import random

#  シード固定関数
def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# モデル定義（転移学習
def create_model(device):
    model = models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 10)
    return model.to(device)

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

#  メイン処理
def main():
    seed_value = 42
    fix_seed(seed_value)
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    
    # Transform
    train_transform = transforms.Compose([
        transforms.Resize(128),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize(128),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_ds_full = datasets.FashionMNIST(root="data", train=True, download=True, transform=train_transform)
    val_ds_full = datasets.FashionMNIST(root="data", train=True, download=True, transform=val_transform)
    test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=val_transform)
    classes = train_ds_full.classes

    kf = KFold(n_splits=5, shuffle=True, random_state=seed_value)
    fold_accuracies = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(range(len(train_ds_full)))):
        print(f"\n>>> Fold {fold+1}/5")
        
        wandb.init(
            project="fashion-mnist-efficientnet",
            group="EfficientNet-V2-Transfer",
            name=f"fold-{fold+1}",
            config={
                "lr": 6.886128605289248e-06, 
                "batch_size": 32, 
                "epochs": 20,
                "patience": 5 
            }
        )

        train_loader = DataLoader(Subset(train_ds_full, train_idx), batch_size=wandb.config.batch_size, shuffle=True)
        val_loader = DataLoader(Subset(val_ds_full, val_idx), batch_size=wandb.config.batch_size, shuffle=False)
        test_loader = DataLoader(test_data, batch_size=wandb.config.batch_size, shuffle=False)

        model = create_model(device)
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=wandb.config.lr)

        best_val_loss = float('inf')
        counter = 0

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
                if counter >= wandb.config.patience: 
                    print(f"Early Stopping at epoch {epoch+1}")
                    break

        # 最終評価 ＆ 誤分類調査
        model.load_state_dict(torch.load(f"best_model_fold{fold+1}.pth"))
        test_loss, test_acc = evaluate(test_loader, model, loss_fn, device)
        wandb.run.summary["test_accuracy"] = test_acc

        all_preds, all_labels = [], []
        error_table = wandb.Table(columns=["Image", "Predicted", "Actual", "Confidence"])
        
        model.eval()
        error_count = 0
        with torch.no_grad():
            for X, y in test_loader:
                X_dev, y_dev = X.to(device), y.to(device)
                outputs = model(X_dev)
                probs = torch.softmax(outputs, dim=1)
                preds = outputs.argmax(1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y_dev.cpu().numpy())

                if error_count < 30:
                    for i in range(len(y_dev)):
                        if preds[i] != y_dev[i] and error_count < 30:
                            img = X[i][0].cpu().numpy() 
                            error_table.add_data(
                                wandb.Image(img), 
                                classes[preds[i]], 
                                classes[y_dev[i]], 
                                probs[i][preds[i]].item()
                            )
                            error_count += 1

        wandb.log({
            "confusion_matrix": wandb.plot.confusion_matrix(y_true=all_labels, preds=all_preds, class_names=classes),
            "error_examples": error_table 
        })
        
        fold_accuracies.append(test_acc)
        wandb.finish()

    print(f"\nFinal CV Accuracy: {np.mean(fold_accuracies)*100:.2f}% ± {np.std(fold_accuracies)*100:.2f}%")

if __name__ == "__main__":
    main()

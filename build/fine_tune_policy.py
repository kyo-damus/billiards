# fine_tune_policy.py
import torch
import torch.nn as nn
import torch.optim as optim
from train_policy import PolicyNetwork

def fine_tune():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. MCTSが集めた「成功体験」の読み込み
    try:
        experience = torch.load("mcts_experience.pt", weights_only=True)
        inputs = experience["states"].to(device)
        targets = experience["targets"].to(device)
        print(f"--- {len(inputs)}件の成功体験(MCTSの発見)をロードしました ---")
    except FileNotFoundError:
        print("エラー: mcts_experience.pt が見つかりません。先に自己対局を実行してください。")
        return

    # 2. 現在の脳みそ（Policy Network）を読み込む
    model = PolicyNetwork().to(device)
    model.load_state_dict(torch.load("policy_model.pth", weights_only=True))

    # 3. 再学習（ファインチューニング）の設定
    criterion = nn.MSELoss()
    # 既に学習済みの脳を壊さないよう、学習率(lr)は通常より小さく設定
    optimizer = optim.Adam(model.parameters(), lr=0.0005) 

    epochs = 100
    model.train()
    print("\n--- 自己更新（ファインチューニング）開始 ---")
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        predictions = model(inputs)
        loss = criterion(predictions, targets)
        
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    # 4. 賢くなったモデルを上書き保存
    torch.save(model.state_dict(), "policy_model.pth")
    print("\n自己更新完了！賢くなったモデルを 'policy_model.pth' に上書き保存しました。")

if __name__ == "__main__":
    fine_tune()
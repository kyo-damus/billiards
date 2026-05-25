import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 【1】PyTorchモデルの定義 (シンプルな多層パーセプトロン)
class PolicyNetwork(nn.Module):
    def __init__(self):
        super(PolicyNetwork, self).__init__()
        # 入力: 4次元 (手球x, 手球y, 的球x, 的球y)
        # 出力: 1次元 (予測される最適な角度)
        self.fc = nn.Sequential(
            nn.Linear(4, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.fc(x)

# 【2】トレーニング用データセットの自動生成
def generate_synthetic_data(num_samples=10000):
    print(f"--- {num_samples}件の教師データ（配置と正解角度）を生成中 ---")
    inputs = []
    targets = []
    pocket_x, pocket_y = 1.0, 1.0
    r = 0.05

    for _ in range(num_samples):
        # ランダムな配置を生成
        cue_x, cue_y = np.random.uniform(-1.0, 1.0, 2)
        obj_x, obj_y = np.random.uniform(-0.8, 0.8, 2)
        
        # ゴーストボールの計算（教師データの「正解ラベル」を作成）
        dx, dy = pocket_x - obj_x, pocket_y - obj_y
        dist = np.hypot(dx, dy)
        ghost_x = obj_x - (dx / dist) * (2.0 * r)
        ghost_y = obj_y - (dy / dist) * (2.0 * r)
        
        ideal_angle = np.degrees(np.arctan2(ghost_y - cue_y, ghost_x - cue_x))
        
        inputs.append([cue_x, cue_y, obj_x, obj_y])
        targets.append([ideal_angle])

    # PyTorchで扱えるTensor型に変換（GPUがある場合はGPUへ送る）
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs_tensor = torch.tensor(inputs, dtype=torch.float32).to(device)
    targets_tensor = torch.tensor(targets, dtype=torch.float32).to(device)
    return inputs_tensor, targets_tensor, device

# 【3】モデルの学習ループ
def train_model():
    inputs, targets, device = generate_synthetic_data(50000)
    
    model = PolicyNetwork().to(device)
    criterion = nn.MSELoss() # 損失関数：平均二乗誤差（角度のズレを計算）
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 1000
    print(f"\n--- 学習開始 (Device: {device}) ---")
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        # 予測と誤差計算
        predictions = model(inputs)
        loss = criterion(predictions, targets)
        
        # 逆伝播とパラメータ更新
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss(誤差): {loss.item():.4f}")

    # 学習済みモデルの保存
    torch.save(model.state_dict(), "policy_model.pth")
    print("\n学習完了！モデルを 'policy_model.pth' に保存しました。")

if __name__ == "__main__":
    train_model()
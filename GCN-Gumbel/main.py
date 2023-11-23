import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GCNConv
from torch.distributions import Normal

def gumbel_softmax(logits, temperature=1.0):
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits)))
    y = logits + gumbel_noise
    return torch.sigmoid(y / temperature)

class GCNWithGIB(nn.Module):
    def __init__(self, num_features, num_classes, num_edges):
        super(GCNWithGIB, self).__init__()
        self.conv1 = GCNConv(num_features, 16)
        self.conv2 = GCNConv(16, num_classes)
        self.edge_probs = nn.Parameter(torch.randn(num_edges))

    def forward(self, x, edge_index):
        edge_weights = gumbel_softmax(self.edge_probs)
        selected_edges = edge_index[:, edge_weights > 0.5]

        x = F.relu(self.conv1(x, selected_edges))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, selected_edges)

        return F.log_softmax(x, dim=1)

# 初始化模型和优化器
model = GCNWithGIB(num_features, num_classes, data.edge_index.size(1))
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 训练循环
for epoch in range(200):
    model.train()
    optimizer.zero_grad()

    # 前向传播
    output = model(data.x, data.edge_index)

    # 计算交叉熵损失
    loss = F.nll_loss(output[data.train_mask], data.y[data.train_mask])

    # 计算KL散度
    edge_prior = Normal(torch.zeros_like(model.edge_probs), torch.ones_like(model.edge_probs))
    edge_posterior = Normal(model.edge_probs, torch.ones_like(model.edge_probs))
    kl_divergence = torch.distributions.kl_divergence(edge_posterior, edge_prior).sum()

    # 联合优化
    total_loss = loss + beta * kl_divergence  # beta是超参数

    # 反向传播和优化
    total_loss.backward()
    optimizer.step()

    # 可以添加评估模型性能的代码

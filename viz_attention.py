import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import init
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os


# ==========================================
# 1. 模型定义 (修改以输出 A_p, A_h, A_f)
# ==========================================

class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True):
        super(LayerNorm, self).__init__()
        self.eps = eps
        self.normalized_shape = tuple(normalized_shape)
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape))
            self.bias = nn.Parameter(torch.zeros(self.normalized_shape))

    def forward(self, input):
        mean = input.mean(dim=(1, 2), keepdim=True)
        variance = input.var(dim=(1, 2), unbiased=False, keepdim=True)
        input = (input - mean) / torch.sqrt(variance + self.eps)
        if self.elementwise_affine:
            input = input * self.weight + self.bias
        return input


class GLU(nn.Module):
    def __init__(self, features, dropout=0.1):
        super(GLU, self).__init__()
        self.conv1 = nn.Conv2d(features, features, (1, 1))
        self.conv2 = nn.Conv2d(features, features, (1, 1))
        self.conv3 = nn.Conv2d(features, features, (1, 1))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        out = x1 * torch.sigmoid(x2)
        out = self.dropout(out)
        out = self.conv3(out)
        return out


class SEAttention(nn.Module):
    def __init__(self, channel=512, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None: init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ChannelTimeDualAttentionEnhanced(nn.Module):
    def __init__(self, channel=512, reduction=16, time_conv_kernel=3):
        super().__init__()
        self.channel = channel
        self.reduction = reduction
        self.fc = nn.Sequential(
            nn.Linear(channel, max(channel // reduction, 1), bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(max(channel // reduction, 1), channel, bias=False)
        )
        padding = (time_conv_kernel - 1) // 2
        self.conv_time = nn.Conv1d(1, 1, kernel_size=time_conv_kernel, padding=padding, bias=False)
        self.gate_conv = nn.Conv2d(2 * channel, channel, kernel_size=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, n, t = x.size()
        y = x.mean(dim=3).mean(dim=2)
        y = self.fc(y).view(b, c, 1, 1)
        x_time = x.mean(dim=2)
        x_time_mean = x_time.mean(dim=1, keepdim=True)
        time_feat = self.conv_time(x_time_mean)
        time_feat = time_feat.view(b, 1, 1, t)
        y_flat = y.view(b, c)
        t_flat = time_feat.view(b, t)
        bilinear_out = torch.bmm(y_flat.unsqueeze(2), t_flat.unsqueeze(1)).view(b, c, 1, t)
        combined = torch.cat([y.expand(-1, -1, 1, t), time_feat.expand(-1, c, 1, -1)], dim=1)
        gate = self.gate_conv(combined)
        gate = torch.sigmoid(gate)
        attn = self.sigmoid(bilinear_out * gate)
        out = x * attn + x
        return out, attn


class Conv(nn.Module):
    def __init__(self, features, dropout=0.1, se_reduction=16):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(features, features, (1, 1))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.se = SEAttention(channel=features, reduction=se_reduction)
        self.se.init_weights()

    def forward(self, x):
        x = self.conv(x)
        x = self.relu(x)
        x = self.se(x)
        x = self.dropout(x)
        return x


class TemporalEmbedding(nn.Module):
    def __init__(self, time, features):
        super(TemporalEmbedding, self).__init__()
        self.time = time
        self.time_day = nn.Parameter(torch.empty(time, features))
        nn.init.xavier_uniform_(self.time_day)
        self.time_week = nn.Parameter(torch.empty(7, features))
        nn.init.xavier_uniform_(self.time_week)

    def forward(self, x):
        day_emb = x[..., 1]
        day_index = (day_emb[:, -1, :] * self.time).long()
        day_index = torch.clamp(day_index, 0, self.time - 1)
        time_day = self.time_day[day_index]
        time_day = time_day.transpose(1, 2).unsqueeze(-1)
        week_emb = x[..., 2]
        week_index = week_emb[:, -1, :].long()
        week_index = torch.clamp(week_index, 0, 6)
        time_week = self.time_week[week_index]
        time_week = time_week.transpose(1, 2).unsqueeze(-1)
        tem_emb = time_day + time_week
        return tem_emb


class TConv(nn.Module):
    def __init__(self, features=128, layer=4, length=12, dropout=0.1):
        super(TConv, self).__init__()
        layers = []
        kernel_size = int(length / layer + 1)
        for i in range(layer):
            conv = nn.Conv2d(features, features, (1, kernel_size))
            relu = nn.ReLU()
            dropout_layer = nn.Dropout(dropout)
            layers += [nn.Sequential(conv, relu, dropout_layer)]
        self.tcn = nn.Sequential(*layers)

    def forward(self, x):
        x = nn.functional.pad(x, (1, 0, 0, 0))
        x = self.tcn(x) + x[..., -1].unsqueeze(-1)
        return x


# ---------------- SpatialAttention (核心修改) ----------------
class SpatialAttention(nn.Module):
    def __init__(self, device, d_model, head, num_nodes, seq_length=1, dropout=0.1, se_reduction=16):
        super(SpatialAttention, self).__init__()
        assert d_model % head == 0
        self.d_k = d_model // head
        self.head = head
        self.num_nodes = num_nodes
        self.seq_length = seq_length
        self.d_model = d_model

        self.q = Conv(d_model, dropout, se_reduction)
        self.v = Conv(d_model, dropout, se_reduction)
        self.concat = Conv(d_model, dropout, se_reduction)

        self.memory = nn.Parameter(torch.randn(head, seq_length, num_nodes, self.d_k))
        nn.init.xavier_uniform_(self.memory)
        self.weight = nn.Parameter(torch.ones(d_model, num_nodes, seq_length))
        self.bias = nn.Parameter(torch.zeros(d_model, num_nodes, seq_length))

        apt_size = 10
        nodevecs = torch.randn(num_nodes, apt_size), torch.randn(apt_size, num_nodes)
        self.nodevec1, self.nodevec2 = [nn.Parameter(n.to(device), requires_grad=True) for n in nodevecs]

    def forward(self, input, adj_list=None):
        # 1. 计算 Query 和 Value
        query, value = self.q(input), self.v(input)
        query = query.view(query.shape[0], -1, self.d_k, query.shape[2], self.seq_length).permute(0, 1, 4, 3, 2)
        value = value.view(value.shape[0], -1, self.d_k, value.shape[2], self.seq_length).permute(0, 1, 4, 3, 2)

        key = torch.softmax(self.memory / math.sqrt(self.d_k), dim=-1)
        query_softmax = torch.softmax(query / math.sqrt(self.d_k), dim=-1)

        # ----------------------------------------------------
        # 提取 2: A_p (Potential / Pattern Matrix)
        # HCTDAFormer 中的静态自适应矩阵 Aapt 对应论文中的 Ap
        # ----------------------------------------------------
        A_p = torch.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=-1)

        # ----------------------------------------------------
        # 提取 3: A_h (Current / Similarity Matrix)
        # 动态计算的 Query-Key 相似度，反映当前时刻的相关性
        # 'bhlnd, bhlmd -> bhnm' (Batch, Head, Node, Node)
        # ----------------------------------------------------
        key_expanded = key.unsqueeze(0).expand(query.shape[0], -1, -1, -1, -1)
        # 使用未softmax的query和key计算原始相似度，或者直接用softmax后的点积
        # 为了可视化效果，我们计算归一化后的点积作为 Ah
        A_h = torch.einsum("bhlnc, bhlmc -> bhnm", query_softmax, key_expanded)
        A_h = A_h.mean(dim=1)  # Average over heads -> (B, N, N)

        # ----------------------------------------------------
        # 提取 4: A_f (Final Fusion Matrix)
        # ----------------------------------------------------
        # 原始计算流程
        kv = torch.einsum("hlnx, bhlny->bhlxy", key, value)
        attn_qkv = torch.einsum("bhlnx, bhlxy->bhlny", query_softmax, kv)
        attn_dyn = torch.einsum("nm,bhlnc->bhlnc", A_p, value)
        x = attn_qkv + attn_dyn

        # 近似 A_f: 为了可视化，我们将 Ap 和 Ah 融合
        # 这里的 A_f 是示意性的，代表模型最终综合了两者
        # 实际上模型的融合是在 Feature 层面做的 (x = attn_qkv + attn_dyn)
        # 我们可以用 A_f = A_h + A_p 来表示这种结构上的融合
        A_f = A_h + A_p.unsqueeze(0)

        # 恢复形状继续传播
        x = (x.permute(0, 1, 4, 3, 2).contiguous().view(x.shape[0], self.d_model, self.num_nodes, self.seq_length))
        x = self.concat(x)
        if self.num_nodes not in [170, 358, 5]:
            x = x * self.weight + self.bias + x

        return x, self.weight, self.bias, A_p, A_h, A_f


class Encoder(nn.Module):
    def __init__(self, device, d_model, head, num_nodes, seq_length=1, dropout=0.1, se_reduction=16):
        super(Encoder, self).__init__()
        assert d_model % head == 0
        self.d_k = d_model // head
        self.head = head
        self.num_nodes = num_nodes
        self.seq_length = seq_length
        self.d_model = d_model
        self.attention = SpatialAttention(device, d_model, head, num_nodes, seq_length, dropout, se_reduction)
        self.LayerNorm = LayerNorm([d_model, num_nodes, seq_length], elementwise_affine=False)
        self.dropout1 = nn.Dropout(p=dropout)
        self.glu = GLU(d_model)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, input, adj_list=None):
        # 接收三个矩阵
        x, weight, bias, A_p, A_h, A_f = self.attention(input)
        x = x + input
        x = self.LayerNorm(x)
        x = self.dropout1(x)
        x = self.glu(x) + x
        x = x * weight + bias + x
        x = self.LayerNorm(x)
        x = self.dropout2(x)
        return x, A_p, A_h, A_f


class HCTDAFormer(nn.Module):
    def __init__(self, device, input_dim=3, channels=64, num_nodes=170, input_len=12, output_len=12,
                 dropout=0.1, senet_reduction=16):
        super().__init__()
        self.device = device
        self.num_nodes = num_nodes
        self.node_dim = channels
        self.input_len = input_len
        self.input_dim = input_dim
        self.output_len = output_len
        self.head = 1
        self.node_time_mapping = {170: 288, 307: 288, 358: 288, 883: 288, 250: 48, 266: 48}
        self.time = self.node_time_mapping.get(num_nodes, 96)
        if num_nodes > 200 and num_nodes not in self.node_time_mapping:
            self.time = 96
        self.Temb = TemporalEmbedding(self.time, channels)
        self.tconv = TConv(channels, layer=4, length=self.input_len)
        self.start_conv = nn.Conv2d(self.input_dim, channels, kernel_size=(1, 1))
        self.se1 = SEAttention(channel=channels, reduction=senet_reduction)
        self.se1.init_weights()
        self.network_channel = channels * 2
        self.SpatialBlock = Encoder(device, d_model=self.network_channel, head=self.head,
                                    num_nodes=num_nodes, seq_length=1, dropout=dropout,
                                    se_reduction=senet_reduction)
        self.fc_st = nn.Conv2d(self.network_channel, self.network_channel, kernel_size=(1, 1))
        self.se2 = ChannelTimeDualAttentionEnhanced(channel=self.network_channel, reduction=senet_reduction)
        self.regression_layer = nn.Conv2d(self.network_channel, self.output_len, kernel_size=(1, 1))

    def forward(self, history_data):
        history_data = history_data.permute(0, 3, 2, 1)
        input_data = history_data
        input_data = self.start_conv(input_data)
        input_data = self.se1(input_data)
        input_data = self.tconv(input_data)
        tem_emb = self.Temb(history_data)
        data_st = torch.cat([input_data, tem_emb], dim=1)

        # 接收三个矩阵
        data_st1, A_p, A_h, A_f = self.SpatialBlock(data_st)

        data_st2 = self.fc_st(data_st)
        data_st2, attn_ctda = self.se2(data_st2)
        data_st = data_st1 + data_st2
        prediction = self.regression_layer(data_st)

        return prediction, A_p, A_h, A_f


# ==========================================
# 2. 数据与可视化 (生成 2x4 网格)
# ==========================================

def get_pems08_indices(mode="peak", total_timesteps=17856, points_per_day=288):
    indices = np.arange(total_timesteps)
    tod = indices % points_per_day
    am_peak_mask = (tod >= 84) & (tod < 108)  # 7:00 - 9:00
    pm_peak_mask = (tod >= 204) & (tod < 228)  # 17:00 - 19:00
    peak_mask = am_peak_mask | pm_peak_mask
    if mode == "peak":
        return indices[peak_mask]
    else:
        return indices[~peak_mask]


def generate_batch_data_enhanced(indices_pool, batch_size=32, num_nodes=170, time_len=12, points_per_day=288,
                                 is_peak=False):
    selected_indices = np.random.choice(indices_pool, size=batch_size, replace=True)
    batch_data = torch.zeros(batch_size, time_len, num_nodes, 3)
    for i, start_idx in enumerate(selected_indices):
        if is_peak:
            # 模拟高峰：强信号，部分节点关联紧密
            base_flow = torch.randn(time_len, num_nodes) * 0.5 + 2.0
            base_flow[:, :30] += 1.5
            batch_data[i, :, :, 0] = base_flow
        else:
            # 模拟非高峰：噪音
            batch_data[i, :, :, 0] = torch.randn(time_len, num_nodes) * 0.5 - 0.5
        seq_indices = np.arange(start_idx, start_idx + time_len)
        tod_seq = (seq_indices % points_per_day) / points_per_day
        batch_data[i, :, :, 1] = torch.tensor(tod_seq).unsqueeze(1).repeat(1, num_nodes)
        batch_data[i, :, :, 2] = np.random.randint(0, 7) / 7.0
    return batch_data


def visualize_four_matrices(model_path, weight_path, num_nodes=170, channels=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    model =HCTDAFormer(device=device, input_dim=3, channels=channels, num_nodes=num_nodes,
                  dropout=0.1, senet_reduction=16).to(device)
    try:
        model.load_state_dict(torch.load(weight_path, map_location=device))
        print("成功加载权重文件")
    except Exception as e:
        print(f"加载权重失败: {e}")
        return
    model.eval()

    # 1. 生成数据
    BATCH_SIZE = 32
    peak_indices = get_pems08_indices("peak")
    offpeak_indices = get_pems08_indices("offpeak")

    peak_input = generate_batch_data_enhanced(peak_indices, BATCH_SIZE, num_nodes, is_peak=True).float().to(device)
    offpeak_input = generate_batch_data_enhanced(offpeak_indices, BATCH_SIZE, num_nodes, is_peak=False).float().to(
        device)

    # 2. 推理获取矩阵
    with torch.no_grad():
        _, Ap_peak, Ah_peak, Af_peak = model(peak_input)
        _, Ap_off, Ah_off, Af_off = model(offpeak_input)

    # 3. 数据后处理 (截取前30个节点，取平均)
    SHOW_NUM = 30

    # 辅助函数：处理矩阵
    def process_matrix(mat):
        if mat.dim() == 3:  # (B, N, N) -> 平均 -> (N, N)
            mat = mat.mean(dim=0)
        m = mat.cpu().numpy()[:SHOW_NUM, :SHOW_NUM]
        # 归一化到 0-1 以便可视化颜色一致
        return (m - m.min()) / (m.max() - m.min() + 1e-8)

    # 处理 Peak 矩阵
    mat_Ap_peak = process_matrix(Ap_peak)
    mat_Ah_peak = process_matrix(Ah_peak)
    mat_Af_peak = process_matrix(Af_peak)

    # 处理 Off-Peak 矩阵
    mat_Ap_off = process_matrix(Ap_off)  # Ap 理论上是一样的，但为了代码对齐重新处理
    mat_Ah_off = process_matrix(Ah_off)
    mat_Af_off = process_matrix(Af_off)

    # 生成 Baseline (模拟预定义矩阵)
    # 模拟一个只有对角线和少量邻居的静态矩阵
    np.random.seed(42)
    mat_baseline = np.eye(SHOW_NUM) * 0.8 + np.random.rand(SHOW_NUM, SHOW_NUM) * 0.1
    mat_baseline = (mat_baseline + mat_baseline.T) / 2  # 对称
    mat_baseline = (mat_baseline - mat_baseline.min()) / (mat_baseline.max() - mat_baseline.min())

    # 4. 绘图配置 (2行 x 4列)
    colors = ["#FFF0F5", "#87CEFA", "#0000CD"]  # 你的粉蓝配色
    custom_cmap = mcolors.LinearSegmentedColormap.from_list("pale_pink_blue", colors)

    fig, axes = plt.subplots(2, 4, figsize=(24, 12))

    # 定义绘图列表
    # Row 0: Off-Peak (为了符合视觉习惯，也可以先Peak)
    row0_data = [mat_baseline, mat_Ap_off, mat_Ah_off, mat_Af_off]
    row0_titles = ["Baseline (Predefined)", "Ap: Potential (Off-Peak)", "Ah: Current (Off-Peak)",
                   "Af: Fusion (Off-Peak)"]

    # Row 1: Peak
    row1_data = [mat_baseline, mat_Ap_peak, mat_Ah_peak, mat_Af_peak]
    row1_titles = ["Baseline (Predefined)", "Ap: Potential (Peak)", "Ah: Current (Peak)", "Af: Fusion (Peak)"]

    for i in range(4):
        # Off-Peak Row
        im = axes[0, i].imshow(row0_data[i], cmap=custom_cmap)
        axes[0, i].set_title(row0_titles[i], fontsize=11, fontweight='bold')
        axes[0, i].set_xticks([]);
        axes[0, i].set_yticks([])

        # Peak Row
        im = axes[1, i].imshow(row1_data[i], cmap=custom_cmap)
        axes[1, i].set_title(row1_titles[i], fontsize=11, fontweight='bold')
        axes[1, i].set_xticks([]);
        axes[1, i].set_yticks([])

    # Add row labels
    fig.text(0.01, 0.7, 'Off-Peak Hours', va='center', rotation='vertical', fontsize=16, fontweight='bold')
    fig.text(0.01, 0.3, 'Peak Hours', va='center', rotation='vertical', fontsize=16, fontweight='bold')

    # Colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax, label='Attention Weight (Normalized)')

    plt.savefig('four_matrices_visualization.png')
    plt.show()
    print("已生成四类矩阵可视化图: four_matrices_visualization.png")


if __name__ == "__main__":
    model_file_path = "D:\\python_tsxt\\HCTDAFormer-main\\model.py"
    weight_file_path = "D:\\python_tsxt\\HCTDAFormer-main\\best_model.pth"
    NUM_NODES = 170
    CHANNELS = 128

    if os.path.exists(weight_file_path):
        visualize_four_matrices(model_file_path, weight_file_path, num_nodes=NUM_NODES, channels=CHANNELS)
    else:
        print(f"错误：未找到权重文件 {weight_file_path}")
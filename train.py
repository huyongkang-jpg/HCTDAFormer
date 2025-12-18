import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn import init
import numpy as np
import matplotlib.pyplot as plt
import os


# ---------------- LayerNorm ----------------
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


# ---------------- GLU ----------------
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
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# ---------------- ChannelTimeDualAttentionEnhanced ----------------
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

    # ---------------- Conv ----------------


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


# ---------------- TemporalEmbedding ----------------
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


# ---------------- TConv ----------------
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


# ---------------- SpatialAttention ----------------
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
        query, value = self.q(input), self.v(input)
        query = query.view(query.shape[0], -1, self.d_k, query.shape[2], self.seq_length).permute(0, 1, 4, 3, 2)
        value = value.view(value.shape[0], -1, self.d_k, value.shape[2], self.seq_length).permute(0, 1, 4, 3, 2)
        key = torch.softmax(self.memory / math.sqrt(self.d_k), dim=-1)
        query = torch.softmax(query / math.sqrt(self.d_k), dim=-1)

        # **Aapt**
        Aapt = torch.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec2)), dim=-1)

        kv = torch.einsum("hlnx, bhlny->bhlxy", key, value)
        attn_qkv = torch.einsum("bhlnx, bhlxy->bhlny", query, kv)
        attn_dyn = torch.einsum("nm,bhlnc->bhlnc", Aapt, value)
        x = attn_qkv + attn_dyn
        x = (x.permute(0, 1, 4, 3, 2).contiguous().view(x.shape[0], self.d_model, self.num_nodes, self.seq_length))
        x = self.concat(x)
        if self.num_nodes not in [170, 358, 5]:
            x = x * self.weight + self.bias + x
        return x, self.weight, self.bias, Aapt

    # ---------------- Encoder ----------------


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
        x, weight, bias, Aapt = self.attention(input)
        x = x + input
        x = self.LayerNorm(x)
        x = self.dropout1(x)
        x = self.glu(x) + x
        x = x * weight + bias + x
        x = self.LayerNorm(x)
        x = self.dropout2(x)
        return x, Aapt


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
        # --- 修正步骤 1：先 permute ---
        history_data = history_data.permute(0, 3, 2, 1)
        # --- 修正步骤 2：再赋值 ---
        input_data = history_data

        input_data = self.start_conv(input_data)
        input_data = self.se1(input_data)
        input_data = self.tconv(input_data)

        tem_emb = self.Temb(history_data)
        data_st = torch.cat([input_data, tem_emb], dim=1)

        data_st1, Aapt = self.SpatialBlock(data_st)

        data_st2 = self.fc_st(data_st)
        data_st2, attn_ctda = self.se2(data_st2)

        data_st = data_st1 + data_st2
        prediction = self.regression_layer(data_st)
        return prediction, Aapt, attn_ctda


# --- 可视化函数 (已修改：只取前30个节点) ---
def visualize_attention(model_path, weight_path, num_nodes=170, channels=128):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 初始化模型
    model = HCTDAFormer(
        device=device,
        input_dim=3,
        channels=channels,
        num_nodes=num_nodes,
        input_len=12,
        output_len=12,
        dropout=0.1,
        senet_reduction=16
    ).to(device)

    # 加载权重
    try:
        model.load_state_dict(torch.load(weight_path, map_location=device))
        print(f"成功加载权重文件")
    except Exception as e:
        print(f"加载权重文件失败: {e}")
        return

    model.eval()

    # 模拟输入
    dummy_input = torch.randn(1, 12, num_nodes, 3).float().to(device)

    with torch.no_grad():
        _, aapt_attn, ctda_attn = model(dummy_input)

    # --- 可视化 Aapt (修改处：只显示前 30 个节点) ---
    aapt_matrix = aapt_attn.cpu().numpy()

    # ********** 关键修改 **********
    SHOW_NUM = 30  # 设置要显示的节点数量
    if aapt_matrix.shape[0] > SHOW_NUM:
        print(f"原始矩阵大小 {aapt_matrix.shape}, 正在截取前 {SHOW_NUM} 个节点进行可视化...")
        aapt_matrix = aapt_matrix[:SHOW_NUM, :SHOW_NUM]
    # ******************************

    plt.figure(figsize=(10, 10))
    plt.imshow(aapt_matrix, cmap='viridis')
    plt.colorbar(label='Attention Weight (Softmax)')
    plt.title(f'Spatial Adaptive Attention Matrix (First {SHOW_NUM} Nodes)')
    plt.xlabel(f'Node Index (0-{SHOW_NUM - 1})')
    plt.ylabel(f'Node Index (0-{SHOW_NUM - 1})')
    # 调整刻度，使其更清晰
    plt.xticks(np.arange(0, SHOW_NUM, 2))
    plt.yticks(np.arange(0, SHOW_NUM, 2))
    plt.savefig('spatial_attention_map_top30.png')
    plt.show()
    print("已生成空间注意力图: spatial_attention_map_top30.png")

    # --- 可视化 CTDA (保持不变) ---
    time_attn_vector = ctda_attn[0, :, 0, :].mean(dim=0).cpu().numpy()
    plt.figure(figsize=(8, 4))
    plt.plot(np.arange(len(time_attn_vector)), time_attn_vector, marker='o')
    plt.title('Channel-Time Dual Attention (Time Dimension Avg-Pooled)')
    plt.xlabel('Time Step Index (T=12)')
    plt.ylabel('Average Attention Weight')
    plt.grid(True)
    plt.savefig('time_attention_vector.png')
    plt.show()
    print("已生成时间注意力向量图: time_attention_vector.png")


if __name__ == "__main__":
    model_file_path = "D:\\python_tsxt\\HCTDAFormer-main\\model.py"
    weight_file_path = "D:\\python_tsxt\\HCTDAFormer-main\\best_model.pth"

    # 这里的参数必须和权重文件匹配，不能改！
    # 只有画图的时候才切片，模型加载时必须是 full size
    NUM_NODES = 170
    CHANNELS = 128

    if os.path.exists(weight_file_path):
        visualize_attention(model_file_path, weight_file_path, num_nodes=NUM_NODES, channels=CHANNELS)
    else:
        print(f"错误：未找到权重文件 {weight_file_path}")
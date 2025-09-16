import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
import numpy as np


class SubjectEmbedding(nn.Module):
    def __init__(self, num_subjects, embedding_dim=16):
        super(SubjectEmbedding, self).__init__()
        self.embedding = nn.Embedding(num_subjects, embedding_dim)

    def forward(self, x):
        # x should be the subject indices
        return self.embedding(x)


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=8):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1)
        return x * y.expand_as(x)

class ResidualSECNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, pool_size=2, dropout=0.3):
        super().__init__()
        
        # First conv block
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        # Second conv block
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # SE block
        self.se = SEBlock(out_channels)
        
        # Shortcut connection
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        
        self.pool = nn.MaxPool1d(pool_size)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        shortcut = self.shortcut(x)
        
        # First conv
        out = F.relu(self.bn1(self.conv1(x)))
        # Second conv
        out = self.bn2(self.conv2(out))
        
        # SE block
        out = self.se(out)
        
        # Add shortcut
        out += shortcut
        out = F.relu(out)
        
        # Pool and dropout
        out = self.pool(out)
        out = self.dropout(out)
        
        return out

class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)
        
    def forward(self, x):
        # x shape: (batch, seq_len, hidden_dim)
        scores = torch.tanh(self.attention(x))  # (batch, seq_len, 1)
        #mask = torch.arange(x.size(1), device=x.device).unsqueeze(0) < torch.LongTensor(valid_len).to(device).unsqueeze(1)  # (B, T)
        #scores = scores.masked_fill(~mask.unsqueeze(-1), float('-inf'))  # Apply mask
        weights = F.softmax(scores.squeeze(-1), dim=1)  # (batch, seq_len)
        context = torch.sum(x * weights.unsqueeze(-1), dim=1)  # (batch, hidden_dim)
        return context

class independent_conv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(1,6,kernel_size=3,padding=1)

    def forward(self,x):
        # x [N,seq_len,channels]
        channels = x.size(2)
        seq_len = x.size(1)
        batch_size=x.size(0)
        x = x.permute(0,2,1)
        x = x.reshape(batch_size*channels,1,seq_len)
        x = self.conv(x)
        x = x.reshape(batch_size,channels*6,seq_len)
        x = x.permute(0,2,1)
        return x

class BilinearSym(nn.Module):
    def __init__(self, in_dim,out_dim):
        super().__init__()
        self.U = nn.Linear(in_dim, out_dim)  # 独立变换U
        self.V = nn.Linear(in_dim, out_dim)  # 独立变换V
    
    def forward(self, x):
        return self.U(x) * self.V(x) + self.U(-x) * self.V(-x)

class IMUBackBone(nn.Module):
    def __init__(self):
        super().__init__()
        
        # IMU deep branch
        self.acc_block1 = ResidualSECNNBlock(3, 32, 3, dropout=0.3)
        self.acc_block2 = ResidualSECNNBlock(32, 64, 5, dropout=0.3,pool_size=2)
        self.acc_block1__2 = ResidualSECNNBlock(3, 32, 3, dropout=0.3)
        self.acc_block2__2 = ResidualSECNNBlock(32, 64, 5, dropout=0.3,pool_size=2)

        self.rot_block1 = ResidualSECNNBlock(4, 32, 3, dropout=0.3)
        self.rot_block2 = ResidualSECNNBlock(32, 64, 5, dropout=0.3,pool_size=2)
        self.rot_block1__2 = ResidualSECNNBlock(4, 32, 3, dropout=0.3)
        self.rot_block2__2 = ResidualSECNNBlock(32, 64, 5, dropout=0.3,pool_size=2)

        self.accnew_block1 = ResidualSECNNBlock(3, 32, 3, dropout=0.3)
        self.accnew_block2 = ResidualSECNNBlock(32, 64, 5, dropout=0.3,pool_size=2)
        self.accnew_block1__2 = ResidualSECNNBlock(3, 32, 3, dropout=0.3)
        self.accnew_block2__2 = ResidualSECNNBlock(32, 64, 5, dropout=0.3,pool_size=2)

        self.rotdiff_block1 = ResidualSECNNBlock(4, 32, 3, dropout=0.3)
        self.rotdiff_block2 = ResidualSECNNBlock(32, 64, 5, dropout=0.3,pool_size=2)
        self.rotdiff_block1__2 = ResidualSECNNBlock(4, 32, 3, dropout=0.3)
        self.rotdiff_block2__2 = ResidualSECNNBlock(32, 64, 5, dropout=0.3,pool_size=2)

        self.rotdiff2_block1 = ResidualSECNNBlock(4, 32, 3, dropout=0.3)
        self.rotdiff2_block2 = ResidualSECNNBlock(32, 64, 5, dropout=0.3,pool_size=2)
        self.rotdiff2_block1__2 = ResidualSECNNBlock(4, 32, 3, dropout=0.3)
        self.rotdiff2_block2__2 = ResidualSECNNBlock(32, 64, 5, dropout=0.3,pool_size=2)

        self.extfea_block1 = ResidualSECNNBlock(4, 32, 3, dropout=0.3)
        self.extfea_block2 = ResidualSECNNBlock(32, 64, 5, dropout=0.3,pool_size=2)
        self.extfea_block1__2 = ResidualSECNNBlock(4, 32, 3, dropout=0.3)
        self.extfea_block2__2 = ResidualSECNNBlock(32, 64, 5, dropout=0.3,pool_size=2)

        self.ang_jerk_block1 = ResidualSECNNBlock(3, 32, 3, dropout=0.3)
        self.ang_jerk_block2 = ResidualSECNNBlock(32, 64, 5, dropout=0.3,pool_size=2)
        self.ang_jerk_block1__2 = ResidualSECNNBlock(3, 32, 3, dropout=0.3)
        self.ang_jerk_block2__2 = ResidualSECNNBlock(32, 64, 5, dropout=0.3,pool_size=2)

        self.ang_snap_block1 = ResidualSECNNBlock(3, 32, 3, dropout=0.3)
        self.ang_snap_block2 = ResidualSECNNBlock(32, 64, 5, dropout=0.3,pool_size=2)
        self.ang_snap_block1__2 = ResidualSECNNBlock(3, 32, 3, dropout=0.3)
        self.ang_snap_block2__2 = ResidualSECNNBlock(32, 64, 5, dropout=0.3,pool_size=2)


        self.acc_jerk_block1 = ResidualSECNNBlock(3, 32, 3, dropout=0.3)
        self.acc_jerk_block2 = ResidualSECNNBlock(32, 64, 5, dropout=0.3,pool_size=2)
        self.acc_jerk_block1__2 = ResidualSECNNBlock(3, 32, 3, dropout=0.3)
        self.acc_jerk_block2__2 = ResidualSECNNBlock(32, 64, 5, dropout=0.3,pool_size=2)

        self.acc_snap_block1 = ResidualSECNNBlock(3, 32, 3, dropout=0.3)
        self.acc_snap_block2 = ResidualSECNNBlock(32, 64, 5, dropout=0.3,pool_size=2)
        self.acc_snap_block1__2 = ResidualSECNNBlock(3, 32, 3, dropout=0.3)
        self.acc_snap_block2__2 = ResidualSECNNBlock(32, 64, 5, dropout=0.3,pool_size=2)
       
    def forward(self, x):
        # x [batch,seq_len,channel]
        batch = x.size(0)
        seq_len = x.size(1)

        x = x.permute(0,2,1)  

        # IMU branch
        acc = self.acc_block1(x[:,:3,:])
        acc = self.acc_block2(acc)
        acc__2 = self.acc_block1__2(x[:,:3,:])
        acc__2 = self.acc_block2__2(acc__2)

        #rot_bi = self.rot_biblock(x[:, 3:7, :].permute(0,2,1)).permute(0,2,1)
        rot_bi = x[:, 3:7, :]
        rot = self.rot_block1(rot_bi)+self.rot_block1(-rot_bi)
        rot = self.rot_block2(rot)
        rot__2 = self.rot_block1__2(rot_bi)+self.rot_block1__2(-rot_bi)
        rot__2 = self.rot_block2__2(rot__2)

        accnew = self.accnew_block1(x[:,7:10,:])
        accnew = self.accnew_block2(accnew)
        accnew__2 = self.accnew_block1__2(x[:, 7:10, :])
        accnew__2 = self.accnew_block2__2(accnew__2)

        rotdiff = self.rotdiff_block1(x[:, 10:14, :])#+self.rotdiff_block1(-x[:, 10:14, :])
        rotdiff = self.rotdiff_block2(rotdiff)
        rotdiff__2 = self.rotdiff_block1__2(x[:, 10:14, :])#+self.rotdiff_block1__2(-x[:, 10:14, :])
        rotdiff__2 = self.rotdiff_block2__2(rotdiff__2)

        extfea = self.extfea_block1(x[:,14:18,:])
        extfea = self.extfea_block2(extfea)
        extfea__2 = self.extfea_block1__2(x[:, 14:18, :])
        extfea__2 = self.extfea_block2__2(extfea__2)

        rotdiff2 =    self.rotdiff2_block1(x[:, 18:22, :])#+self.rotdiff2_block1(-x[:, 18:22, :])
        rotdiff2 =    self.rotdiff2_block2(rotdiff2)
        rotdiff2__2 = self.rotdiff2_block1__2(x[:, 18:22,:])#+self.rotdiff2_block1__2(-x[:, 18:22 :])
        rotdiff2__2 = self.rotdiff2_block2__2(rotdiff2__2)

        ang_jerk =    self.ang_jerk_block1(x[:, 22:25, :])#+self.rotdiff2_block1(-x[:, 18:22, :])
        ang_jerk =    self.ang_jerk_block2(ang_jerk)
        ang_jerk__2 =    self.ang_jerk_block1__2(x[:, 22:25, :])#+self.rotdiff2_block1(-x[:, 18:22, :])
        ang_jerk__2 =    self.ang_jerk_block2__2(ang_jerk__2)

        ang_snap =    self.ang_snap_block1(x[:, 25:28, :])#+self.rotdiff2_block1(-x[:, 18:22, :])
        ang_snap =    self.ang_snap_block2(ang_snap)
        ang_snap__2 =    self.ang_snap_block1__2(x[:, 25:28, :])#+self.rotdiff2_block1(-x[:, 18:22, :])
        ang_snap__2 =    self.ang_snap_block2__2(ang_snap__2)

        acc_jerk =    self.ang_jerk_block1(x[:, 28:31, :])#+self.rotdiff2_block1(-x[:, 18:22, :])
        acc_jerk =    self.ang_jerk_block2(acc_jerk)
        acc_jerk__2 =    self.ang_jerk_block1__2(x[:, 28:31, :])#+self.rotdiff2_block1(-x[:, 18:22, :])
        acc_jerk__2 =    self.ang_jerk_block2__2(acc_jerk__2)

        acc_snap =    self.ang_snap_block1(x[:, 31:34, :])#+self.rotdiff2_block1(-x[:, 18:22, :])
        acc_snap =    self.ang_snap_block2(acc_snap)
        acc_snap__2 =    self.ang_snap_block1__2(x[:, 31:34, :])#+self.rotdiff2_block1(-x[:, 18:22, :])
        acc_snap__2 =    self.ang_snap_block2__2(acc_snap__2)

        # Concatenate branches
        merged = torch.cat([acc,acc__2,
                            rot,rot__2,
                            accnew,accnew__2,
                            rotdiff,rotdiff__2,
                            rotdiff2,rotdiff2__2,
                            extfea,extfea__2,
                            ang_jerk,ang_jerk__2,
                            ang_snap,ang_snap__2,
                            acc_jerk,acc_jerk__2,
                            acc_snap,acc_snap__2
                            ], dim=1).transpose(1, 2)  # (batch, seq_len, 256)

        return merged
      

###################################################################################
# 全量传感器模型
###################################################################################
class CoordinateAttention(nn.Module):
    def __init__(self, in_channels, reduction=32):
        super(CoordinateAttention, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))  # 高度方向的池化
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))  # 宽度方向的池化
        
        # 中间通道数（缩减后）
        mid_channels = max(8, in_channels // reduction)
        
        # 1x1卷积用于特征转换
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.act = nn.ReLU(inplace=True)
        
        # 分离处理两个空间方向的卷积
        self.conv_h = nn.Conv2d(mid_channels, in_channels, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mid_channels, in_channels, kernel_size=1, stride=1, padding=0)
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        identity = x
        
        n, c, h, w = x.size()
        
        # 水平方向池化 [n,c,h,1]
        x_h = self.pool_h(x)
        # 垂直方向池化 [n,c,w,1]
        x_w = self.pool_w(x).permute(0,1,3,2)
        # 拼接两个方向的特征 [n,c,h+w,1]
        x_cat = torch.cat([x_h, x_w], dim=2)
        
        # 1x1卷积 + BN + 激活
        out = self.conv1(x_cat)
        out = self.bn1(out)
        out = self.act(out)

        # 拆分回水平和垂直分量
        x_h, x_w = torch.split(out, [h, w], dim=2)
        
        # 1x1卷积生成注意力图
        att_h = self.sigmoid(self.conv_h(x_h))
        att_w = self.sigmoid(self.conv_w(x_w))
        
        # 应用注意力权重
        return identity * att_h * att_w


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        # 使用带瓶颈层的多层感知机
        self.mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        channel_att = self.sigmoid(avg_out + max_out)
        return x * channel_att

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "Kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 沿通道维度计算平均和最大值
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.conv(spatial_att)
        spatial_att = self.sigmoid(spatial_att)
        return x * spatial_att

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16, spatial_kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_att = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_att = SpatialAttention(spatial_kernel_size)

    def forward(self, x):
        # 通道注意力 -> 空间注意力
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x



class CoorderAdder(nn.Module):
    def __init__(self):
        super().__init__()
        xx,yy = torch.meshgrid(torch.arange(0, 8), torch.arange(0, 8))
        #xx = (xx-3.5)/2.3
        #yy = (yy-3.5)/2.3
        self.coorder = torch.stack((xx, yy), dim=0).float().unsqueeze(0).to(device) # (1, 2, 8, 8)
    def forward(self, x):
        x = torch.cat((x, self.coorder.repeat(x.size(0), 1, 1, 1)),dim=1)
        return x
    


class TofBlock(nn.Module):
    def __init__(self, out_channels,out_dim, kernel_size=3, pool_size=2, dropout=0.3):
        super().__init__()
        self.coorderadd = CoorderAdder()  #增加位置编码
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(6+2, out_channels, kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(pool_size),
            nn.ReLU(),
            nn.Dropout(dropout),
   )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=kernel_size//2, bias=False),
            nn.BatchNorm2d(out_channels),
            #nn.MaxPool2d(pool_size),
            nn.ReLU(),
            nn.Dropout(dropout)
            )
        self.linear = nn.Linear(out_channels * (8 // pool_size) * (8 // pool_size),out_dim)
     

    def forward(self, x):
        # x shape: (batch,seq_len,4, height, width)
        batch = x.size(0)
        seq_len = x.size(1)
       
        x = x.reshape(batch*seq_len,6,x.size(3), x.size(4))  # (batch*seq_len, in_channels, height, width)
        x = self.coorderadd(x)  # Add positional encoding 
        
        x = self.conv_block1(x)
        x = self.conv_block2(x)

        x = x.reshape(batch, seq_len, x.size(1), x.size(2), x.size(3))  # (batch, seq_len, out_channels, height, width)
        x = x.flatten(start_dim=2)  # (batch, seq_len, out_channels * height * width)
        x = self.linear(x)
        x = F.relu(x) # (batch, seq_len, out_dim)
        return x


class ThmTofBackBone(nn.Module):
    def __init__(self):
        super().__init__()
 
        self.thm_block1 = ResidualSECNNBlock(5, 32, 3, dropout=0.3)
        self.thm_block2 = ResidualSECNNBlock(32, 64, 5, dropout=0.3)
        self.thm_block1__2 = ResidualSECNNBlock(5, 32, 3, dropout=0.3)
        self.thm_block2__2 = ResidualSECNNBlock(32, 64, 5, dropout=0.3)

        self.tof_groups = 5
        self.tof_time_conv = nn.Sequential(nn.Conv1d(
            in_channels=1, 
            out_channels=6, 
            kernel_size=3, 
            padding=1, 
            bias=False
        )
        )
        
        self.tof_space_cnn = nn.ModuleList([
            TofBlock(out_channels=32,out_dim=64,kernel_size=3, pool_size=1, dropout=0.3) for _ in range(self.tof_groups)])

        #TOF lighter branch
        self.tof_conv1 = nn.Conv1d(64*self.tof_groups, 128*self.tof_groups, 3, padding=1, bias=False,groups=self.tof_groups)
        self.tof_bn1 = nn.BatchNorm1d(128*self.tof_groups)
        self.tof_pool1 = nn.MaxPool1d(2)
        self.tof_drop1 = nn.Dropout(0.3)
        
        self.tof_conv2 = nn.Conv1d(128*self.tof_groups, 64*self.tof_groups, 5, padding=2, bias=False, groups=self.tof_groups)
        self.tof_bn2 = nn.BatchNorm1d(64*self.tof_groups)
        self.tof_pool2 = nn.MaxPool1d(2)
        self.tof_drop2 = nn.Dropout(0.3)
        
    def forward(self, x):
        x = x.permute(0,2,1)

        thm_input = x[:, 0:5, :]
        tof_input = x[:, 5:5+5*64, :]

        thm = self.thm_block1(thm_input)
        thm = self.thm_block2(thm)
        thm__2 = self.thm_block1__2(thm_input)
        thm__2 = self.thm_block2__2(thm__2)
        
        tof_input = tof_input.permute(0,2,1)
        # tof简单时间卷积
        tof_raw = tof_input.reshape(x.size(0), 128, self.tof_groups, 64).permute(0,2,3,1)  # (batch, tof_dim, 64,seq_len)
        tof_raw = tof_raw.reshape(x.size(0)*5*64,1,128)   #[batch*5*64,1,seq_len]
        tof_raw = self.tof_time_conv(tof_raw)  #[batch*5*64,expand_dim,seq_len]

        # tof空间卷积
        tof_raw = tof_raw.reshape(x.size(0),5,8,8,6,128)  
        tof = tof_raw.permute(0,1,5,4,2,3) # (batch, tof_dim, seq_len,3, height, width)
        tof = [self.tof_space_cnn[i](tof[:,i,:,:,:,:]) for i in range(self.tof_groups)]  # Apply each TofBlock

        # tof时间分组卷积
        tof = torch.cat(tof,dim=2)  # (batch, seq_len, 5*out_dim)

        #tof = torch.cat([tof[0],x[:, :, 14:15],tof[1],x[:, :, 15:16],tof[2],x[:, :, 16:17],tof[3],x[:, :, 17:18],tof[4],x[:, :, 18:19]],dim=2)
        tof = F.relu(self.tof_bn1(self.tof_conv1(tof.transpose(1, 2))))
        tof = self.tof_drop1(self.tof_pool1(tof))
        tof = F.relu(self.tof_bn2(self.tof_conv2(tof)))
        tof = self.tof_drop2(self.tof_pool2(tof))

        merged = torch.cat([
                            thm,thm__2,#thm__3,thm__4,
                            tof
                            ], dim=1).transpose(1, 2)  # (batch, seq_len, 256)

        return merged

class RNNClassifier(nn.Module):
    def __init__(self,imu_only,imu_dim,thmtof_dim,n_classes,fea_std=None,fea_mean=None):
        super().__init__()
        self.imu_dim = imu_dim
        self.n_classes = n_classes
        self.imu_only = imu_only

        if imu_only:
            self.total_dim = imu_dim
        else:
            self.total_dim = imu_dim+thmtof_dim
        if fea_std is not None:
            self.register_buffer('fea_std',fea_std)
        else:
            self.register_buffer('fea_std',torch.FloatTensor(np.zeros(self.total_dim)))
        if fea_mean is not None:
            self.register_buffer('fea_mean',fea_mean)
        else:
            self.register_buffer('fea_mean',torch.FloatTensor(np.zeros(self.total_dim)))

        self.imu_block = IMUBackBone()
        self.bigru_imu = nn.GRU(64*10*2, 128, bidirectional=True, batch_first=True,num_layers=1)
        gru_dim = 128*2
        if not imu_only:
            self.thmtof_block = ThmTofBackBone()
            self.bigru_thmtof = nn.GRU(64*1*2+64*5, 128, bidirectional=True, batch_first=True,num_layers=1)
            gru_dim = 128*4

        self.gru_dropout = nn.Dropout(0.4)

        self.attention = AttentionLayer(gru_dim)  # 128*2 for bidirectional  
        
        self.output_head = nn.Sequential(
        nn.Linear(gru_dim, 256, bias=False),
        nn.BatchNorm1d(256),
        nn.ReLU(),
        nn.Dropout(0.5),
        
        nn.Linear(256, 128, bias=False),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.3),
        
        nn.Linear(128, n_classes)
        )


        
    def forward(self, x):
        if self.imu_only:
            x = x[:,:,0:self.imu_dim]
        if torch.sum(self.fea_std)!=0 and torch.sum(self.fea_mean)!=0:
            x = (x-self.fea_mean)/self.fea_std
        merged1 = self.imu_block(x[:,:,0:self.imu_dim])
        gru_out, _ = self.bigru_imu(merged1)
        gru_out = self.gru_dropout(gru_out)

        if not self.imu_only:
            merged2 = self.thmtof_block(x[:,:,self.imu_dim:])
            gru_out2, _ = self.bigru_thmtof(merged2)
            gru_out2 = self.gru_dropout(gru_out2)
            gru_out = torch.cat([gru_out,gru_out2],dim=2)
        # Attention
        attended = self.attention(gru_out)
        
        logits = self.output_head(attended)
        return logits


if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
    # Set device to cuda:1 (second GPU)
    device = torch.device('cuda:1')
    print(f"Found 2 GPUs. Using GPU 1: {torch.cuda.get_device_name(1)}")
elif torch.cuda.is_available() and torch.cuda.device_count() == 1:
    device = torch.device('cuda:0')
    print(f"Found 1 GPUs. Using GPU 0: {torch.cuda.get_device_name(0)}")
else:
    device = torch.device('cpu')


if __name__=="__main__":
    from torchinfo import summary
    in_channels=34
    model = RNNClassifier(imu_dim=34,imu_only=True).to(device)
    
    input_data = {
        "x": torch.ones(64, 128,in_channels).to(device)
    }
    print(summary(model, input_data=input_data))

    in_channels=34+5+64*5
    model =RNNClassifier(imu_dim=34,imu_only=False).to(device)
    
    input_data = {
        "x": torch.ones(64, 128,in_channels).to(device)
    }
    print(summary(model, input_data=input_data))



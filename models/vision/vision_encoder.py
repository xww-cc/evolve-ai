#!/usr/bin/env python3
"""
视觉编码器 - CNN + Transformer 架构
实现图像特征提取和视觉注意力机制
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np

class VisionEncoder(nn.Module):
    """视觉编码器 - CNN + Transformer 架构"""
    
    def __init__(self, 
                 input_channels: int = 3,
                 hidden_dim: int = 256,
                 num_layers: int = 6,
                 num_heads: int = 8,
                 dropout: float = 0.1,
                 image_size: int = 224):
        super().__init__()
        
        self.input_channels = input_channels
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.image_size = image_size
        
        # CNN特征提取器
        self.cnn_encoder = self._build_cnn_encoder()
        
        # Transformer编码器
        self.transformer_encoder = self._build_transformer_encoder()
        
        # 空间位置编码
        self.spatial_encoding = self._build_spatial_encoding()
        
        # 输出投影层
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        
        # 特征融合层
        self.feature_fusion = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
    def _build_cnn_encoder(self) -> nn.Module:
        """构建CNN编码器"""
        return nn.Sequential(
            # 第一层卷积
            nn.Conv2d(self.input_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # 第二层卷积
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # 第三层卷积
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            # 第四层卷积
            nn.Conv2d(256, self.hidden_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.hidden_dim),
            nn.ReLU(inplace=True),
            
            # 全局平均池化
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
    
    def _build_transformer_encoder(self) -> nn.Module:
        """构建Transformer编码器"""
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dim_feedforward=self.hidden_dim * 4,
            dropout=self.dropout,
            batch_first=True
        )
        return nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
    
    def _build_spatial_encoding(self) -> nn.Module:
        """构建空间位置编码"""
        return nn.Parameter(torch.randn(1, 196, self.hidden_dim))  # 14x14=196
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            x: 输入图像 [batch_size, channels, height, width]
            
        Returns:
            包含视觉特征的字典
        """
        batch_size = x.size(0)
        
        # CNN特征提取
        cnn_features = self.cnn_encoder(x)  # [batch_size, hidden_dim]
        
        # 重塑为序列形式
        sequence_features = cnn_features.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        # 添加空间位置编码
        spatial_features = self.spatial_encoding.expand(batch_size, -1, -1)
        
        # 合并特征
        combined_features = torch.cat([sequence_features, spatial_features], dim=1)
        
        # Transformer编码
        transformer_output = self.transformer_encoder(combined_features)
        
        # 特征融合
        fused_features, attention_weights = self.feature_fusion(
            transformer_output, transformer_output, transformer_output
        )
        
        # 输出投影
        output_features = self.output_projection(fused_features)
        
        return {
            'features': output_features,
            'attention_weights': attention_weights,
            'cnn_features': cnn_features,
            'transformer_features': transformer_output
        }
    
    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """提取视觉特征"""
        outputs = self.forward(x)
        return outputs['features']
    
    def get_attention_weights(self, x: torch.Tensor) -> torch.Tensor:
        """获取注意力权重"""
        outputs = self.forward(x)
        return outputs['attention_weights']

class VisionAttention(nn.Module):
    """视觉注意力机制"""
    
    def __init__(self, hidden_dim: int, num_heads: int = 8):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        注意力计算
        
        Args:
            x: 输入特征 [batch_size, seq_len, hidden_dim]
            
        Returns:
            注意力输出和权重
        """
        batch_size, seq_len, _ = x.size()
        
        # 计算Q, K, V
        Q = self.query(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        K = self.key(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        V = self.value(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # 转置以便计算注意力
        Q = Q.transpose(1, 2)  # [batch_size, num_heads, seq_len, head_dim]
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # 计算注意力分数
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        # 应用注意力权重
        attention_output = torch.matmul(attention_weights, V)
        
        # 重塑并投影
        attention_output = attention_output.transpose(1, 2).contiguous()
        attention_output = attention_output.view(batch_size, seq_len, self.hidden_dim)
        output = self.output_projection(attention_output)
        
        return output, attention_weights

class VisionMemory(nn.Module):
    """视觉记忆模块"""
    
    def __init__(self, hidden_dim: int, memory_size: int = 100):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.memory_size = memory_size
        
        # 记忆存储
        self.memory = nn.Parameter(torch.randn(memory_size, hidden_dim))
        
        # 记忆更新门
        self.update_gate = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # 记忆读取门
        self.read_gate = nn.Linear(hidden_dim * 2, hidden_dim)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        记忆操作
        
        Args:
            x: 输入特征 [batch_size, seq_len, hidden_dim]
            
        Returns:
            记忆增强的特征和记忆权重
        """
        batch_size, seq_len, _ = x.size()
        
        # 计算与记忆的相似度
        memory_expanded = self.memory.unsqueeze(0).expand(batch_size, -1, -1)
        similarity = torch.matmul(x, memory_expanded.transpose(-2, -1))
        memory_weights = F.softmax(similarity, dim=-1)
        
        # 读取记忆
        memory_read = torch.matmul(memory_weights, memory_expanded)
        
        # 更新记忆
        update_input = torch.cat([x, memory_read], dim=-1)
        update_signal = torch.sigmoid(self.update_gate(update_input))
        
        # 融合特征
        enhanced_features = x + update_signal * memory_read
        
        return enhanced_features, memory_weights
    
    def update_memory(self, new_features: torch.Tensor):
        """更新记忆"""
        # 简单的记忆更新策略
        with torch.no_grad():
            self.memory.data = 0.9 * self.memory.data + 0.1 * new_features.mean(dim=0) 
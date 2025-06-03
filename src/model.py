#!/usr/bin/env python3
"""
KoBERT 기반 멀티태스크 공손도 예측 모델
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from typing import Dict, List, Optional, Tuple


class MultitaskPolitenessModel(nn.Module):
    """
    KoBERT 기반 멀티태스크 공손도 예측 모델
    
    Architecture:
    - Encoder: KoBERT (768차원)
    - Feature Heads: 7개 분류 헤드 (각각 4클래스)
    - Score Head: 1개 회귀 헤드 (최종 공손도 점수)
    """
    
    def __init__(self, model_name: str = "monologg/kobert", 
                 num_features: int = 7, num_classes_per_feature: int = 4,
                 dropout_rate: float = 0.3, hidden_dim: int = 768):
        super().__init__()
        
        self.num_features = num_features
        self.num_classes_per_feature = num_classes_per_feature
        
        # KoBERT Encoder
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        
        # Dropout layers
        self.dropout = nn.Dropout(dropout_rate)
        
        # Feature classification heads (7개 피처, 각각 4클래스)
        self.feature_heads = nn.ModuleList([
            nn.Linear(hidden_dim, num_classes_per_feature) 
            for _ in range(num_features)
        ])
        
        # Score regression head (최종 공손도 점수)
        self.score_head = nn.Linear(hidden_dim, 1)
        
        # Feature names for interpretability
        self.feature_names = [
            'feat_ending', 'feat_strat_cnt', 'feat_indirect', 
            'feat_command', 'feat_attack', 'feat_power', 'feat_distance'
        ]
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            
        Returns:
            Dict containing feature logits and score predictions
        """
        # BERT encoding
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # [batch_size, hidden_dim]
        
        # Apply dropout
        pooled_output = self.dropout(pooled_output)
        
        # Feature predictions (7개 분류 헤드)
        feature_logits = []
        for i, head in enumerate(self.feature_heads):
            logits = head(pooled_output)  # [batch_size, num_classes_per_feature]
            feature_logits.append(logits)
        
        # Score prediction (회귀)
        score_pred = self.score_head(pooled_output).squeeze(-1)  # [batch_size]
        
        return {
            'feature_logits': feature_logits,  # List of [batch_size, 4]
            'score_pred': score_pred,          # [batch_size]
            'pooled_output': pooled_output     # [batch_size, hidden_dim] for analysis
        }
    
    def get_feature_predictions(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict:
        """피처별 예측값과 확률 반환 (추론용)"""
        self.eval()
        with torch.no_grad():
            outputs = self.forward(input_ids, attention_mask)
            
            feature_probs = []
            feature_preds = []
            
            for logits in outputs['feature_logits']:
                probs = F.softmax(logits, dim=-1)  # [batch_size, 4]
                preds = torch.argmax(logits, dim=-1)  # [batch_size]
                feature_probs.append(probs)
                feature_preds.append(preds)
            
            return {
                'feature_predictions': feature_preds,  # List of [batch_size]
                'feature_probabilities': feature_probs,  # List of [batch_size, 4]
                'score_prediction': outputs['score_pred'],  # [batch_size]
                'feature_names': self.feature_names
            }


class MultitaskLoss(nn.Module):
    """멀티태스크 손실 함수"""
    
    def __init__(self, feature_weights: Optional[Dict[str, torch.Tensor]] = None,
                 score_weight: float = 1.0, feature_weight: float = 1.0):
        super().__init__()
        
        self.feature_weights = feature_weights
        self.score_weight = score_weight
        self.feature_weight = feature_weight
        
        # 손실 함수들
        self.mse_loss = nn.MSELoss(reduction='none')
        self.ce_loss = nn.CrossEntropyLoss(reduction='none')
        
    def forward(self, predictions: Dict, batch: Dict) -> Dict[str, torch.Tensor]:
        """
        멀티태스크 손실 계산
        
        Args:
            predictions: 모델 예측값
            batch: 배치 데이터 (feature_labels, feature_mask, score, score_mask 포함)
            
        Returns:
            손실값들이 담긴 딕셔너리
        """
        device = predictions['score_pred'].device
        
        # 1. Feature Classification Losses (Manual 데이터만)
        feature_losses = []
        total_feature_loss = torch.tensor(0.0, device=device)
        
        feature_labels = batch['feature_labels']  # [batch_size, 7]
        feature_mask = batch['feature_mask']      # [batch_size, 7]
        
        for i, logits in enumerate(predictions['feature_logits']):
            # 해당 피처가 유효한 샘플들만 선택
            valid_mask = feature_mask[:, i]  # [batch_size]
            
            if valid_mask.sum() > 0:
                valid_logits = logits[valid_mask]  # [valid_samples, 4]
                valid_labels = feature_labels[valid_mask, i]  # [valid_samples]
                
                # CrossEntropy Loss
                loss = self.ce_loss(valid_logits, valid_labels)  # [valid_samples]
                
                # 클래스 가중치 적용 (옵션)
                if self.feature_weights is not None:
                    feature_name = f'feat_{i}'  # 실제로는 feature_names[i] 사용
                    if feature_name in self.feature_weights:
                        weights = self.feature_weights[feature_name].to(device)
                        class_weights = weights[valid_labels]
                        loss = loss * class_weights
                
                feature_loss = loss.mean()
                feature_losses.append(feature_loss)
                total_feature_loss += feature_loss
            else:
                feature_losses.append(torch.tensor(0.0, device=device))
        
        # 2. Score Regression Loss (모든 데이터)
        score_pred = predictions['score_pred']  # [batch_size]
        score_target = batch['score']           # [batch_size]
        score_mask = batch['score_mask']        # [batch_size]
        
        if score_mask.sum() > 0:
            valid_score_pred = score_pred[score_mask]
            valid_score_target = score_target[score_mask]
            score_loss = self.mse_loss(valid_score_pred, valid_score_target).mean()
        else:
            score_loss = torch.tensor(0.0, device=device)
        
        # 3. Total Loss
        total_loss = (self.feature_weight * total_feature_loss + 
                     self.score_weight * score_loss)
        
        return {
            'total_loss': total_loss,
            'feature_loss': total_feature_loss,
            'score_loss': score_loss,
            'individual_feature_losses': feature_losses
        }


def create_model_and_loss(model_name: str = "monologg/kobert",
                         class_weights: Optional[Dict] = None,
                         **kwargs) -> Tuple[MultitaskPolitenessModel, MultitaskLoss]:
    """모델과 손실함수 생성"""
    
    model = MultitaskPolitenessModel(model_name=model_name, **kwargs)
    loss_fn = MultitaskLoss(feature_weights=class_weights)
    
    return model, loss_fn 
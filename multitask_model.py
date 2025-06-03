import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, BertTokenizer
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from torch.utils.data import Dataset, DataLoader

class KoPolitenessDataset(Dataset):
    """한국어 공손도 데이터셋"""
    def __init__(self, sentences, feat_labels=None, scores=None, tokenizer=None, max_length=512):
        self.sentences = sentences
        self.feat_labels = feat_labels  # Manual 데이터: [N, 7], TyDiP: None
        self.scores = scores
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 피처 이름 정의
        self.feature_names = [
            "feat_ending", "feat_strat_cnt", "feat_command",
            "feat_attack", "feat_power", "feat_distance", "feat_indirect"
        ]
        
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sentence = str(self.sentences[idx])
        
        # 토크나이징
        encoded = self.tokenizer(
            sentence,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze()
        }
        
        # 피처 라벨 (Manual 데이터만)
        if self.feat_labels is not None:
            feat_labels_tensor = torch.tensor(self.feat_labels[idx], dtype=torch.long)
            feat_mask = torch.ones(7, dtype=torch.bool)  # Manual 데이터는 모든 피처 사용
        else:
            feat_labels_tensor = torch.tensor([-1] * 7, dtype=torch.long)  # TyDiP 데이터
            feat_mask = torch.zeros(7, dtype=torch.bool)  # 피처 마스킹
        
        item['feat_labels'] = feat_labels_tensor
        item['feat_mask'] = feat_mask
        
        # 스코어
        if self.scores is not None:
            item['score'] = torch.tensor(self.scores[idx], dtype=torch.float)
            item['score_mask'] = torch.tensor(True, dtype=torch.bool)
        else:
            item['score'] = torch.tensor(0.0, dtype=torch.float)
            item['score_mask'] = torch.tensor(False, dtype=torch.bool)
            
        return item

class ImprovedMultiTaskModel(nn.Module):
    """개선된 멀티태스크 공손도 예측 모델"""
    def __init__(self, model_name='monologg/kobert', num_classes=4, dropout_rate=0.3):
        super().__init__()
        
        # KoBERT 인코더
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.bert.config.hidden_size  # 768
        
        # 드롭아웃
        self.dropout = nn.Dropout(dropout_rate)
        
        # 피처 간 상호작용 학습을 위한 공통 레이어
        self.feature_interaction = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_size // 2, self.hidden_size // 2)
        )
        
        # 각 피처별 분류 헤드
        self.feature_heads = nn.ModuleDict({
            "feat_ending": nn.Linear(self.hidden_size // 2, num_classes),
            "feat_strat_cnt": nn.Linear(self.hidden_size // 2, 3),  # 0~2
            "feat_command": nn.Linear(self.hidden_size // 2, num_classes),
            "feat_attack": nn.Linear(self.hidden_size // 2, num_classes),
            "feat_power": nn.Linear(self.hidden_size // 2, num_classes),
            "feat_distance": nn.Linear(self.hidden_size // 2, num_classes),
            "feat_indirect": nn.Linear(self.hidden_size // 2, 3)  # 0~2
        })
        
        # 최종 점수 회귀 헤드 (차원 수정)
        # BERT features (768) + feature predictions (4+3+4+4+4+4+3 = 26)
        feature_pred_size = 4 + 3 + 4 + 4 + 4 + 4 + 3
        self.score_head = nn.Sequential(
            nn.Linear(self.hidden_size + feature_pred_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 1)
        )
        
    def forward(self, input_ids, attention_mask):
        # BERT 인코딩
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = bert_output.pooler_output  # [batch_size, hidden_size]
        pooled_output = self.dropout(pooled_output)
        
        # 피처 상호작용 학습
        interaction_features = self.feature_interaction(pooled_output)
        
        # 각 피처별 예측
        feature_logits = {}
        feature_probs = {}
        
        for feat_name, head in self.feature_heads.items():
            logits = head(interaction_features)
            feature_logits[feat_name] = logits
            feature_probs[feat_name] = F.softmax(logits, dim=-1)
        
        # 최종 점수 예측 (BERT features + feature predictions 결합)
        all_probs = torch.cat(list(feature_probs.values()), dim=-1)
        score_input = torch.cat([pooled_output, all_probs], dim=-1)
        score_output = self.score_head(score_input)
        
        return feature_logits, score_output.squeeze(-1)

class FocalLoss(nn.Module):
    """Focal Loss for imbalanced classification"""
    def __init__(self, alpha=1, gamma=2, weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()

def compute_class_weights(labels, num_classes):
    """클래스 가중치 계산"""
    unique_labels = np.unique(labels)
    class_weights = compute_class_weight(
        'balanced', 
        classes=unique_labels, 
        y=labels
    )
    
    # 누락된 클래스에 대해 기본값 설정
    weights = np.ones(num_classes)
    for i, label in enumerate(unique_labels):
        weights[label] = class_weights[i]
    
    return torch.FloatTensor(weights)

def smart_train_val_split(df, test_size=0.2, min_rare_samples=2, random_state=42):
    """희귀 클래스를 보장하는 스마트 분할"""
    feature_cols = [
        "feat_ending", "feat_strat_cnt", "feat_command",
        "feat_attack", "feat_power", "feat_distance", "feat_indirect"
    ]
    
    train_indices = []
    val_indices = []
    
    # 각 피처별로 희귀 클래스 샘플을 먼저 validation에 배정
    for col in feature_cols:
        value_counts = df[col].value_counts()
        
        for class_val, count in value_counts.items():
            class_indices = df[df[col] == class_val].index.tolist()
            
            if count < min_rare_samples * 2:  # 매우 희귀한 클래스
                # 최소 1개는 validation에 배정
                val_sample = min(1, count // 2)
                np.random.seed(random_state)
                val_cls_indices = np.random.choice(class_indices, val_sample, replace=False)
                val_indices.extend(val_cls_indices)
                
                # 나머지는 train에
                train_cls_indices = [idx for idx in class_indices if idx not in val_cls_indices]
                train_indices.extend(train_cls_indices)
    
    # 중복 제거
    val_indices = list(set(val_indices))
    train_indices = list(set(train_indices))
    
    # 아직 배정되지 않은 샘플들에 대해 일반적인 층화 추출
    remaining_indices = [idx for idx in df.index if idx not in train_indices and idx not in val_indices]
    
    if remaining_indices:
        remaining_df = df.loc[remaining_indices]
        # 복합 층화 (score 기반)
        score_bins = pd.cut(remaining_df['score'], bins=5, labels=False)
        
        from sklearn.model_selection import train_test_split
        remain_train, remain_val = train_test_split(
            remaining_indices,
            test_size=test_size,
            stratify=score_bins,
            random_state=random_state
        )
        
        train_indices.extend(remain_train)
        val_indices.extend(remain_val)
    
    return train_indices, val_indices

def compute_multitask_loss(feature_logits, score_pred, feat_labels, score_labels, 
                          feat_mask, score_mask, class_weights_dict, use_focal=True):
    """멀티태스크 손실 계산"""
    total_loss = 0.0
    feature_names = list(feature_logits.keys())
    
    # 피처별 분류 손실
    for i, feat_name in enumerate(feature_names):
        if feat_mask[:, i].any():  # 해당 피처가 마스킹되지 않은 샘플이 있는 경우
            valid_mask = feat_mask[:, i]
            valid_logits = feature_logits[feat_name][valid_mask]
            valid_labels = feat_labels[valid_mask, i]
            
            if len(valid_logits) > 0:
                weights = class_weights_dict.get(feat_name, None)
                
                if use_focal:
                    focal_loss = FocalLoss(alpha=1, gamma=2, weight=weights)
                    feat_loss = focal_loss(valid_logits, valid_labels)
                else:
                    feat_loss = F.cross_entropy(valid_logits, valid_labels, weight=weights)
                
                total_loss += feat_loss
    
    # 스코어 회귀 손실
    if score_mask.any():
        valid_score_pred = score_pred[score_mask]
        valid_score_labels = score_labels[score_mask]
        score_loss = F.mse_loss(valid_score_pred, valid_score_labels)
        total_loss += score_loss
    
    return total_loss 
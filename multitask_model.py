import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, BertTokenizer
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import random
from sklearn.metrics import classification_report, mean_absolute_error

# FEATURE_CONFIGS import
FEATURE_CONFIGS = {
    'feat_ending': 3,      # 0, 1, 2 (원래 3이 2로 병합)
    'feat_strat_cnt': 3,   # 0, 1, 2 (변화 없음)
    'feat_command': 3,     # 0, 1, 2 (원래 3이 2로 병합)
    'feat_attack': 3,      # 0, 1, 2 (원래 3이 2로 병합)
    'feat_power': 3,       # 0, 1, 2 (원래 3이 2로 병합)
    'feat_distance': 3,    # 0, 1, 2 (원래 3이 2로 병합)
    'feat_indirect': 3     # 0, 1, 2 (변화 없음)
}

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
    def __init__(self, model_name='monologg/kobert', dropout_rate=0.3):
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
        
        # 각 피처별 분류 헤드 (업데이트된 클래스 수 사용)
        self.feature_heads = nn.ModuleDict({
            "feat_ending": nn.Linear(self.hidden_size // 2, FEATURE_CONFIGS['feat_ending']),
            "feat_strat_cnt": nn.Linear(self.hidden_size // 2, FEATURE_CONFIGS['feat_strat_cnt']),
            "feat_command": nn.Linear(self.hidden_size // 2, FEATURE_CONFIGS['feat_command']),
            "feat_attack": nn.Linear(self.hidden_size // 2, FEATURE_CONFIGS['feat_attack']),
            "feat_power": nn.Linear(self.hidden_size // 2, FEATURE_CONFIGS['feat_power']),
            "feat_distance": nn.Linear(self.hidden_size // 2, FEATURE_CONFIGS['feat_distance']),
            "feat_indirect": nn.Linear(self.hidden_size // 2, FEATURE_CONFIGS['feat_indirect'])
        })
        
        # 최종 점수 회귀 헤드 (업데이트된 차원 수정)
        # BERT features (768) + feature predictions (3*7 = 21)
        feature_pred_size = sum(FEATURE_CONFIGS.values())  # 모든 피처의 클래스 수 합
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

def compute_class_weights(labels, num_classes, feature_name=None):
    """클래스 가중치 계산 (중요 클래스 부스트 포함)"""
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
    
    # 중요한 희귀 클래스에 추가 부스트 적용 (클래스 3 → 2 병합 반영)
    if feature_name == 'feat_attack':
        # feat_attack class 2 (클래스 3이 병합됨)는 공손도에 매우 중요하므로 강하게 부스트
        if 2 < len(weights):
            weights[2] *= 4.0  # 클래스 3의 중요도를 클래스 2에 반영
            print(f"🚀 Boosted {feature_name} class 2 weight: {weights[2]:.1f}")
            
        # class 1도 부스트 (희귀하므로)
        if 1 < len(weights):
            weights[1] *= 2.5
            print(f"🔧 Boosted {feature_name} class 1 weight: {weights[1]:.1f}")
    
    elif feature_name == 'feat_distance':
        # feat_distance class 2 (클래스 3이 병합됨)도 희귀하고 중요
        if 2 < len(weights):
            weights[2] *= 3.0
            print(f"🔧 Boosted {feature_name} class 2 weight: {weights[2]:.1f}")
    
    elif feature_name == 'feat_command':
        # feat_command class 2 (클래스 3이 병합됨)도 공손도에 중요
        if 2 < len(weights):
            weights[2] *= 3.0
            print(f"🔧 Boosted {feature_name} class 2 weight: {weights[2]:.1f}")
        
        # class 1도 부스트
        if 1 < len(weights):
            weights[1] *= 1.5
            print(f"🔧 Boosted {feature_name} class 1 weight: {weights[1]:.1f}")
    
    elif feature_name == 'feat_ending':
        # feat_ending class 2 (클래스 3이 병합됨)도 희귀하므로 부스트
        if 2 < len(weights):
            weights[2] *= 2.5
            print(f"🔧 Boosted {feature_name} class 2 weight: {weights[2]:.1f}")
    
    elif feature_name == 'feat_power':
        # feat_power class 2 (클래스 3이 병합됨)도 부스트
        if 2 < len(weights):
            weights[2] *= 2.0
            print(f"🔧 Boosted {feature_name} class 2 weight: {weights[2]:.1f}")
    
    return torch.FloatTensor(weights)

def smart_train_val_split(df, test_size=0.2, min_rare_samples=2, random_state=42):
    """희귀 클래스 강제배치를 통한 스마트 분할"""
    np.random.seed(random_state)
    
    feature_cols = [
        "feat_ending", "feat_strat_cnt", "feat_command",
        "feat_attack", "feat_power", "feat_distance", "feat_indirect"
    ]
    
    train_indices = []
    val_indices = []
    forced_allocation = {}  # 강제 배치된 샘플 추적
    
    print("🔧 Forced allocation for rare classes:")
    
    # 1단계: 극희귀 클래스 강제 배치 (< 10개)
    for col in feature_cols:
        value_counts = df[col].value_counts()
        
        for class_val, count in value_counts.items():
            class_indices = df[df[col] == class_val].index.tolist()
            
            if count < 10:  # 극희귀 클래스
                # 최소 2개는 train, 1개는 val에 강제 배치
                min_train = max(2, count - 2)
                min_val = min(1, count - min_train)
                
                np.random.shuffle(class_indices)
                
                forced_train = class_indices[:min_train]
                forced_val = class_indices[min_train:min_train + min_val]
                
                train_indices.extend(forced_train)
                val_indices.extend(forced_val)
                
                # 추적 정보 저장
                forced_allocation[f"{col}_{class_val}"] = {
                    'total': count,
                    'train': len(forced_train),
                    'val': len(forced_val)
                }
                
                print(f"  {col} class {class_val}: {count} samples → train: {len(forced_train)}, val: {len(forced_val)}")
    
    # 중복 제거
    val_indices = list(set(val_indices))
    train_indices = list(set(train_indices))
    
    print(f"🎯 Forced allocation completed: {len(train_indices)} train, {len(val_indices)} val")
    
    # 2단계: 남은 샘플들 층화 추출
    remaining_indices = [idx for idx in df.index if idx not in train_indices and idx not in val_indices]
    
    if remaining_indices:
        remaining_df = df.loc[remaining_indices]
        
        # 다중 피처 기반 복합 층화
        # 주요 피처들의 조합으로 층화 타겟 생성
        primary_features = ['feat_ending', 'feat_attack', 'feat_power']
        stratify_labels = []
        
        for idx in remaining_indices:
            row = df.loc[idx]
            # 주요 피처들의 조합으로 층화 키 생성
            key = f"{row['feat_ending']}-{row['feat_attack']}-{row['feat_power']}"
            stratify_labels.append(key)
        
        # 너무 적은 그룹들은 통합
        from collections import Counter
        label_counts = Counter(stratify_labels)
        
        # 3개 미만인 그룹들을 'other'로 통합
        final_labels = []
        for label in stratify_labels:
            if label_counts[label] >= 3:
                final_labels.append(label)
            else:
                final_labels.append('other')
        
        try:
            from sklearn.model_selection import train_test_split
            remain_train, remain_val = train_test_split(
                remaining_indices,
                test_size=test_size,
                stratify=final_labels,
                random_state=random_state
            )
            
            train_indices.extend(remain_train)
            val_indices.extend(remain_val)
            
        except ValueError as e:
            print(f"⚠️  Stratification failed: {e}")
            # 층화 실패시 단순 랜덤 분할
            np.random.shuffle(remaining_indices)
            split_point = int(len(remaining_indices) * (1 - test_size))
            train_indices.extend(remaining_indices[:split_point])
            val_indices.extend(remaining_indices[split_point:])
    
    print(f"📊 Final split: {len(train_indices)} train, {len(val_indices)} val")
    return train_indices, val_indices

def forced_kfold_split(df, n_folds=3, random_state=42):
    """희귀 클래스 강제배치 K-fold"""
    np.random.seed(random_state)
    
    feature_cols = [
        "feat_ending", "feat_strat_cnt", "feat_command",
        "feat_attack", "feat_power", "feat_distance", "feat_indirect"
    ]
    
    # 각 fold 초기화
    folds = [[] for _ in range(n_folds)]
    allocated_indices = set()
    
    print(f"🔧 Creating {n_folds}-fold splits with forced rare class allocation:")
    
    # 1단계: 희귀 클래스 강제 배치
    for col in feature_cols:
        value_counts = df[col].value_counts()
        
        for class_val, count in value_counts.items():
            if count < 15:  # 희귀 클래스 기준
                class_indices = df[df[col] == class_val].index.tolist()
                class_indices = [idx for idx in class_indices if idx not in allocated_indices]
                
                if len(class_indices) == 0:
                    continue
                
                np.random.shuffle(class_indices)
                
                # 각 fold에 최소 1개씩 배치
                for fold_idx in range(min(n_folds, len(class_indices))):
                    if fold_idx < len(class_indices):
                        folds[fold_idx].append(class_indices[fold_idx])
                        allocated_indices.add(class_indices[fold_idx])
                
                # 남은 샘플들을 fold에 순환 배치
                remaining = class_indices[n_folds:]
                for i, idx in enumerate(remaining):
                    fold_idx = i % n_folds
                    folds[fold_idx].append(idx)
                    allocated_indices.add(idx)
                
                print(f"  {col} class {class_val}: {count} samples distributed across folds")
    
    # 2단계: 남은 샘플들 균등 배치
    remaining_indices = [idx for idx in df.index if idx not in allocated_indices]
    np.random.shuffle(remaining_indices)
    
    # 가장 작은 fold부터 순환하며 배치
    for i, idx in enumerate(remaining_indices):
        # fold 크기 기준으로 정렬해서 작은 fold부터 채움
        fold_sizes = [(len(fold), fold_idx) for fold_idx, fold in enumerate(folds)]
        fold_sizes.sort()
        smallest_fold_idx = fold_sizes[0][1]
        folds[smallest_fold_idx].append(idx)
    
    # 결과 출력
    print("📊 Final fold distribution:")
    for i, fold in enumerate(folds):
        print(f"  Fold {i+1}: {len(fold)} samples")
    
    return folds

def compute_multitask_loss(feature_logits, score_pred, feat_labels, score_labels, 
                          feat_mask, score_mask, class_weights_dict, use_focal=True, alpha=2, gamma=3):
    """멀티태스크 손실 계산 (피처 중요도 가중치 포함)"""
    total_loss = 0.0
    feature_names = list(feature_logits.keys())
    
    # 피처별 중요도 가중치 (공손도에 미치는 영향 기준)
    feature_importance = {
        'feat_attack': 3.5,    # 공격성이 가장 중요 (증가)
        'feat_command': 2.0,   # 명령성도 매우 중요 (증가)
        'feat_ending': 3.0,    # 어미도 중요 (증가)
        'feat_distance': 2.5,  # 사회적 거리 (증가)
        'feat_power': 2.0,     # 권력거리 (증가)
        'feat_indirect': 1.5,  # 간접성 (증가)
        'feat_strat_cnt': 1.0  # 전략적 표현 (기본)
    }
    
    # 피처별 분류 손실
    for i, feat_name in enumerate(feature_names):
        if feat_mask[:, i].any():  # 해당 피처가 마스킹되지 않은 샘플이 있는 경우
            valid_mask = feat_mask[:, i]
            valid_logits = feature_logits[feat_name][valid_mask]
            valid_labels = feat_labels[valid_mask, i]
            
            if len(valid_logits) > 0:
                weights = class_weights_dict.get(feat_name, None)
                
                if use_focal:
                    focal_loss = FocalLoss(alpha=alpha, gamma=gamma, weight=weights)
                    feat_loss = focal_loss(valid_logits, valid_labels)
                else:
                    feat_loss = F.cross_entropy(valid_logits, valid_labels, weight=weights)
                
                # 피처 중요도 가중치 적용
                importance_weight = feature_importance.get(feat_name, 1.0)
                weighted_feat_loss = feat_loss * importance_weight
                
                total_loss += weighted_feat_loss
                
                # 디버깅용 출력 (첫 번째 배치에서만)
                if torch.rand(1).item() < 0.01:  # 1% 확률로 출력
                    print(f"  {feat_name}: loss={feat_loss:.4f}, importance={importance_weight}, weighted={weighted_feat_loss:.4f}")
    
    # 스코어 회귀 손실 (중요도 높게)
    if score_mask.any():
        valid_score_pred = score_pred[score_mask]
        valid_score_labels = score_labels[score_mask]
        score_loss = F.mse_loss(valid_score_pred, valid_score_labels)
        
        # 스코어 손실은 최종 목표이므로 높은 가중치
        weighted_score_loss = score_loss * 2.0
        total_loss += weighted_score_loss
    
    return total_loss
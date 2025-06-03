#!/usr/bin/env python3
"""
데이터 처리 유틸리티
한국어 공손도 예측 시스템을 위한 데이터 로딩, 전처리, 분할 기능
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight
from transformers import AutoTokenizer
from collections import Counter
import torch
from torch.utils.data import Dataset


class PolitenessDataProcessor:
    """공손도 데이터 처리 클래스"""
    
    def __init__(self, manual_data_path: str, tydip_data_path: str, tokenizer_name: str = "monologg/kobert"):
        self.manual_data_path = manual_data_path
        self.tydip_data_path = tydip_data_path
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # 피처 컬럼 정의
        self.feature_cols = [
            'feat_ending', 'feat_strat_cnt', 'feat_indirect', 
            'feat_command', 'feat_attack', 'feat_power', 'feat_distance'
        ]
        
        # 클래스 개수 (각 피처는 0-3점, 총 4개 클래스)
        self.num_classes_per_feature = 4
        
    def load_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """데이터 로딩"""
        print("📊 데이터 로딩 중...")
        
        manual_df = pd.read_csv(self.manual_data_path)
        tydip_df = pd.read_csv(self.tydip_data_path)
        
        print(f"   Manual 데이터: {len(manual_df)}개 문장")
        print(f"   TyDiP 데이터: {len(tydip_df)}개 문장")
        
        return manual_df, tydip_df
    
    def analyze_class_distribution(self, manual_df: pd.DataFrame) -> Dict:
        """클래스 분포 분석"""
        print("📈 클래스 분포 분석...")
        
        distribution = {}
        for feature in self.feature_cols:
            dist = manual_df[feature].value_counts().sort_index()
            distribution[feature] = dist
            print(f"   {feature}: {dict(dist)}")
        
        # 최종 점수 분포 (3구간으로 분할)
        score_bins = pd.cut(manual_df['score'], bins=3, labels=['낮음', '보통', '높음'])
        score_dist = score_bins.value_counts()
        distribution['score_bins'] = score_dist
        print(f"   점수 분포: {dict(score_dist)}")
        
        return distribution
    
    def compute_class_weights(self, manual_df: pd.DataFrame) -> Dict:
        """클래스 가중치 계산"""
        print("⚖️  클래스 가중치 계산...")
        
        class_weights = {}
        
        # 각 피처별 클래스 가중치
        for feature in self.feature_cols:
            feature_values = manual_df[feature].values
            unique_classes = np.unique(feature_values)
            weights = compute_class_weight(
                'balanced', 
                classes=unique_classes, 
                y=feature_values
            )
            class_weights[feature] = torch.FloatTensor(weights)
            print(f"   {feature} 가중치: {weights}")
        
        return class_weights
    
    def split_data_stratified(self, manual_df: pd.DataFrame, tydip_df: pd.DataFrame, 
                            k_folds: int = 5, val_ratio: float = 0.2, test_ratio: float = 0.1) -> Dict:
        """층화 분할 + K-fold 준비"""
        print("✂️  데이터 분할 중...")
        
        # 1. Manual 데이터 분할 (피처별 층화)
        # 피처별 층화를 위해 복합 라벨 생성
        manual_df['stratify_label'] = (
            manual_df['feat_attack'].astype(str) + '_' +
            manual_df['feat_command'].astype(str) + '_' +
            manual_df['feat_power'].astype(str)
        )
        
        # 희소 클래스 처리: 3점 피처들 강제 배치
        rare_mask = (
            (manual_df['feat_attack'] == 3) | 
            (manual_df['feat_command'] == 3) | 
            (manual_df['feat_power'] == 3)
        )
        
        rare_samples = manual_df[rare_mask].copy()
        common_samples = manual_df[~rare_mask].copy()
        
        print(f"   희소 클래스 샘플: {len(rare_samples)}개")
        print(f"   일반 샘플: {len(common_samples)}개")
        
        # 희소 클래스 고정 분할
        np.random.seed(42)
        rare_indices = np.random.permutation(len(rare_samples))
        rare_train_size = max(1, int(len(rare_samples) * (1 - val_ratio - test_ratio)))
        rare_val_size = max(1, int(len(rare_samples) * val_ratio))
        
        rare_train = rare_samples.iloc[rare_indices[:rare_train_size]].copy()
        rare_val = rare_samples.iloc[rare_indices[rare_train_size:rare_train_size + rare_val_size]].copy()
        rare_test = rare_samples.iloc[rare_indices[rare_train_size + rare_val_size:]].copy()
        
        # 일반 샘플 층화 분할
        from sklearn.model_selection import train_test_split
        
        # 층화 라벨이 너무 세분화된 경우 score 기준으로 단순화
        stratify_col = common_samples['stratify_label']
        if len(np.unique(stratify_col)) > len(common_samples) // 3:
            # score를 3구간으로 나누어 층화
            stratify_col = pd.cut(common_samples['score'], bins=3, labels=[0, 1, 2])
        
        common_train_val, common_test = train_test_split(
            common_samples, test_size=test_ratio, stratify=stratify_col, random_state=42
        )
        
        stratify_col_train_val = common_train_val['stratify_label']
        if len(np.unique(stratify_col_train_val)) > len(common_train_val) // 3:
            stratify_col_train_val = pd.cut(common_train_val['score'], bins=3, labels=[0, 1, 2])
        
        common_train, common_val = train_test_split(
            common_train_val, test_size=val_ratio/(1-test_ratio), 
            stratify=stratify_col_train_val, random_state=43
        )
        
        # Manual 데이터 최종 결합
        manual_train = pd.concat([rare_train, common_train], ignore_index=True)
        manual_val = pd.concat([rare_val, common_val], ignore_index=True)
        manual_test = pd.concat([rare_test, common_test], ignore_index=True)
        
        # 2. TyDiP 데이터 분할 (점수 기준 층화)
        tydip_score_bins = pd.cut(tydip_df['score'], bins=3, labels=[0, 1, 2])
        
        tydip_train_val, tydip_test = train_test_split(
            tydip_df, test_size=test_ratio, stratify=tydip_score_bins, random_state=42
        )
        
        tydip_score_bins_train_val = pd.cut(tydip_train_val['score'], bins=3, labels=[0, 1, 2])
        tydip_train, tydip_val = train_test_split(
            tydip_train_val, test_size=val_ratio/(1-test_ratio), 
            stratify=tydip_score_bins_train_val, random_state=43
        )
        
        # 3. K-fold 준비 (Train 데이터에서)
        train_combined = pd.concat([manual_train, tydip_train], ignore_index=True)
        train_combined['is_manual'] = [True] * len(manual_train) + [False] * len(tydip_train)
        
        # K-fold 분할용 층화 라벨
        kfold_stratify = []
        for idx, row in train_combined.iterrows():
            if row['is_manual']:
                # Manual: score 기준 3구간
                score_bin = int(pd.cut([row['score']], bins=3, labels=[0, 1, 2])[0])
                kfold_stratify.append(f"manual_{score_bin}")
            else:
                # TyDiP: score 기준 3구간
                score_bin = int(pd.cut([row['score']], bins=3, labels=[0, 1, 2])[0])
                kfold_stratify.append(f"tydip_{score_bin}")
        
        kfold = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
        
        splits_data = {
            'manual_train': manual_train,
            'manual_val': manual_val,
            'manual_test': manual_test,
            'tydip_train': tydip_train,
            'tydip_val': tydip_val,
            'tydip_test': tydip_test,
            'train_combined': train_combined,
            'kfold': kfold,
            'kfold_stratify': kfold_stratify
        }
        
        print(f"   Manual - Train: {len(manual_train)}, Val: {len(manual_val)}, Test: {len(manual_test)}")
        print(f"   TyDiP - Train: {len(tydip_train)}, Val: {len(tydip_val)}, Test: {len(tydip_test)}")
        print(f"   K-fold 설정: {k_folds}개 fold")
        
        return splits_data


class PolitenessDataset(Dataset):
    """공손도 예측을 위한 PyTorch Dataset"""
    
    def __init__(self, texts: List[str], feature_labels: Optional[List[List[int]]] = None,
                 scores: List[float] = None, tokenizer = None, max_length: int = 256,
                 is_manual: List[bool] = None):
        
        self.texts = texts
        self.feature_labels = feature_labels  # Manual 데이터의 7개 피처
        self.scores = scores
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_manual = is_manual if is_manual is not None else [feature_labels is not None] * len(texts)
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        
        # 토크나이징
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        item = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'is_manual': torch.tensor(self.is_manual[idx], dtype=torch.bool)
        }
        
        # 피처 라벨 (Manual 데이터만)
        if self.feature_labels is not None and self.is_manual[idx]:
            item['feature_labels'] = torch.tensor(self.feature_labels[idx], dtype=torch.long)
            item['feature_mask'] = torch.ones(7, dtype=torch.bool)  # 7개 피처 모두 유효
        else:
            item['feature_labels'] = torch.full((7,), -1, dtype=torch.long)  # 마스킹용 -1
            item['feature_mask'] = torch.zeros(7, dtype=torch.bool)  # 피처 손실 계산 제외
        
        # 점수 라벨
        if self.scores is not None:
            item['score'] = torch.tensor(self.scores[idx], dtype=torch.float)
            item['score_mask'] = torch.tensor(True, dtype=torch.bool)
        else:
            item['score'] = torch.tensor(0.0, dtype=torch.float)
            item['score_mask'] = torch.tensor(False, dtype=torch.bool)
        
        return item


def create_datasets(splits_data: Dict, tokenizer, max_length: int = 256) -> Dict:
    """Dataset 객체들 생성"""
    
    def df_to_dataset(manual_df, tydip_df, split_name):
        # Manual 데이터
        manual_texts = manual_df['sentence'].tolist()
        manual_features = manual_df[['feat_ending', 'feat_strat_cnt', 'feat_indirect', 
                                   'feat_command', 'feat_attack', 'feat_power', 'feat_distance']].values.tolist()
        manual_scores = manual_df['score'].tolist()
        manual_is_manual = [True] * len(manual_df)
        
        # TyDiP 데이터  
        tydip_texts = tydip_df['sentence'].tolist()
        tydip_features = None
        tydip_scores = tydip_df['score'].tolist()
        tydip_is_manual = [False] * len(tydip_df)
        
        # 결합
        all_texts = manual_texts + tydip_texts
        all_features = manual_features + [None] * len(tydip_texts)
        all_scores = manual_scores + tydip_scores
        all_is_manual = manual_is_manual + tydip_is_manual
        
        return PolitenessDataset(
            texts=all_texts,
            feature_labels=manual_features if len(manual_features) > 0 else None,
            scores=all_scores,
            tokenizer=tokenizer,
            max_length=max_length,
            is_manual=all_is_manual
        )
    
    datasets = {
        'train': df_to_dataset(splits_data['manual_train'], splits_data['tydip_train'], 'train'),
        'val': df_to_dataset(splits_data['manual_val'], splits_data['tydip_val'], 'val'),
        'test': df_to_dataset(splits_data['manual_test'], splits_data['tydip_test'], 'test')
    }
    
    return datasets 
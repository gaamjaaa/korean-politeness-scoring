#!/usr/bin/env python3
"""
K-fold 교차검증 트레이너
"""

import torch
from torch.utils.data import DataLoader, Subset
from typing import Dict, List, Tuple
import numpy as np
from .trainer import MultitaskTrainer
from .data_utils import PolitenessDataset
import pandas as pd
from tqdm import tqdm
import json
import os


class KFoldMultitaskTrainer:
    """K-fold 교차검증을 지원하는 멀티태스크 트레이너"""
    
    def __init__(self, splits_data: Dict, datasets: Dict, model_factory, loss_factory, 
                 k_folds: int = 5, batch_size: int = 16, **trainer_kwargs):
        
        self.splits_data = splits_data
        self.datasets = datasets
        self.model_factory = model_factory  # 모델 생성 함수
        self.loss_factory = loss_factory    # 손실함수 생성 함수
        self.k_folds = k_folds
        self.batch_size = batch_size
        self.trainer_kwargs = trainer_kwargs
        
        self.fold_results = []
        
    def run_k_fold_training(self) -> Dict:
        """K-fold 교차검증 실행"""
        print(f"🔄 K-fold 교차검증 시작 (K={self.k_folds})")
        
        # K-fold 분할 정보 가져오기
        train_combined = self.splits_data['train_combined']
        kfold = self.splits_data['kfold']
        kfold_stratify = self.splits_data['kfold_stratify']
        
        fold_metrics = []
        
        for fold, (train_idx, val_idx) in enumerate(kfold.split(train_combined, kfold_stratify)):
            print(f"\n📁 Fold {fold + 1}/{self.k_folds}")
            
            # 이 fold의 train/val 데이터 생성
            fold_train_data = train_combined.iloc[train_idx].copy()
            fold_val_data = train_combined.iloc[val_idx].copy()
            
            # Manual과 TyDiP 데이터 분리
            fold_manual_train = fold_train_data[fold_train_data['is_manual'] == True].copy()
            fold_tydip_train = fold_train_data[fold_train_data['is_manual'] == False].copy()
            fold_manual_val = fold_val_data[fold_val_data['is_manual'] == True].copy()
            fold_tydip_val = fold_val_data[fold_val_data['is_manual'] == False].copy()
            
            print(f"   Train: Manual {len(fold_manual_train)}, TyDiP {len(fold_tydip_train)}")
            print(f"   Val: Manual {len(fold_manual_val)}, TyDiP {len(fold_tydip_val)}")
            
            # Dataset 생성
            fold_train_dataset = self._create_fold_dataset(fold_manual_train, fold_tydip_train)
            fold_val_dataset = self._create_fold_dataset(fold_manual_val, fold_tydip_val)
            
            # DataLoader 생성
            fold_train_loader = DataLoader(
                fold_train_dataset, batch_size=self.batch_size, 
                shuffle=True, collate_fn=self._collate_fn
            )
            fold_val_loader = DataLoader(
                fold_val_dataset, batch_size=self.batch_size, 
                shuffle=False, collate_fn=self._collate_fn
            )
            
            # 모델과 손실함수 생성
            model = self.model_factory()
            loss_fn = self.loss_factory()
            
            # 트레이너 생성
            trainer = MultitaskTrainer(
                model=model,
                loss_fn=loss_fn,
                train_dataloader=fold_train_loader,
                val_dataloader=fold_val_loader,
                save_dir=os.path.join(self.trainer_kwargs.get('save_dir', './checkpoints'), f'fold_{fold+1}'),
                **{k: v for k, v in self.trainer_kwargs.items() if k != 'save_dir'}
            )
            
            # 학습
            history = trainer.train(num_epochs=self.trainer_kwargs.get('num_epochs', 5))
            
            # 최종 평가
            final_metrics = trainer.evaluate(fold_val_loader)
            final_metrics['fold'] = fold + 1
            final_metrics['best_model_path'] = trainer.best_model_path
            
            fold_metrics.append(final_metrics)
            self.fold_results.append({
                'fold': fold + 1,
                'metrics': final_metrics,
                'history': history,
                'best_model_path': trainer.best_model_path
            })
            
            print(f"   ✅ Fold {fold + 1} 완료 - Avg F1: {final_metrics.get('avg_feature_f1', 0):.4f}")
        
        # 전체 결과 집계
        summary = self._summarize_fold_results(fold_metrics)
        
        # 결과 저장
        self._save_k_fold_results(summary)
        
        return {
            'fold_results': self.fold_results,
            'summary': summary
        }
    
    def _create_fold_dataset(self, manual_df: pd.DataFrame, tydip_df: pd.DataFrame) -> PolitenessDataset:
        """Fold별 Dataset 생성"""
        from .data_utils import create_datasets
        
        # Manual 데이터
        if len(manual_df) > 0:
            manual_texts = manual_df['sentence'].tolist()
            manual_features = manual_df[['feat_ending', 'feat_strat_cnt', 'feat_indirect', 
                                       'feat_command', 'feat_attack', 'feat_power', 'feat_distance']].values.tolist()
            manual_scores = manual_df['score'].tolist()
            manual_is_manual = [True] * len(manual_df)
        else:
            manual_texts, manual_features, manual_scores, manual_is_manual = [], [], [], []
        
        # TyDiP 데이터
        if len(tydip_df) > 0:
            tydip_texts = tydip_df['sentence'].tolist()
            tydip_scores = tydip_df['score'].tolist()
            tydip_is_manual = [False] * len(tydip_df)
        else:
            tydip_texts, tydip_scores, tydip_is_manual = [], [], []
        
        # 결합
        all_texts = manual_texts + tydip_texts
        all_scores = manual_scores + tydip_scores
        all_is_manual = manual_is_manual + tydip_is_manual
        
        # 토크나이저 가져오기 (첫 번째 dataset에서)
        tokenizer = self.datasets['train'].tokenizer
        
        return PolitenessDataset(
            texts=all_texts,
            feature_labels=manual_features if len(manual_features) > 0 else None,
            scores=all_scores,
            tokenizer=tokenizer,
            max_length=256,
            is_manual=all_is_manual
        )
    
    def _collate_fn(self, batch):
        """배치 생성을 위한 collate function"""
        # 각 키별로 리스트 수집
        keys = batch[0].keys()
        collated = {}
        
        for key in keys:
            if key in ['input_ids', 'attention_mask', 'feature_labels', 'feature_mask']:
                collated[key] = torch.stack([item[key] for item in batch])
            elif key in ['score', 'is_manual', 'score_mask']:
                collated[key] = torch.stack([item[key] for item in batch])
            else:
                collated[key] = [item[key] for item in batch]
        
        return collated
    
    def _summarize_fold_results(self, fold_metrics: List[Dict]) -> Dict:
        """Fold 결과 요약"""
        # 수치형 메트릭들만 집계
        numeric_metrics = {}
        for key in fold_metrics[0].keys():
            if key not in ['fold', 'best_model_path'] and isinstance(fold_metrics[0][key], (int, float)):
                values = [m[key] for m in fold_metrics if key in m and not np.isnan(m[key])]
                if values:
                    numeric_metrics[key] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'values': values
                    }
        
        summary = {
            'k_folds': self.k_folds,
            'metrics': numeric_metrics,
            'best_fold': None,
            'best_score': float('-inf')
        }
        
        # 최고 성능 fold 찾기 (avg_feature_f1 기준)
        if 'avg_feature_f1' in numeric_metrics:
            best_f1_idx = np.argmax(numeric_metrics['avg_feature_f1']['values'])
            summary['best_fold'] = best_f1_idx + 1
            summary['best_score'] = numeric_metrics['avg_feature_f1']['values'][best_f1_idx]
        
        return summary
    
    def _save_k_fold_results(self, summary: Dict):
        """K-fold 결과 저장"""
        save_dir = self.trainer_kwargs.get('save_dir', './checkpoints')
        os.makedirs(save_dir, exist_ok=True)
        
        # 요약 결과 저장
        summary_path = os.path.join(save_dir, 'k_fold_summary.json')
        with open(summary_path, 'w', encoding='utf-8') as f:
            # numpy 객체들을 직렬화 가능하게 변환
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.int64, np.int32)):
                    return int(obj)
                elif isinstance(obj, (np.float64, np.float32)):
                    return float(obj)
                return obj
            
            def recursive_convert(obj):
                if isinstance(obj, dict):
                    return {k: recursive_convert(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [recursive_convert(item) for item in obj]
                else:
                    return convert_numpy(obj)
            
            json.dump(recursive_convert(summary), f, ensure_ascii=False, indent=2)
        
        # 상세 결과 저장
        details_path = os.path.join(save_dir, 'k_fold_details.json')
        with open(details_path, 'w', encoding='utf-8') as f:
            json.dump(recursive_convert(self.fold_results), f, ensure_ascii=False, indent=2)
        
        print(f"\n📋 K-fold 결과 저장됨:")
        print(f"   요약: {summary_path}")
        print(f"   상세: {details_path}")


def run_k_fold_cv(splits_data: Dict, datasets: Dict, model_factory, loss_factory, 
                  k_folds: int = 5, **kwargs) -> Dict:
    """K-fold 교차검증 실행 헬퍼 함수"""
    
    trainer = KFoldMultitaskTrainer(
        splits_data=splits_data,
        datasets=datasets,
        model_factory=model_factory,
        loss_factory=loss_factory,
        k_folds=k_folds,
        **kwargs
    )
    
    return trainer.run_k_fold_training() 
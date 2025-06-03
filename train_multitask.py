import os
import json
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from multitask_model import (
    KoPolitenessDataset, ImprovedMultiTaskModel, compute_multitask_loss,
    compute_class_weights, smart_train_val_split
)

class MultitaskTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🚀 Using device: {self.device}")
        
        # 토크나이저 로드
        self.tokenizer = BertTokenizer.from_pretrained(config['model_name'])
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.unk_token
        
        # 결과 저장 디렉토리
        os.makedirs(config['save_dir'], exist_ok=True)
        
    def load_data(self):
        """데이터 로드 및 전처리"""
        print("📁 Loading data...")
        
        # Manual 데이터 로드
        manual_df = pd.read_csv(self.config['manual_data_path'])
        print(f"Manual data: {len(manual_df)} samples")
        
        # TyDiP 데이터 로드
        tydip_df = pd.read_csv(self.config['tydip_data_path'])
        print(f"TyDiP data: {len(tydip_df)} samples")
        
        # 스코어 정규화 (선택적)
        if self.config.get('normalize_scores', True):
            manual_df['score_normalized'] = (manual_df['score'] + 3.0) / 6.0
            tydip_df['score_normalized'] = (tydip_df['score'] + 3.0) / 6.0
            score_col = 'score_normalized'
        else:
            score_col = 'score'
        
        self.manual_df = manual_df
        self.tydip_df = tydip_df
        self.score_col = score_col
        
        # 피처 분포 출력
        self.print_feature_distribution()
        
    def print_feature_distribution(self):
        """피처별 클래스 분포 출력"""
        print("\n📊 Feature distribution analysis:")
        feature_cols = [
            "feat_ending", "feat_strat_cnt", "feat_command",
            "feat_attack", "feat_power", "feat_distance", "feat_indirect"
        ]
        
        for col in feature_cols:
            counts = self.manual_df[col].value_counts().sort_index()
            print(f"{col}: {dict(counts)}")
            
            # 희귀 클래스 경고
            rare_classes = counts[counts < 10].index.tolist()
            if rare_classes:
                print(f"Rare classes (< 10 samples): {rare_classes}")
        print()
    
    def compute_class_weights_all_features(self, train_df):
        """모든 피처에 대한 클래스 가중치 계산"""
        feature_cols = [
            "feat_ending", "feat_strat_cnt", "feat_command",
            "feat_attack", "feat_power", "feat_distance", "feat_indirect"
        ]
        
        class_weights_dict = {}
        
        for col in feature_cols:
            labels = train_df[col].values
            if col in ["feat_strat_cnt", "feat_indirect"]:
                num_classes = 3  # 0~2
            else:
                num_classes = 4  # 0~3
            
            weights = compute_class_weights(labels, num_classes)
            class_weights_dict[col] = weights.to(self.device)
            
            print(f"Class weights for {col}: {weights.numpy()}")
        
        return class_weights_dict
    
    def create_dataloaders(self, train_indices, val_indices, manual_df, tydip_df):
        """데이터로더 생성"""
        # Manual train/val split
        manual_train = manual_df.iloc[train_indices].copy()
        manual_val = manual_df.iloc[val_indices].copy()
        
        # TyDiP train/val split (일반적인 split)
        tydip_train_size = int(len(tydip_df) * 0.8)
        tydip_train = tydip_df.iloc[:tydip_train_size].copy()
        tydip_val = tydip_df.iloc[tydip_train_size:].copy()
        
        # 피처 라벨 추출 (Manual만)
        feature_cols = [
            "feat_ending", "feat_strat_cnt", "feat_command",
            "feat_attack", "feat_power", "feat_distance", "feat_indirect"
        ]
        
        manual_train_features = manual_train[feature_cols].values
        manual_val_features = manual_val[feature_cols].values
        
        # 데이터셋 생성
        train_manual_dataset = KoPolitenessDataset(
            sentences=manual_train['sentence'].tolist(),
            feat_labels=manual_train_features,
            scores=manual_train[self.score_col].tolist(),
            tokenizer=self.tokenizer,
            max_length=self.config['max_length']
        )
        
        train_tydip_dataset = KoPolitenessDataset(
            sentences=tydip_train['sentence'].tolist(),
            feat_labels=None,  # TyDiP는 피처 라벨 없음
            scores=tydip_train[self.score_col].tolist(),
            tokenizer=self.tokenizer,
            max_length=self.config['max_length']
        )
        
        val_manual_dataset = KoPolitenessDataset(
            sentences=manual_val['sentence'].tolist(),
            feat_labels=manual_val_features,
            scores=manual_val[self.score_col].tolist(),
            tokenizer=self.tokenizer,
            max_length=self.config['max_length']
        )
        
        val_tydip_dataset = KoPolitenessDataset(
            sentences=tydip_val['sentence'].tolist(),
            feat_labels=None,
            scores=tydip_val[self.score_col].tolist(),
            tokenizer=self.tokenizer,
            max_length=self.config['max_length']
        )
        
        # 데이터로더 생성
        train_manual_loader = DataLoader(
            train_manual_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=True,
            num_workers=0
        )
        
        train_tydip_loader = DataLoader(
            train_tydip_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=True,
            num_workers=0
        )
        
        val_manual_loader = DataLoader(
            val_manual_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False,
            num_workers=0
        )
        
        val_tydip_loader = DataLoader(
            val_tydip_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False,
            num_workers=0
        )
        
        return {
            'train_manual': train_manual_loader,
            'train_tydip': train_tydip_loader,
            'val_manual': val_manual_loader,
            'val_tydip': val_tydip_loader
        }, manual_train
    
    def train_epoch(self, model, train_loaders, optimizer, scheduler, class_weights_dict):
        """한 에포크 훈련"""
        model.train()
        total_loss = 0.0
        num_batches = 0
        
        # Manual 데이터로 훈련
        for batch in tqdm(train_loaders['train_manual'], desc="Training Manual"):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            feat_labels = batch['feat_labels'].to(self.device)
            feat_mask = batch['feat_mask'].to(self.device)
            score_labels = batch['score'].to(self.device)
            score_mask = batch['score_mask'].to(self.device)
            
            # Forward pass
            feature_logits, score_pred = model(input_ids, attention_mask)
            
            # Loss 계산
            loss = compute_multitask_loss(
                feature_logits, score_pred, feat_labels, score_labels,
                feat_mask, score_mask, class_weights_dict, 
                use_focal=self.config.get('use_focal_loss', True)
            )
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        # TyDiP 데이터로 훈련 (스코어만)
        for batch in tqdm(train_loaders['train_tydip'], desc="Training TyDiP"):
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            feat_labels = batch['feat_labels'].to(self.device)
            feat_mask = batch['feat_mask'].to(self.device)
            score_labels = batch['score'].to(self.device)
            score_mask = batch['score_mask'].to(self.device)
            
            # Forward pass
            feature_logits, score_pred = model(input_ids, attention_mask)
            
            # Loss 계산 (스코어만)
            loss = compute_multitask_loss(
                feature_logits, score_pred, feat_labels, score_labels,
                feat_mask, score_mask, class_weights_dict, 
                use_focal=self.config.get('use_focal_loss', True)
            )
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def evaluate(self, model, val_loaders):
        """모델 평가"""
        model.eval()
        results = {}
        
        with torch.no_grad():
            # Manual validation
            manual_results = self.evaluate_manual(model, val_loaders['val_manual'])
            results['manual'] = manual_results
            
            # TyDiP validation
            tydip_results = self.evaluate_tydip(model, val_loaders['val_tydip'])
            results['tydip'] = tydip_results
        
        return results
    
    def evaluate_manual(self, model, val_loader):
        """Manual 데이터 평가 (피처별 분류 + 스코어 회귀)"""
        all_feature_preds = {name: [] for name in model.feature_heads.keys()}
        all_feature_labels = {name: [] for name in model.feature_heads.keys()}
        all_score_preds = []
        all_score_labels = []
        
        for batch in val_loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            feat_labels = batch['feat_labels'].to(self.device)
            score_labels = batch['score'].to(self.device)
            
            feature_logits, score_pred = model(input_ids, attention_mask)
            
            # 피처별 예측 수집
            for i, feat_name in enumerate(model.feature_heads.keys()):
                preds = torch.argmax(feature_logits[feat_name], dim=-1)
                all_feature_preds[feat_name].extend(preds.cpu().numpy())
                all_feature_labels[feat_name].extend(feat_labels[:, i].cpu().numpy())
            
            # 스코어 예측 수집
            all_score_preds.extend(score_pred.cpu().numpy())
            all_score_labels.extend(score_labels.cpu().numpy())
        
        # 피처별 성능 계산
        feature_metrics = {}
        for feat_name in model.feature_heads.keys():
            preds = np.array(all_feature_preds[feat_name])
            labels = np.array(all_feature_labels[feat_name])
            
            f1_macro = f1_score(labels, preds, average='macro', zero_division=0)
            f1_micro = f1_score(labels, preds, average='micro', zero_division=0)
            
            feature_metrics[feat_name] = {
                'f1_macro': f1_macro,
                'f1_micro': f1_micro,
                'accuracy': (preds == labels).mean()
            }
        
        # 스코어 성능 계산
        score_preds = np.array(all_score_preds)
        score_labels = np.array(all_score_labels)
        score_mae = np.mean(np.abs(score_preds - score_labels))
        score_rmse = np.sqrt(np.mean((score_preds - score_labels) ** 2))
        
        return {
            'features': feature_metrics,
            'score': {
                'mae': score_mae,
                'rmse': score_rmse
            }
        }
    
    def evaluate_tydip(self, model, val_loader):
        """TyDiP 데이터 평가 (스코어만)"""
        all_score_preds = []
        all_score_labels = []
        
        for batch in val_loader:
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            score_labels = batch['score'].to(self.device)
            
            _, score_pred = model(input_ids, attention_mask)
            
            all_score_preds.extend(score_pred.cpu().numpy())
            all_score_labels.extend(score_labels.cpu().numpy())
        
        score_preds = np.array(all_score_preds)
        score_labels = np.array(all_score_labels)
        score_mae = np.mean(np.abs(score_preds - score_labels))
        score_rmse = np.sqrt(np.mean((score_preds - score_labels) ** 2))
        
        return {
            'mae': score_mae,
            'rmse': score_rmse
        }
    
    def train_fold(self, fold, train_indices, val_indices):
        """단일 fold 훈련"""
        print(f"\n🚀 Starting fold {fold + 1}")
        
        # 데이터로더 생성
        dataloaders, manual_train = self.create_dataloaders(
            train_indices, val_indices, self.manual_df, self.tydip_df
        )
        
        # 클래스 가중치 계산
        class_weights_dict = self.compute_class_weights_all_features(manual_train)
        
        # 모델 초기화
        model = ImprovedMultiTaskModel(
            model_name=self.config['model_name'],
            dropout_rate=self.config['dropout_rate']
        ).to(self.device)
        
        # 옵티마이저 및 스케줄러 설정
        total_steps = (len(dataloaders['train_manual']) + len(dataloaders['train_tydip'])) * self.config['num_epochs']
        
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config['weight_decay']
        )
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(total_steps * 0.1),
            num_training_steps=total_steps
        )
        
        # 훈련 루프
        best_manual_f1 = 0.0
        patience_counter = 0
        
        for epoch in range(self.config['num_epochs']):
            print(f"\nEpoch {epoch + 1}/{self.config['num_epochs']}")
            
            # 훈련
            train_loss = self.train_epoch(model, dataloaders, optimizer, scheduler, class_weights_dict)
            
            # 평가
            eval_results = self.evaluate(model, dataloaders)
            
            # 현재 성능 출력
            manual_f1_avg = np.mean([metrics['f1_macro'] for metrics in eval_results['manual']['features'].values()])
            manual_score_mae = eval_results['manual']['score']['mae']
            tydip_score_mae = eval_results['tydip']['mae']
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Manual Avg F1: {manual_f1_avg:.4f}, Score MAE: {manual_score_mae:.4f}")
            print(f"TyDiP Score MAE: {tydip_score_mae:.4f}")
            
            # Early stopping
            if manual_f1_avg > best_manual_f1:
                best_manual_f1 = manual_f1_avg
                patience_counter = 0
                
                # 최고 모델 저장
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'eval_results': eval_results,
                    'fold': fold,
                    'epoch': epoch
                }, f"{self.config['save_dir']}/best_model_fold_{fold}.pt")
                
            else:
                patience_counter += 1
                if patience_counter >= self.config['patience']:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break
        
        return eval_results
    
    def run_kfold_training(self):
        """K-fold 교차검증 실행"""
        print("🔄 Starting K-fold cross-validation")
        
        # 스마트 분할로 전체 validation 생성
        train_indices, val_indices = smart_train_val_split(
            self.manual_df, 
            test_size=0.2, 
            min_rare_samples=2,
            random_state=42
        )
        
        print(f"Initial split - Train: {len(train_indices)}, Val: {len(val_indices)}")
        
        # K-fold를 train 부분에서만 수행
        train_df = self.manual_df.iloc[train_indices]
        kfold = StratifiedKFold(n_splits=self.config['n_folds'], shuffle=True, random_state=42)
        
        # 층화를 위한 복합 타겟 생성 (주요 피처들의 조합)
        stratify_target = train_df['feat_ending'] * 100 + train_df['feat_attack'] * 10 + train_df['feat_power']
        
        all_results = []
        
        for fold, (kfold_train_idx, kfold_val_idx) in enumerate(kfold.split(train_df, stratify_target)):
            # 원래 인덱스로 변환
            fold_train_indices = [train_indices[i] for i in kfold_train_idx]
            fold_val_indices = [train_indices[i] for i in kfold_val_idx]
            
            # fold 훈련
            fold_results = self.train_fold(fold, fold_train_indices, fold_val_indices)
            all_results.append(fold_results)
        
        # 최종 결과 집계
        self.aggregate_results(all_results)
        
        # 최종 홀드아웃 테스트
        print("\n🎯 Final evaluation on hold-out validation set")
        self.final_evaluation(val_indices)
    
    def aggregate_results(self, all_results):
        """K-fold 결과 집계"""
        print("\n📊 K-fold Results Summary")
        
        # 피처별 성능 집계
        feature_names = list(all_results[0]['manual']['features'].keys())
        
        for feat_name in feature_names:
            f1_scores = [result['manual']['features'][feat_name]['f1_macro'] for result in all_results]
            print(f"{feat_name} F1: {np.mean(f1_scores):.3f} ± {np.std(f1_scores):.3f}")
        
        # 스코어 성능 집계
        manual_maes = [result['manual']['score']['mae'] for result in all_results]
        tydip_maes = [result['tydip']['mae'] for result in all_results]
        
        print(f"Manual Score MAE: {np.mean(manual_maes):.3f} ± {np.std(manual_maes):.3f}")
        print(f"TyDiP Score MAE: {np.mean(tydip_maes):.3f} ± {np.std(tydip_maes):.3f}")
    
    def final_evaluation(self, val_indices):
        """최종 홀드아웃 평가"""
        # 가장 좋은 fold 모델 로드
        best_model_path = None
        best_f1 = 0.0
        
        for fold in range(self.config['n_folds']):
            model_path = f"{self.config['save_dir']}/best_model_fold_{fold}.pt"
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device)
                avg_f1 = np.mean([metrics['f1_macro'] for metrics in checkpoint['eval_results']['manual']['features'].values()])
                
                if avg_f1 > best_f1:
                    best_f1 = avg_f1
                    best_model_path = model_path
        
        if best_model_path:
            print(f"Loading best model from {best_model_path}")
            
            # 최고 모델로 최종 평가
            model = ImprovedMultiTaskModel(
                model_name=self.config['model_name'],
                dropout_rate=self.config['dropout_rate']
            ).to(self.device)
            
            checkpoint = torch.load(best_model_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # 홀드아웃 validation 평가
            dataloaders, _ = self.create_dataloaders(
                [], val_indices, self.manual_df, self.tydip_df
            )
            
            final_results = self.evaluate(model, {'val_manual': dataloaders['val_manual'], 'val_tydip': dataloaders['val_tydip']})
            
            print("\n🎯 Final Hold-out Results:")
            manual_f1_avg = np.mean([metrics['f1_macro'] for metrics in final_results['manual']['features'].values()])
            print(f"Manual Avg F1: {manual_f1_avg:.4f}")
            print(f"Manual Score MAE: {final_results['manual']['score']['mae']:.4f}")
            print(f"TyDiP Score MAE: {final_results['tydip']['mae']:.4f}")

# 설정
config = {
    'model_name': 'monologg/kobert',
    'manual_data_path': './data/manual_labels.csv',
    'tydip_data_path': './data/ko_test.csv',
    'save_dir': './results',
    'max_length': 256,
    'batch_size': 16,
    'learning_rate': 2e-5,
    'weight_decay': 0.01,
    'dropout_rate': 0.3,
    'num_epochs': 8,
    'patience': 3,
    'n_folds': 3,
    'use_focal_loss': True,
    'normalize_scores': True
}

if __name__ == "__main__":
    trainer = MultitaskTrainer(config)
    trainer.load_data()
    trainer.run_kfold_training() 
#!/usr/bin/env python3
"""
멀티태스크 공손도 예측 모델 트레이너
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import wandb
import os
import json
from datetime import datetime


class MultitaskTrainer:
    """멀티태스크 공손도 예측 모델 트레이너"""
    
    def __init__(self, model, loss_fn, train_dataloader, val_dataloader, 
                 device: str = 'cuda', learning_rate: float = 2e-5, 
                 warmup_ratio: float = 0.1, weight_decay: float = 0.01,
                 max_grad_norm: float = 1.0, save_dir: str = './checkpoints',
                 use_wandb: bool = False, project_name: str = "ko-politeness"):
        
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.device = device
        self.max_grad_norm = max_grad_norm
        self.save_dir = save_dir
        self.use_wandb = use_wandb
        
        # 옵티마이저 설정
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # 스케줄러 설정
        total_steps = len(train_dataloader) * 10  # 최대 10 epochs 가정
        warmup_steps = int(total_steps * warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
        
        # 베스트 모델 추적
        self.best_val_score = float('-inf')
        self.best_model_path = None
        self.early_stopping_patience = 3
        self.early_stopping_counter = 0
        
        # 로깅 설정
        if use_wandb:
            wandb.init(project=project_name, config={
                'learning_rate': learning_rate,
                'warmup_ratio': warmup_ratio,
                'weight_decay': weight_decay,
                'max_grad_norm': max_grad_norm
            })
        
        os.makedirs(save_dir, exist_ok=True)
        
    def train_epoch(self) -> Dict[str, float]:
        """한 epoch 학습"""
        self.model.train()
        total_loss = 0
        total_feature_loss = 0
        total_score_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(self.train_dataloader, desc="Training")
        
        for batch in progress_bar:
            # 데이터를 device로 이동
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            predictions = self.model(batch['input_ids'], batch['attention_mask'])
            
            # Loss 계산
            losses = self.loss_fn(predictions, batch)
            loss = losses['total_loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            
            # Update
            self.optimizer.step()
            self.scheduler.step()
            
            # 통계 업데이트
            total_loss += loss.item()
            total_feature_loss += losses['feature_loss'].item()
            total_score_loss += losses['score_loss'].item()
            num_batches += 1
            
            # Progress bar 업데이트
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'feat_loss': f"{losses['feature_loss'].item():.4f}",
                'score_loss': f"{losses['score_loss'].item():.4f}"
            })
        
        return {
            'train_loss': total_loss / num_batches,
            'train_feature_loss': total_feature_loss / num_batches,
            'train_score_loss': total_score_loss / num_batches
        }
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """모델 평가"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        # 예측값과 실제값 저장
        all_feature_preds = [[] for _ in range(7)]  # 7개 피처
        all_feature_labels = [[] for _ in range(7)]
        all_score_preds = []
        all_score_labels = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                # 데이터를 device로 이동
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                predictions = self.model(batch['input_ids'], batch['attention_mask'])
                
                # Loss 계산
                losses = self.loss_fn(predictions, batch)
                total_loss += losses['total_loss'].item()
                num_batches += 1
                
                # 예측값 수집 - 피처 분류
                feature_mask = batch['feature_mask']  # [batch_size, 7]
                feature_labels = batch['feature_labels']  # [batch_size, 7]
                
                for i, logits in enumerate(predictions['feature_logits']):
                    valid_mask = feature_mask[:, i]  # 유효한 샘플들
                    if valid_mask.sum() > 0:
                        preds = torch.argmax(logits[valid_mask], dim=-1)
                        labels = feature_labels[valid_mask, i]
                        
                        all_feature_preds[i].extend(preds.cpu().numpy())
                        all_feature_labels[i].extend(labels.cpu().numpy())
                
                # 예측값 수집 - 점수 회귀
                score_mask = batch['score_mask']
                if score_mask.sum() > 0:
                    score_preds = predictions['score_pred'][score_mask]
                    score_labels = batch['score'][score_mask]
                    
                    all_score_preds.extend(score_preds.cpu().numpy())
                    all_score_labels.extend(score_labels.cpu().numpy())
        
        # 메트릭 계산
        metrics = {'val_loss': total_loss / num_batches}
        
        # 피처별 분류 메트릭
        feature_names = ['feat_ending', 'feat_strat_cnt', 'feat_indirect', 
                        'feat_command', 'feat_attack', 'feat_power', 'feat_distance']
        
        feature_f1_scores = []
        for i, feature_name in enumerate(feature_names):
            if len(all_feature_preds[i]) > 0:
                f1_macro = f1_score(all_feature_labels[i], all_feature_preds[i], average='macro', zero_division=0)
                acc = accuracy_score(all_feature_labels[i], all_feature_preds[i])
                
                metrics[f'{feature_name}_f1'] = f1_macro
                metrics[f'{feature_name}_acc'] = acc
                feature_f1_scores.append(f1_macro)
            else:
                feature_f1_scores.append(0.0)
        
        # 전체 피처 평균 F1
        metrics['avg_feature_f1'] = np.mean(feature_f1_scores)
        
        # 점수 회귀 메트릭
        if len(all_score_preds) > 0:
            mae = mean_absolute_error(all_score_labels, all_score_preds)
            rmse = np.sqrt(mean_squared_error(all_score_labels, all_score_preds))
            
            metrics['score_mae'] = mae
            metrics['score_rmse'] = rmse
        
        return metrics
    
    def train(self, num_epochs: int = 5) -> Dict:
        """전체 학습 프로세스"""
        print(f"🚀 멀티태스크 학습 시작 (총 {num_epochs} epochs)")
        
        training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_avg_feature_f1': [],
            'val_score_mae': []
        }
        
        for epoch in range(num_epochs):
            print(f"\n📅 Epoch {epoch + 1}/{num_epochs}")
            
            # Training
            train_metrics = self.train_epoch()
            
            # Validation  
            val_metrics = self.evaluate(self.val_dataloader)
            
            # 로깅
            epoch_metrics = {**train_metrics, **val_metrics, 'epoch': epoch + 1}
            
            if self.use_wandb:
                wandb.log(epoch_metrics)
            
            # 출력
            print(f"   Train Loss: {train_metrics['train_loss']:.4f}")
            print(f"   Val Loss: {val_metrics['val_loss']:.4f}")
            print(f"   Avg Feature F1: {val_metrics.get('avg_feature_f1', 0):.4f}")
            if 'score_mae' in val_metrics:
                print(f"   Score MAE: {val_metrics['score_mae']:.4f}")
            
            # 히스토리 저장
            training_history['train_loss'].append(train_metrics['train_loss'])
            training_history['val_loss'].append(val_metrics['val_loss'])
            training_history['val_avg_feature_f1'].append(val_metrics.get('avg_feature_f1', 0))
            training_history['val_score_mae'].append(val_metrics.get('score_mae', 0))
            
            # 베스트 모델 저장 (평균 F1 기준)
            current_score = val_metrics.get('avg_feature_f1', 0)
            if current_score > self.best_val_score:
                self.best_val_score = current_score
                self.best_model_path = self.save_model(epoch + 1, val_metrics)
                self.early_stopping_counter = 0
                print(f"   ✅ 베스트 모델 저장됨! (F1: {current_score:.4f})")
            else:
                self.early_stopping_counter += 1
                print(f"   ⏳ Early stopping: {self.early_stopping_counter}/{self.early_stopping_patience}")
            
            # Early stopping
            if self.early_stopping_counter >= self.early_stopping_patience:
                print(f"   🛑 Early stopping 발동! (Patience: {self.early_stopping_patience})")
                break
        
        print(f"\n🎯 학습 완료! 베스트 모델: {self.best_model_path}")
        return training_history
    
    def save_model(self, epoch: int, metrics: Dict) -> str:
        """모델 저장"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"multitask_kobert_epoch{epoch}_{timestamp}.pt"
        model_path = os.path.join(self.save_dir, model_name)
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'metrics': metrics,
            'best_val_score': self.best_val_score
        }, model_path)
        
        # 메타데이터 저장
        meta_path = os.path.join(self.save_dir, f"meta_{model_name}.json")
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump({
                'epoch': epoch,
                'metrics': {k: float(v) if isinstance(v, (int, float, np.float32, np.float64)) 
                           else str(v) for k, v in metrics.items()},
                'timestamp': timestamp,
                'model_path': model_path
            }, f, ensure_ascii=False, indent=2)
        
        return model_path
    
    def load_model(self, model_path: str):
        """모델 로드"""
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_score = checkpoint['best_val_score']
        
        print(f"✅ 모델 로드 완료: {model_path}")
        return checkpoint['metrics']


def create_trainer(model, loss_fn, train_dataloader, val_dataloader, **kwargs) -> MultitaskTrainer:
    """트레이너 생성 헬퍼 함수"""
    return MultitaskTrainer(
        model=model,
        loss_fn=loss_fn,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        **kwargs
    ) 
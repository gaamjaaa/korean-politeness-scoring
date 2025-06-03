#!/usr/bin/env python3
"""
올바른 멀티태스크 학습 (데이터 리키지 제거):
텍스트 → 피처 추출 → 피처별 점수 + 전체 점수 예측
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import json
import os

def main():
    print("🔧 CORRECTED MULTITASK Korean Politeness Training")
    print("=" * 60)
    print("🚨 FIXING DATA LEAKAGE:")
    print("   ❌ Before: Manual features → Manual features (LEAKAGE!)")
    print("   ✅ After: Text → Extracted features → Manual features")
    print()
    print("📊 Target Tasks:")
    print("   1. feat_ending prediction")
    print("   2. feat_strat_cnt prediction") 
    print("   3. feat_indirect prediction")
    print("   4. feat_command prediction")
    print("   5. feat_attack prediction")
    print("   6. feat_power prediction")
    print("   7. feat_distance prediction")
    print("   8. Overall score prediction")
    
    # Manual Labels 로드
    manual_df = pd.read_csv('./data/manual_labels_new_system.csv')
    print(f"\n📊 Using Manual Labels: {len(manual_df)} samples")
    
    # 텍스트에서 피처 추출 (입력으로 사용)
    print(f"\n🔍 Extracting features from text (INPUT)...")
    
    from preprocessing.korean_politeness_analyzer import KoreanPolitenessAnalyzer
    analyzer = KoreanPolitenessAnalyzer()
    
    feature_cols = ['feat_ending', 'feat_strat_cnt', 'feat_indirect', 
                   'feat_command', 'feat_attack', 'feat_power', 'feat_distance']
    
    # X: 텍스트에서 추출한 피처들 (입력)
    extracted_features = []
    for sentence in manual_df['sentence']:
        features = analyzer.extract_features(sentence)
        extracted_features.append([features[feat] for feat in feature_cols])
    
    X = np.array(extracted_features)
    
    # Y: Manual Labels의 ground truth 피처 점수들 + 전체 점수 (타겟)
    target_cols = feature_cols + ['score']
    Y = manual_df[target_cols].values
    
    print(f"\n🎯 Corrected Multitask Setup:")
    print(f"   Input: Extracted features from TEXT")
    print(f"   Output: Manual Labels ground truth scores")
    print(f"   Input Features: {X.shape[1]} features")
    print(f"   Output Targets: {Y.shape[1]} targets")
    print(f"   Target Tasks: {target_cols}")
    
    # 입력과 출력 통계 비교
    print(f"\n📊 Input vs Output Statistics:")
    for i, feat in enumerate(feature_cols):
        input_mean = np.mean(X[:, i])
        output_mean = np.mean(Y[:, i])
        print(f"   {feat}: Input={input_mean:.3f}, Target={output_mean:.3f}")
    
    # 층화분할을 위해 전체 점수 기준으로 분할
    y_discrete = pd.cut(manual_df['score'], bins=5, labels=False)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # 멀티태스크 모델
    multitask_model = MultiOutputRegressor(
        RandomForestRegressor(
            n_estimators=200, 
            max_depth=10, 
            min_samples_split=5,
            random_state=42
        )
    )
    
    print(f"\n🔄 Starting 5-Fold Cross Validation...")
    
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y_discrete)):
        print(f"\n📊 Fold {fold + 1}/5:")
        
        X_train, X_val = X[train_idx], X[val_idx]
        Y_train, Y_val = Y[train_idx], Y[val_idx]
        
        print(f"   Train: {len(X_train)} samples")
        print(f"   Val: {len(X_val)} samples")
        
        # 훈련
        multitask_model.fit(X_train, Y_train)
        
        # 예측
        Y_pred = multitask_model.predict(X_val)
        
        # 각 태스크별 평가
        task_results = {}
        
        for i, task in enumerate(target_cols):
            y_true = Y_val[:, i]
            y_pred = Y_pred[:, i]
            
            mae = mean_absolute_error(y_true, y_pred)
            r2 = r2_score(y_true, y_pred)
            
            task_results[task] = {'mae': mae, 'r2': r2}
            
            print(f"   {task}: MAE={mae:.4f}, R²={r2:.4f}")
        
        fold_results.append(task_results)
    
    # 평균 성능 계산
    print(f"\n🏆 CORRECTED MULTITASK RESULTS (5-Fold CV):")
    print("=" * 60)
    
    avg_results = {}
    for i, task in enumerate(target_cols):
        maes = [fold[task]['mae'] for fold in fold_results]
        r2s = [fold[task]['r2'] for fold in fold_results]
        
        avg_mae = np.mean(maes)
        avg_r2 = np.mean(r2s)
        std_r2 = np.std(r2s)
        
        avg_results[task] = {
            'avg_mae': avg_mae,
            'avg_r2': avg_r2,
            'std_r2': std_r2
        }
        
        print(f"📊 {task}:")
        print(f"   MAE: {avg_mae:.4f}")
        print(f"   R²: {avg_r2:.4f} ± {std_r2:.4f}")
        print()
    
    # 최종 모델 훈련
    print(f"🚀 Training final corrected multitask model...")
    multitask_model.fit(X, Y)
    
    # 피처 중요도 (첫 번째 태스크 기준)
    if hasattr(multitask_model.estimators_[0], 'feature_importances_'):
        importances = multitask_model.estimators_[0].feature_importances_
        print(f"\n🎯 Feature Importance (feat_ending task):")
        for feat, imp in zip(feature_cols, importances):
            print(f"   {feat}: {imp:.4f}")
    
    # Ko Test로 일반화 테스트
    print(f"\n🧪 Generalization Test on Ko Test...")
    
    ko_test_df = pd.read_csv('./data/ko_test.csv')
    
    ko_features = []
    for sentence in ko_test_df['sentence']:
        features = analyzer.extract_features(sentence)
        ko_features.append([features[feat] for feat in feature_cols])
    
    X_ko = np.array(ko_features)
    
    # 멀티태스크 예측
    Y_pred_ko = multitask_model.predict(X_ko)
    
    # Ko Test는 전체 점수만 있으므로 마지막 태스크(score)만 평가
    y_ko_true = ko_test_df['score'].values
    y_ko_pred = Y_pred_ko[:, -1]  # 마지막 컬럼이 score
    
    mae_ko = mean_absolute_error(y_ko_true, y_ko_pred)
    r2_ko = r2_score(y_ko_true, y_ko_pred)
    
    print(f"   Ko Test Overall Score - MAE: {mae_ko:.4f}, R²: {r2_ko:.4f}")
    
    # 결과 저장
    os.makedirs('./corrected_multitask_results', exist_ok=True)
    
    # 성능 요약
    summary = {
        'approach': 'Corrected Multitask Learning (No Data Leakage)',
        'description': 'Text → Extracted features → Feature scores + Overall score',
        'data_leakage_fixed': True,
        'input_source': 'Text-extracted features',
        'output_target': 'Manual Labels ground truth scores',
        'dataset_size': len(manual_df),
        'cv_results': avg_results,
        'ko_test_generalization': {
            'overall_score_mae': mae_ko,
            'overall_score_r2': r2_ko
        }
    }
    
    with open('./corrected_multitask_results/corrected_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    
    # Ko Test 예측 결과 저장
    ko_results_df = ko_test_df.copy()
    ko_results_df['predicted_score'] = y_ko_pred
    
    # 각 피처 예측값도 저장
    for i, feat in enumerate(feature_cols):
        ko_results_df[f'predicted_{feat}'] = Y_pred_ko[:, i]
        ko_results_df[f'extracted_{feat}'] = X_ko[:, i]
    
    ko_results_df.to_csv('./corrected_multitask_results/ko_test_corrected_predictions.csv', index=False)
    
    print(f"\n💾 Results saved to './corrected_multitask_results/'")
    
    # 성능 비교
    print(f"\n🔍 DATA LEAKAGE IMPACT:")
    print(f"📊 Previous (WITH LEAKAGE):")
    print(f"   feat_power: R² 0.9988 (too perfect!)")
    print(f"   feat_ending: R² 0.9947 (suspicious)")
    print(f"📊 Current (CORRECTED):")
    print(f"   Overall Score R²: {avg_results['score']['avg_r2']:.4f}")
    
    best_feature = max(
        [task for task in target_cols if task != 'score'], 
        key=lambda x: avg_results[x]['avg_r2']
    )
    worst_feature = min(
        [task for task in target_cols if task != 'score'], 
        key=lambda x: avg_results[x]['avg_r2']
    )
    
    print(f"   Best Feature: {best_feature} (R² {avg_results[best_feature]['avg_r2']:.4f})")
    print(f"   Worst Feature: {worst_feature} (R² {avg_results[worst_feature]['avg_r2']:.4f})")
    
    print(f"\n🎯 Corrected Multitask Learning:")
    print(f"   ✓ No data leakage")
    print(f"   ✓ Text → Features → Scores")
    print(f"   ✓ Realistic performance")
    print(f"   ✓ Individual feature evaluation possible")

if __name__ == "__main__":
    main() 
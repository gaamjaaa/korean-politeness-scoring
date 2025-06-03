#!/usr/bin/env python3
"""
Manual Labels vs Ko Test 데이터 차이 분석
성능 저하 원인 파악 및 개선 방안 제시
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

def analyze_data_differences():
    """두 데이터셋 간 차이점 분석"""
    print("🔍 Data Distribution Analysis")
    print("=" * 50)
    
    # 데이터 로드
    manual_df = pd.read_csv('./data/manual_labels_new_system.csv')
    ko_test_df = pd.read_csv('./data/ko_test_new_system.csv')
    
    print(f"📊 Dataset Sizes:")
    print(f"   Manual Labels: {len(manual_df)} samples")
    print(f"   Ko Test: {len(ko_test_df)} samples")
    
    # 점수 분포 분석
    print(f"\n📈 Score Distribution:")
    print(f"   Manual - Mean: {manual_df['score'].mean():.3f}, Std: {manual_df['score'].std():.3f}")
    print(f"   Ko Test - Mean: {ko_test_df['score'].mean():.3f}, Std: {ko_test_df['score'].std():.3f}")
    print(f"   Manual Range: {manual_df['score'].min()} ~ {manual_df['score'].max()}")
    print(f"   Ko Test Range: {ko_test_df['score'].min()} ~ {ko_test_df['score'].max()}")
    
    # 피처별 분포 비교
    feature_cols = ['feat_ending', 'feat_strat_cnt', 'feat_indirect', 
                   'feat_command', 'feat_attack', 'feat_power', 'feat_distance']
    
    print(f"\n🎯 Feature Distribution Comparison:")
    for feat in feature_cols:
        manual_mean = manual_df[feat].mean()
        ko_test_mean = ko_test_df[feat].mean()
        diff = abs(manual_mean - ko_test_mean)
        
        print(f"   {feat}:")
        print(f"     Manual: {manual_mean:.3f}, Ko Test: {ko_test_mean:.3f}, Diff: {diff:.3f}")
    
    return manual_df, ko_test_df, feature_cols

def test_separate_training(manual_df, ko_test_df, feature_cols):
    """개별 데이터셋으로 훈련했을 때 성능"""
    print(f"\n🧪 Separate Training Test")
    print("-" * 40)
    
    # Manual Labels만으로 훈련
    X_manual = manual_df[feature_cols]
    y_manual = manual_df['score']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_manual, y_manual, test_size=0.2, random_state=42
    )
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mae_manual = mean_absolute_error(y_test, y_pred)
    r2_manual = r2_score(y_test, y_pred)
    
    print(f"Manual Only - MAE: {mae_manual:.4f}, R²: {r2_manual:.4f}")
    
    # Ko Test만으로 훈련
    X_ko = ko_test_df[feature_cols]
    y_ko = ko_test_df['score']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_ko, y_ko, test_size=0.2, random_state=42
    )
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    mae_ko = mean_absolute_error(y_test, y_pred)
    r2_ko = r2_score(y_test, y_pred)
    
    print(f"Ko Test Only - MAE: {mae_ko:.4f}, R²: {r2_ko:.4f}")
    
    return r2_manual, r2_ko

def test_weighted_training():
    """가중치 기반 훈련"""
    print(f"\n⚖️ Weighted Training Test")
    print("-" * 40)
    
    manual_df = pd.read_csv('./data/manual_labels_new_system.csv')
    ko_test_df = pd.read_csv('./data/ko_test_new_system.csv')
    
    feature_cols = ['feat_ending', 'feat_strat_cnt', 'feat_indirect', 
                   'feat_command', 'feat_attack', 'feat_power', 'feat_distance']
    
    # 데이터 합치기
    manual_df['source'] = 'manual'
    ko_test_df['source'] = 'ko_test'
    combined_df = pd.concat([manual_df, ko_test_df], ignore_index=True)
    
    X = combined_df[feature_cols]
    y = combined_df['score']
    
    # Manual 데이터에 더 높은 가중치 부여
    sample_weights = np.where(combined_df['source'] == 'manual', 2.0, 1.0)
    
    X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
        X, y, sample_weights, test_size=0.2, random_state=42
    )
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train, sample_weight=weights_train)
    
    y_pred = model.predict(X_test)
    mae_weighted = mean_absolute_error(y_test, y_pred)
    r2_weighted = r2_score(y_test, y_pred)
    
    print(f"Weighted Training - MAE: {mae_weighted:.4f}, R²: {r2_weighted:.4f}")
    
    return r2_weighted

def suggest_improvements():
    """성능 개선 방안 제시"""
    print(f"\n💡 Performance Improvement Suggestions")
    print("=" * 50)
    
    print("1. 🎯 데이터 품질 개선:")
    print("   - Manual Labels 데이터 추가 수집")
    print("   - Ko Test 데이터 재라벨링 검토")
    print("   - 도메인 특화 피처 추가")
    
    print("\n2. ⚖️ 훈련 방식 개선:")
    print("   - Manual 데이터에 높은 가중치 부여")
    print("   - 도메인 적응(Domain Adaptation) 기법")
    print("   - 앙상블 모델 (Manual + Ko Test 별도 훈련)")
    
    print("\n3. 🤖 모델 개선:")
    print("   - KoBERT 기반 텍스트 임베딩 추가")
    print("   - 멀티태스크 학습 (피처 예측 + 점수 예측)")
    print("   - 규제 강화 (L1/L2, Dropout)")
    
    print("\n4. 📊 평가 방식 개선:")
    print("   - 도메인별 분리 평가")
    print("   - 가중 평균 성능 지표")
    print("   - 실제 사용 케이스 기반 평가")

def main():
    """메인 함수"""
    print("🔍 Korean Politeness Score - Data Analysis & Improvement")
    print("=" * 60)
    
    # 1. 데이터 차이 분석
    manual_df, ko_test_df, feature_cols = analyze_data_differences()
    
    # 2. 개별 훈련 테스트
    r2_manual, r2_ko = test_separate_training(manual_df, ko_test_df, feature_cols)
    
    # 3. 가중치 훈련 테스트
    r2_weighted = test_weighted_training()
    
    # 4. 결과 요약
    print(f"\n📊 Performance Summary:")
    print(f"   Manual Only: R² = {r2_manual:.4f}")
    print(f"   Ko Test Only: R² = {r2_ko:.4f}")
    print(f"   Combined (Equal): R² = 0.0590 (from previous result)")
    print(f"   Combined (Weighted): R² = {r2_weighted:.4f}")
    
    # 5. 개선 방안 제시
    suggest_improvements()
    
    print(f"\n🎉 Analysis completed!")
    print(f"💡 Recommendation: Use weighted training or separate models for better performance.")

if __name__ == "__main__":
    main() 
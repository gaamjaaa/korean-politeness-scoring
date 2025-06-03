import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

def analyze_class_distribution(df):
    """각 피처별 클래스 분포 분석"""
    feature_cols = [
        "feat_ending", "feat_strat_cnt", "feat_command",
        "feat_attack", "feat_power", "feat_distance", "feat_indirect"
    ]
    
    print("📊 현재 클래스 분포 분석")
    print("=" * 60)
    
    distribution_info = {}
    
    for col in feature_cols:
        counts = df[col].value_counts().sort_index()
        distribution_info[col] = counts
        
        print(f"\n🔍 {col}:")
        total = len(df)
        
        for class_val, count in counts.items():
            percentage = (count / total) * 100
            if count < 20:
                status = "❌ 매우 적음"
            elif count < 50:
                status = "⚠️ 적음"
            else:
                status = "✅ 충분"
            
            print(f"  클래스 {class_val}: {count:3d}개 ({percentage:5.1f}%) {status}")
        
        # 희귀 클래스 확인
        rare_classes = counts[counts < 20].index.tolist()
        if rare_classes:
            print(f"  💡 병합 후보: {rare_classes}")
    
    return distribution_info

def merge_rare_classes(df, min_samples_per_class=20, dry_run=True, custom_merge_rules=None):
    """희귀 클래스 병합"""
    feature_cols = [
        "feat_ending", "feat_strat_cnt", "feat_command",
        "feat_attack", "feat_power", "feat_distance", "feat_indirect"
    ]
    
    df_merged = df.copy()
    merge_actions = {}
    
    print(f"\n🔧 희귀 클래스 병합 (최소 {min_samples_per_class}개 기준)")
    print("=" * 60)
    
    for col in feature_cols:
        counts = df[col].value_counts().sort_index()
        max_class = counts.index.max()
        
        print(f"\n📝 {col} 처리:")
        
        # 사용자 정의 병합 규칙이 있는 경우
        if custom_merge_rules and col in custom_merge_rules:
            merge_plan = custom_merge_rules[col]
            for from_class, to_class in merge_plan:
                sample_count = counts.get(from_class, 0)
                print(f"  클래스 {from_class} ({sample_count}개) → 클래스 {to_class}로 병합 (사용자 지정)")
        else:
            # 기본 규칙: 클래스 3만 클래스 2로 병합
            merge_plan = []
            
            for class_val in sorted(counts.index, reverse=True):
                sample_count = counts[class_val]
                
                # 클래스 3만 클래스 2로 병합
                if class_val == 3 and sample_count < min_samples_per_class:
                    target_class = 2
                    merge_plan.append((class_val, target_class))
                    print(f"  클래스 {class_val} ({sample_count}개) → 클래스 {target_class}로 병합")
                elif sample_count < min_samples_per_class and class_val > 0 and class_val != 3:
                    # 클래스 3이 아닌 경우는 경고만 표시
                    print(f"  ⚠️ 클래스 {class_val} ({sample_count}개) - 적지만 유지")
        
        # 실제 병합 수행
        if not dry_run:
            if custom_merge_rules and col in custom_merge_rules:
                merge_plan = custom_merge_rules[col]
            
            for from_class, to_class in merge_plan:
                mask = df_merged[col] == from_class
                df_merged.loc[mask, col] = to_class
                count = mask.sum()
                print(f"  ✅ 실제 병합: {from_class} → {to_class} ({count}개)")
        
        merge_actions[col] = merge_plan if 'merge_plan' in locals() else []
    
    if dry_run:
        print(f"\n💡 이것은 미리보기입니다. 실제 적용하려면 dry_run=False로 설정하세요.")
    
    return df_merged, merge_actions

def create_custom_merge_rules():
    """사용자 정의 병합 규칙 생성"""
    # 클래스 3만 클래스 2로 병합하는 규칙
    custom_rules = {
        "feat_ending": [(3, 2)],
        "feat_strat_cnt": [],  # 원래 최대가 2라서 클래스 3이 없음
        "feat_command": [(3, 2)],
        "feat_attack": [(3, 2)],
        "feat_power": [(3, 2)],
        "feat_distance": [(3, 2)],
        "feat_indirect": []  # 원래 최대가 2라서 클래스 3이 없음
    }
    
    return custom_rules

def visualize_before_after(df_original, df_merged):
    """병합 전후 분포 시각화"""
    feature_cols = [
        "feat_ending", "feat_strat_cnt", "feat_command",
        "feat_attack", "feat_power", "feat_distance", "feat_indirect"
    ]
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for i, col in enumerate(feature_cols):
        if i < len(axes):
            ax = axes[i]
            
            # 원본 분포
            original_counts = df_original[col].value_counts().sort_index()
            merged_counts = df_merged[col].value_counts().sort_index()
            
            # 모든 클래스 인덱스 통합
            all_classes = sorted(set(original_counts.index) | set(merged_counts.index))
            
            # 0으로 채우기
            original_values = [original_counts.get(cls, 0) for cls in all_classes]
            merged_values = [merged_counts.get(cls, 0) for cls in all_classes]
            
            # 막대 그래프
            x_pos = np.arange(len(all_classes))
            width = 0.35
            
            ax.bar(x_pos - width/2, original_values, width, 
                   label='원본', alpha=0.7, color='red')
            ax.bar(x_pos + width/2, merged_values, width,
                   label='병합 후', alpha=0.7, color='blue')
            
            ax.set_xlabel('클래스')
            ax.set_ylabel('샘플 수')
            ax.set_title(f'{col}')
            ax.set_xticks(x_pos)
            ax.set_xticklabels(all_classes)
            ax.legend()
            ax.grid(True, alpha=0.3)
    
    # 빈 subplot 제거
    if len(feature_cols) < len(axes):
        axes[-1].remove()
    
    plt.tight_layout()
    plt.savefig('./results/class_distribution_comparison.png', dpi=300, bbox_inches='tight')
    print("📊 시각화 저장: ./results/class_distribution_comparison.png")
    plt.close()  # 메모리 절약

def update_model_config(merge_actions):
    """병합된 클래스에 맞게 모델 설정 업데이트"""
    print(f"\n⚙️ 모델 설정 업데이트 정보:")
    print("=" * 40)
    
    updated_configs = {}
    
    for col, actions in merge_actions.items():
        # 원래 최대 클래스 번호
        if col in ["feat_strat_cnt", "feat_indirect"]:
            original_max = 2
        else:
            original_max = 3
        
        # 병합 후 최대 클래스 번호 계산
        merged_classes = set(range(original_max + 1))
        for from_class, to_class in actions:
            merged_classes.discard(from_class)
        
        new_max = max(merged_classes) if merged_classes else 0
        num_classes = new_max + 1
        
        updated_configs[col] = num_classes
        print(f"{col}: {original_max + 1} → {num_classes} 클래스")
    
    return updated_configs

def preprocess_data(input_path, output_path, min_samples=20, visualize=True, merge_mode="class3_only"):
    """데이터 전처리 메인 함수"""
    print("🔄 데이터 전처리 시작")
    
    # 데이터 로드
    df = pd.read_csv(input_path)
    print(f"📁 원본 데이터: {len(df)}개 샘플")
    
    # 1단계: 현재 분포 분석
    print("\n" + "=" * 60)
    print("1️⃣ 현재 클래스 분포 분석")
    distribution_info = analyze_class_distribution(df)
    
    # 2단계: 병합 모드 설정
    print("\n" + "=" * 60)
    print("2️⃣ 병합 모드 설정")
    
    custom_rules = None
    if merge_mode == "class3_only":
        print("📋 모드: 클래스 3만 클래스 2로 병합")
        custom_rules = create_custom_merge_rules()
    elif merge_mode == "aggressive":
        print("📋 모드: 모든 희귀 클래스를 한 단계씩 병합")
        custom_rules = None
    
    # 3단계: 병합 계획 미리보기
    print("\n" + "=" * 60)
    print("3️⃣ 병합 계획 미리보기")
    _, merge_actions = merge_rare_classes(df, min_samples, dry_run=True, custom_merge_rules=custom_rules)
    
    # 4단계: 사용자 확인
    print("\n" + "=" * 60)
    confirm = input("계속 진행하시겠습니까? (y/n): ").lower().strip()
    
    if confirm != 'y':
        print("❌ 전처리를 취소했습니다.")
        return None
    
    # 5단계: 실제 병합 수행
    print("\n" + "=" * 60)
    print("4️⃣ 실제 병합 수행")
    df_merged, merge_actions = merge_rare_classes(df, min_samples, dry_run=False, custom_merge_rules=custom_rules)
    
    # 6단계: 결과 분석
    print("\n" + "=" * 60)
    print("5️⃣ 병합 후 분포 확인")
    analyze_class_distribution(df_merged)
    
    # 7단계: 시각화
    if visualize:
        print("\n📊 시각화 생성 중...")
        visualize_before_after(df, df_merged)
    
    # 8단계: 모델 설정 업데이트 정보
    updated_configs = update_model_config(merge_actions)
    
    # 9단계: 저장
    df_merged.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"💾 전처리된 데이터 저장: {output_path}")
    
    # 병합 정보 저장
    merge_info = {
        'merge_actions': merge_actions,
        'updated_configs': updated_configs,
        'min_samples_threshold': min_samples,
        'merge_mode': merge_mode
    }
    
    import json
    info_path = output_path.replace('.csv', '_merge_info.json')
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(merge_info, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"📋 병합 정보 저장: {info_path}")
    
    return df_merged, merge_actions, updated_configs

def quick_analysis():
    """빠른 분석 (현재 데이터만)"""
    try:
        df = pd.read_csv('./data/manual_labels.csv')
        analyze_class_distribution(df)
        
        # 병합 시뮬레이션 - 클래스 3만 병합
        print("\n" + "="*60)
        print("💡 병합 시뮬레이션 (클래스 3만 클래스 2로 병합)")
        custom_rules = create_custom_merge_rules()
        merge_rare_classes(df, min_samples=20, dry_run=True, custom_merge_rules=custom_rules)
        
    except FileNotFoundError:
        print("❌ './data/manual_labels.csv' 파일을 찾을 수 없습니다.")

if __name__ == "__main__":
    import os
    
    # 결과 디렉토리 생성
    os.makedirs('./results', exist_ok=True)
    
    print("🔍 한국어 공손도 데이터 전처리")
    print("="*50)
    print("1. 빠른 분석만 보기")
    print("2. 전체 전처리 수행 (클래스 3만 병합)")
    print("3. 전체 전처리 수행 (모든 희귀 클래스 병합)")
    
    choice = input("\n선택하세요 (1, 2, 또는 3): ").strip()
    
    if choice == "1":
        quick_analysis()
    elif choice == "2":
        input_path = './data/manual_labels.csv'
        output_path = './data/manual_labels_processed.csv'
        
        result = preprocess_data(input_path, output_path, min_samples=20, merge_mode="class3_only")
        
        if result:
            print("\n✅ 전처리 완료!")
            print("📌 다음 단계:")
            print("1. train_multitask.py에서 데이터 경로를 수정하세요:")
            print(f"   'manual_data_path': '{output_path}'")
            print("2. 모델을 다시 훈련하세요.")
    elif choice == "3":
        input_path = './data/manual_labels.csv'
        output_path = './data/manual_labels_processed.csv'
        
        result = preprocess_data(input_path, output_path, min_samples=20, merge_mode="aggressive")
        
        if result:
            print("\n✅ 전처리 완료!")
            print("📌 다음 단계:")
            print("1. train_multitask.py에서 데이터 경로를 수정하세요:")
            print(f"   'manual_data_path': '{output_path}'")
            print("2. 모델을 다시 훈련하세요.")
    else:
        print("❌ 잘못된 선택입니다.") 
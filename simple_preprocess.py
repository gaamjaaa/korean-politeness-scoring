import pandas as pd
import json

def merge_class3_to_class2():
    """클래스 3을 클래스 2로 병합하는 간단한 전처리"""
    
    print("🔄 클래스 3 → 클래스 2 병합 시작")
    
    # 데이터 로드
    df = pd.read_csv('./data/manual_labels.csv')
    print(f"📁 원본 데이터: {len(df)}개 샘플")
    
    # 병합할 피처 목록
    feature_cols = [
        "feat_ending", "feat_command", "feat_attack", 
        "feat_power", "feat_distance"
    ]
    
    merge_stats = {}
    
    print("\n📊 병합 전 분포:")
    for col in feature_cols:
        if col in df.columns:
            counts = df[col].value_counts().sort_index()
            print(f"  {col}: {dict(counts)}")
            
            # 클래스 3이 있으면 클래스 2로 병합
            if 3 in counts.index:
                class3_count = counts[3]
                df.loc[df[col] == 3, col] = 2
                merge_stats[col] = class3_count
                print(f"    → 클래스 3 ({class3_count}개) → 클래스 2로 병합 ✅")
    
    print("\n📊 병합 후 분포:")
    for col in feature_cols:
        if col in df.columns:
            counts = df[col].value_counts().sort_index()
            print(f"  {col}: {dict(counts)}")
    
    # 저장
    output_path = './data/manual_labels_processed.csv'
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"\n💾 전처리된 데이터 저장: {output_path}")
    
    # 병합 정보 저장
    merge_info = {
        'merge_actions': merge_stats,
        'description': '클래스 3을 클래스 2로 병합',
        'total_samples': len(df)
    }
    
    info_path = './data/manual_labels_merge_info.json'
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(merge_info, f, ensure_ascii=False, indent=2)
    
    print(f"📋 병합 정보 저장: {info_path}")
    print(f"\n✅ 전처리 완료! 총 {sum(merge_stats.values())}개 샘플이 병합되었습니다.")
    
    return df

if __name__ == "__main__":
    merge_class3_to_class2() 
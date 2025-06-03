#!/usr/bin/env python3
"""
한국어 공손도 예측 시스템 빠른 테스트
모델 훈련 없이 기본 기능들을 테스트합니다.
"""

import os
import pandas as pd
import torch
from transformers import BertTokenizer

def test_data_loading():
    """데이터 로딩 테스트"""
    print("📁 데이터 로딩 테스트...")
    
    try:
        # Manual 데이터 체크
        manual_path = "./data/manual_labels.csv"
        if os.path.exists(manual_path):
            manual_df = pd.read_csv(manual_path)
            print(f"✅ Manual 데이터: {len(manual_df)}개 문장 로드 성공")
            
            # 필수 컬럼 체크
            required_cols = ['sentence', 'feat_ending', 'feat_strat_cnt', 'feat_command', 
                           'feat_attack', 'feat_power', 'feat_distance', 'feat_indirect', 'score']
            missing_cols = [col for col in required_cols if col not in manual_df.columns]
            
            if missing_cols:
                print(f"⚠️  누락된 컬럼: {missing_cols}")
            else:
                print("✅ 모든 필수 컬럼 존재")
                
            # 기본 통계
            print(f"   - 점수 범위: {manual_df['score'].min():.1f} ~ {manual_df['score'].max():.1f}")
            print(f"   - 평균 점수: {manual_df['score'].mean():.2f}")
            
        else:
            print(f"❌ Manual 데이터 파일 없음: {manual_path}")
        
        # TyDiP 데이터 체크
        tydip_path = "./data/ko_test.csv"
        if os.path.exists(tydip_path):
            tydip_df = pd.read_csv(tydip_path)
            print(f"✅ TyDiP 데이터: {len(tydip_df)}개 문장 로드 성공")
            print(f"   - 점수 범위: {tydip_df['score'].min():.1f} ~ {tydip_df['score'].max():.1f}")
            print(f"   - 평균 점수: {tydip_df['score'].mean():.2f}")
        else:
            print(f"❌ TyDiP 데이터 파일 없음: {tydip_path}")
            
    except Exception as e:
        print(f"❌ 데이터 로딩 오류: {e}")

def test_tokenizer():
    """토크나이저 테스트"""
    print("\n🔤 토크나이저 테스트...")
    
    try:
        tokenizer = BertTokenizer.from_pretrained('monologg/kobert')
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.unk_token
        
        test_sentence = "안녕하세요. 도움이 필요하시면 말씀해 주세요."
        
        encoded = tokenizer(
            test_sentence,
            truncation=True,
            padding='max_length',
            max_length=256,
            return_tensors='pt'
        )
        
        print(f"✅ 토크나이저 로드 성공")
        print(f"   - 테스트 문장: {test_sentence}")
        print(f"   - 토큰 수: {(encoded['attention_mask'] == 1).sum().item()}")
        print(f"   - Input shape: {encoded['input_ids'].shape}")
        
    except Exception as e:
        print(f"❌ 토크나이저 오류: {e}")

def test_model_architecture():
    """모델 아키텍처 테스트"""
    print("\n🧠 모델 아키텍처 테스트...")
    
    try:
        from multitask_model import ImprovedMultiTaskModel, KoPolitenessDataset
        
        # 모델 생성 (실제 가중치 로드 안함)
        model = ImprovedMultiTaskModel(
            model_name='monologg/kobert',
            dropout_rate=0.3
        )
        
        print(f"✅ 모델 생성 성공")
        print(f"   - 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")
        print(f"   - 피처 헤드 수: {len(model.feature_heads)}")
        
        # 더미 입력으로 순전파 테스트
        dummy_input_ids = torch.randint(0, 1000, (2, 256))  # batch_size=2, seq_len=256
        dummy_attention_mask = torch.ones(2, 256)
        
        with torch.no_grad():
            feature_logits, score_pred = model(dummy_input_ids, dummy_attention_mask)
        
        print(f"✅ 순전파 테스트 성공")
        print(f"   - 피처 출력 수: {len(feature_logits)}")
        print(f"   - 점수 출력 shape: {score_pred.shape}")
        
        # 피처별 출력 형태 확인
        for feat_name, logits in feature_logits.items():
            print(f"   - {feat_name}: {logits.shape}")
            
    except Exception as e:
        print(f"❌ 모델 아키텍처 오류: {e}")

def test_dataset():
    """데이터셋 클래스 테스트"""
    print("\n📊 데이터셋 클래스 테스트...")
    
    try:
        from multitask_model import KoPolitenessDataset
        from transformers import BertTokenizer
        
        tokenizer = BertTokenizer.from_pretrained('monologg/kobert')
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.unk_token
        
        # 더미 데이터
        sentences = ["안녕하세요.", "야, 뭐해?", "죄송합니다."]
        feat_labels = [[2, 1, 0, 0, 1, 2, 0], [0, 0, 0, 0, 0, 0, 0], [2, 1, 0, 0, 1, 2, 1]]
        scores = [1.5, -0.5, 1.0]
        
        # Manual 데이터셋 테스트
        dataset = KoPolitenessDataset(
            sentences=sentences,
            feat_labels=feat_labels,
            scores=scores,
            tokenizer=tokenizer,
            max_length=256
        )
        
        print(f"✅ Manual 데이터셋 생성 성공")
        print(f"   - 데이터 수: {len(dataset)}")
        
        # 샘플 확인
        sample = dataset[0]
        print(f"   - 샘플 키: {list(sample.keys())}")
        print(f"   - Input shape: {sample['input_ids'].shape}")
        print(f"   - Feature labels: {sample['feat_labels']}")
        print(f"   - Score: {sample['score'].item():.2f}")
        
        # TyDiP 데이터셋 테스트 (피처 라벨 없음)
        tydip_dataset = KoPolitenessDataset(
            sentences=sentences,
            feat_labels=None,
            scores=scores,
            tokenizer=tokenizer,
            max_length=256
        )
        
        tydip_sample = tydip_dataset[0]
        print(f"✅ TyDiP 데이터셋 생성 성공")
        print(f"   - Feature mask: {tydip_sample['feat_mask']}")  # 모두 False여야 함
        
    except Exception as e:
        print(f"❌ 데이터셋 클래스 오류: {e}")

def test_feature_distribution():
    """피처 분포 분석 테스트"""
    print("\n📈 피처 분포 분석 테스트...")
    
    try:
        manual_path = "./data/manual_labels.csv"
        if not os.path.exists(manual_path):
            print("❌ Manual 데이터 파일이 없어 분포 분석을 건너뜁니다.")
            return
            
        manual_df = pd.read_csv(manual_path)
        feature_cols = [
            "feat_ending", "feat_strat_cnt", "feat_command",
            "feat_attack", "feat_power", "feat_distance", "feat_indirect"
        ]
        
        print("✅ 피처별 클래스 분포:")
        
        for col in feature_cols:
            if col in manual_df.columns:
                counts = manual_df[col].value_counts().sort_index()
                total = len(manual_df)
                
                print(f"\n   {col}:")
                for class_val, count in counts.items():
                    percentage = (count / total) * 100
                    print(f"     클래스 {class_val}: {count:3d}개 ({percentage:5.1f}%)")
                
                # 희귀 클래스 경고
                rare_classes = counts[counts < 10].index.tolist()
                if rare_classes:
                    print(f"     ⚠️  희귀 클래스 (< 10개): {rare_classes}")
            else:
                print(f"   ❌ {col} 컬럼 없음")
                
    except Exception as e:
        print(f"❌ 피처 분포 분석 오류: {e}")

def main():
    """메인 테스트 실행"""
    print("🧪 한국어 공손도 예측 시스템 빠른 테스트")
    print("=" * 60)
    
    # GPU 확인
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  사용 디바이스: {device}")
    
    if torch.cuda.is_available():
        print(f"   - GPU: {torch.cuda.get_device_name(0)}")
        print(f"   - 메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    
    # 개별 테스트 실행
    test_data_loading()
    test_tokenizer()
    test_model_architecture()
    test_dataset()
    test_feature_distribution()
    
    print("\n" + "=" * 60)
    print("🎉 빠른 테스트 완료!")
    print("\n💡 다음 단계:")
    print("   1. python train_multitask.py - 모델 훈련 실행")
    print("   2. python inference_utils.py - 추론 테스트 실행")
    print("   3. 학습된 모델로 실제 문장 분석")

if __name__ == "__main__":
    main()
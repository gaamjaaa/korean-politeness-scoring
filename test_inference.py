#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from inference_utils import PolitenessAnalyzer

def main():
    """공손도 분석 테스트"""
    
    # 모델 설정
    config = {
        'model_name': 'monologg/kobert',
        'max_length': 256,
        'dropout_rate': 0.4,  # 훈련 시 사용한 값과 동일해야 함
        'normalize_scores': True
    }
    
    # 훈련된 모델 경로 (훈련 완료 후 생성됨)
    model_path = "./results/best_model_fold_0.pt"
    
    try:
        print("🚀 모델 로딩 중...")
        analyzer = PolitenessAnalyzer(model_path, config)
        
        print("\n" + "="*60)
        print("🤖 한국어 공손도 분석기")
        print("="*60)
        print("문장을 입력하면 공손도 점수를 알려드립니다!")
        print("종료하려면 'quit' 또는 '종료'를 입력하세요.")
        print("="*60)
        
        while True:
            # 사용자 입력
            sentence = input("\n📝 분석할 문장: ").strip()
            
            # 종료 조건
            if sentence.lower() in ['quit', 'exit', '종료', 'q']:
                print("👋 분석기를 종료합니다.")
                break
            
            if not sentence:
                print("❌ 문장을 입력해주세요.")
                continue
            
            try:
                print("\n" + "-"*50)
                
                # 공손도 분석
                result = analyzer.analyze_politeness(sentence, show_details=True)
                
                # 추가 해석
                score = result['predicted_score']
                print(f"\n💬 한 줄 요약:")
                if score >= 2.0:
                    print("   🌟 매우 공손하고 정중한 표현입니다!")
                elif score >= 1.0:
                    print("   😊 공손한 표현입니다.")
                elif score >= 0:
                    print("   😐 중립적인 표현입니다.")
                elif score >= -1.0:
                    print("   😕 다소 무례할 수 있는 표현입니다.")
                else:
                    print("   😠 매우 무례한 표현입니다.")
                
                # 개선 제안
                if score < 0:
                    print(f"\n💡 더 공손하게 표현하려면:")
                    suggestions = []
                    
                    # 피처별 개선 제안
                    features = result['feature_predictions']
                    
                    if features['feat_ending'] <= 1:
                        suggestions.append("- 존댓말 사용하기 (예: ~어요, ~습니다)")
                    
                    if features['feat_command'] >= 2:
                        suggestions.append("- 명령문보다는 요청문 사용하기 (예: ~해주세요)")
                    
                    if features['feat_attack'] >= 1:
                        suggestions.append("- 비판적 표현을 중립적으로 바꾸기")
                    
                    if features['feat_indirect'] == 0:
                        suggestions.append("- 간접적 표현 사용하기 (예: 혹시, 만약)")
                    
                    if not suggestions:
                        suggestions.append("- 전반적으로 더 정중한 어조 사용하기")
                    
                    for suggestion in suggestions:
                        print(f"   {suggestion}")
                
                print("-"*50)
                
            except Exception as e:
                print(f"❌ 분석 중 오류가 발생했습니다: {e}")
    
    except FileNotFoundError:
        print("❌ 훈련된 모델을 찾을 수 없습니다!")
        print("\n🔧 해결 방법:")
        print("1. 먼저 다음 명령어로 모델을 훈련하세요:")
        print("   python train_multitask.py")
        print("2. 훈련 완료 후 이 스크립트를 다시 실행하세요.")
        
    except Exception as e:
        print(f"❌ 예상치 못한 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    main() 
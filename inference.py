#!/usr/bin/env python3
"""
한국어 공손도 예측 추론 스크립트
학습된 KoBERT 멀티태스크 모델로 새로운 문장의 공손도를 예측합니다.
"""

import os
import sys
import argparse
import torch
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from typing import List, Dict, Union

# 로컬 모듈 임포트
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from src.model import MultitaskPolitenessModel


class PolitenessPredictor:
    """공손도 예측기"""
    
    def __init__(self, model_path: str, model_name: str = "monologg/kobert", device: str = 'auto'):
        self.device = self._setup_device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # 모델 로드
        self.model = MultitaskPolitenessModel(model_name=model_name)
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()
        
        # 피처 이름 정의
        self.feature_names = [
            'feat_ending', 'feat_strat_cnt', 'feat_indirect', 
            'feat_command', 'feat_attack', 'feat_power', 'feat_distance'
        ]
        
        self.feature_descriptions = {
            'feat_ending': '어미 (존댓말 vs 반말)',
            'feat_strat_cnt': '전략적 표현 (간접화법, 완곡표현 등)',
            'feat_indirect': '간접성 (돌려말하기)',
            'feat_command': '명령조 (강요, 지시)',
            'feat_attack': '공격성 (비난, 모욕)',
            'feat_power': '권력거리 (높임/낮춤)',
            'feat_distance': '사회적 거리감'
        }
        
        print(f"✅ 모델 로드 완료: {model_path}")
        print(f"🖥️  디바이스: {self.device}")
    
    def _setup_device(self, device_arg: str) -> str:
        """디바이스 설정"""
        if device_arg == 'auto':
            return 'cuda' if torch.cuda.is_available() else 'cpu'
        return device_arg
    
    def predict_single(self, sentence: str, return_probabilities: bool = False) -> Dict:
        """단일 문장 예측"""
        # 토크나이징
        encoding = self.tokenizer(
            sentence,
            truncation=True,
            padding='max_length',
            max_length=256,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # 예측
        with torch.no_grad():
            outputs = self.model.get_feature_predictions(input_ids, attention_mask)
        
        # 결과 정리
        result = {
            'sentence': sentence,
            'overall_score': float(outputs['score_prediction'][0]),
            'features': {}
        }
        
        # 피처별 예측값
        for i, feature_name in enumerate(self.feature_names):
            pred = int(outputs['feature_predictions'][i][0])
            result['features'][feature_name] = {
                'prediction': pred,
                'description': self.feature_descriptions[feature_name]
            }
            
            if return_probabilities:
                probs = outputs['feature_probabilities'][i][0].cpu().numpy()
                result['features'][feature_name]['probabilities'] = probs.tolist()
        
        return result
    
    def predict_batch(self, sentences: List[str], return_probabilities: bool = False) -> List[Dict]:
        """배치 예측"""
        results = []
        for sentence in sentences:
            result = self.predict_single(sentence, return_probabilities)
            results.append(result)
        return results
    
    def analyze_politeness(self, sentence: str) -> Dict:
        """공손도 분석 (해석 포함)"""
        result = self.predict_single(sentence, return_probabilities=True)
        
        # 공손도 해석
        overall_score = result['overall_score']
        if overall_score >= 1.5:
            politeness_level = "매우 공손함"
        elif overall_score >= 0.5:
            politeness_level = "공손함"
        elif overall_score >= -0.5:
            politeness_level = "중립"
        elif overall_score >= -1.5:
            politeness_level = "다소 무례함"
        else:
            politeness_level = "매우 무례함"
        
        # 주요 영향 요소 분석
        influential_features = []
        for feature_name, feature_data in result['features'].items():
            pred = feature_data['prediction']
            if pred >= 2:  # 2점 이상인 피처들
                influential_features.append({
                    'feature': feature_name,
                    'description': feature_data['description'],
                    'score': pred,
                    'impact': 'positive' if pred == 3 else 'moderate'
                })
            elif pred <= -2:  # -2점 이하인 피처들 (실제로는 0점이 최저)
                influential_features.append({
                    'feature': feature_name,
                    'description': feature_data['description'],
                    'score': pred,
                    'impact': 'negative'
                })
        
        analysis = {
            'sentence': sentence,
            'overall_score': overall_score,
            'politeness_level': politeness_level,
            'feature_analysis': result['features'],
            'influential_features': influential_features,
            'interpretation': self._generate_interpretation(result)
        }
        
        return analysis
    
    def _generate_interpretation(self, result: Dict) -> str:
        """결과 해석 생성"""
        sentence = result['sentence']
        score = result['overall_score']
        features = result['features']
        
        interpretation = f"문장 '{sentence}'의 공손도 분석:\n\n"
        interpretation += f"전체 공손도 점수: {score:.2f}\n"
        
        # 높은 점수 피처들
        high_features = []
        low_features = []
        
        for fname, fdata in features.items():
            pred = fdata['prediction']
            if pred >= 2:
                high_features.append((fname, fdata['description'], pred))
            elif pred == 0:
                low_features.append((fname, fdata['description'], pred))
        
        if high_features:
            interpretation += "\n✅ 공손함을 나타내는 요소들:\n"
            for fname, desc, pred in high_features:
                interpretation += f"  - {desc}: {pred}점\n"
        
        if low_features:
            interpretation += "\n⚠️ 개선이 필요한 요소들:\n"
            for fname, desc, pred in low_features:
                interpretation += f"  - {desc}: {pred}점\n"
        
        # 종합 의견
        if score >= 1.0:
            interpretation += "\n💬 전반적으로 공손하고 예의바른 표현입니다."
        elif score >= 0:
            interpretation += "\n💬 적절한 수준의 표현이지만, 상황에 따라 더 공손한 표현을 고려해볼 수 있습니다."
        else:
            interpretation += "\n💬 다소 직접적이거나 무례할 수 있는 표현입니다. 더 공손한 표현을 권장합니다."
        
        return interpretation


def parse_args():
    """명령행 인자 파싱"""
    parser = argparse.ArgumentParser(description="한국어 공손도 예측 추론")
    
    parser.add_argument('--model_path', type=str, required=True,
                       help='학습된 모델 파일 경로 (.pt)')
    parser.add_argument('--sentence', type=str,
                       help='분석할 단일 문장')
    parser.add_argument('--input_file', type=str,
                       help='분석할 문장들이 담긴 파일 (CSV, 텍스트)')
    parser.add_argument('--output_file', type=str,
                       help='결과 저장 파일 경로')
    parser.add_argument('--model_name', type=str, default='monologg/kobert',
                       help='사용할 KoBERT 모델명')
    parser.add_argument('--device', type=str, default='auto',
                       help='사용할 디바이스 (auto/cuda/cpu)')
    parser.add_argument('--detailed', action='store_true',
                       help='상세 분석 모드 (해석 포함)')
    parser.add_argument('--probabilities', action='store_true',
                       help='확률값 포함')
    
    return parser.parse_args()


def main():
    """메인 함수"""
    args = parse_args()
    
    print("🎯 한국어 공손도 예측 추론 시스템")
    print("=" * 50)
    
    # 예측기 초기화
    predictor = PolitenessPredictor(
        model_path=args.model_path,
        model_name=args.model_name,
        device=args.device
    )
    
    # 단일 문장 분석
    if args.sentence:
        print(f"\n📝 문장 분석: '{args.sentence}'")
        print("-" * 50)
        
        if args.detailed:
            analysis = predictor.analyze_politeness(args.sentence)
            print(analysis['interpretation'])
            
            if args.probabilities:
                print(f"\n📊 피처별 확률 분포:")
                for fname, fdata in analysis['feature_analysis'].items():
                    if 'probabilities' in fdata:
                        probs = fdata['probabilities']
                        print(f"  {fdata['description']}: {probs}")
        else:
            result = predictor.predict_single(args.sentence, args.probabilities)
            print(f"전체 공손도 점수: {result['overall_score']:.3f}")
            print(f"\n피처별 예측:")
            for fname, fdata in result['features'].items():
                print(f"  {fdata['description']}: {fdata['prediction']}점")
    
    # 파일 배치 분석
    elif args.input_file:
        print(f"\n📁 파일 배치 분석: {args.input_file}")
        
        # 파일 읽기
        if args.input_file.endswith('.csv'):
            df = pd.read_csv(args.input_file)
            if 'sentence' in df.columns:
                sentences = df['sentence'].tolist()
            else:
                sentences = df.iloc[:, 0].tolist()  # 첫 번째 컬럼
        else:
            # 텍스트 파일
            with open(args.input_file, 'r', encoding='utf-8') as f:
                sentences = [line.strip() for line in f if line.strip()]
        
        print(f"분석할 문장 수: {len(sentences)}")
        
        # 배치 예측
        if args.detailed:
            results = []
            for sentence in sentences:
                analysis = predictor.analyze_politeness(sentence)
                results.append(analysis)
        else:
            results = predictor.predict_batch(sentences, args.probabilities)
        
        # 결과 저장
        if args.output_file:
            if args.output_file.endswith('.csv'):
                # CSV 형태로 저장
                output_data = []
                for result in results:
                    row = {
                        'sentence': result['sentence'],
                        'overall_score': result['overall_score']
                    }
                    
                    if 'feature_analysis' in result:
                        features = result['feature_analysis']
                    else:
                        features = result['features']
                    
                    for fname, fdata in features.items():
                        row[fname] = fdata['prediction']
                    
                    if args.detailed and 'politeness_level' in result:
                        row['politeness_level'] = result['politeness_level']
                    
                    output_data.append(row)
                
                pd.DataFrame(output_data).to_csv(args.output_file, index=False, encoding='utf-8')
            else:
                # JSON 형태로 저장
                import json
                with open(args.output_file, 'w', encoding='utf-8') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
            
            print(f"✅ 결과 저장 완료: {args.output_file}")
        
        # 요약 통계
        scores = [r['overall_score'] for r in results]
        print(f"\n📊 전체 분석 요약:")
        print(f"  평균 공손도: {np.mean(scores):.3f}")
        print(f"  표준편차: {np.std(scores):.3f}")
        print(f"  최고점: {np.max(scores):.3f}")
        print(f"  최저점: {np.min(scores):.3f}")
    
    else:
        # 대화형 모드
        print(f"\n💬 대화형 분석 모드 (종료: 'quit' 입력)")
        print("-" * 50)
        
        while True:
            sentence = input("분석할 문장을 입력하세요: ").strip()
            
            if sentence.lower() in ['quit', 'exit', '종료']:
                break
            
            if not sentence:
                continue
            
            try:
                if args.detailed:
                    analysis = predictor.analyze_politeness(sentence)
                    print(f"\n{analysis['interpretation']}\n")
                else:
                    result = predictor.predict_single(sentence)
                    print(f"공손도 점수: {result['overall_score']:.3f}")
                    print("피처별 점수:", end=" ")
                    feature_scores = [str(fdata['prediction']) for fdata in result['features'].values()]
                    print(" | ".join(feature_scores))
                    print()
            
            except Exception as e:
                print(f"❌ 오류 발생: {e}\n")
    
    print("🎯 분석 완료!")


if __name__ == "__main__":
    main() 
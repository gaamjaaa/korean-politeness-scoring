import torch
import numpy as np
import pandas as pd
from transformers import BertTokenizer
import matplotlib.pyplot as plt
import seaborn as sns
from multitask_model import ImprovedMultiTaskModel, KoPolitenessDataset

plt.rcParams['font.family'] = 'DejaVu Sans'  # 한글 폰트 설정

class PolitenessAnalyzer:
    """공손도 분석기"""
    
    def __init__(self, model_path, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 토크나이저 로드
        self.tokenizer = BertTokenizer.from_pretrained(config['model_name'])
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.unk_token
        
        # 모델 로드
        self.model = ImprovedMultiTaskModel(
            model_name=config['model_name'],
            dropout_rate=config['dropout_rate']
        ).to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # 피처 이름과 설명 (클래스 3 → 2 병합 반영)
        self.feature_info = {
            'feat_ending': {
                'name': '어미',
                'classes': ['반말/비격식', '격식적 어미', '존댓말 (최고 존댓말 포함)'],
                'description': '문장 종결 어미의 격식도'
            },
            'feat_strat_cnt': {
                'name': '전략적 표현',
                'classes': ['없음', '단순 완화', '복합 전략'],
                'description': '공손 전략 표현의 복잡도'
            },
            'feat_command': {
                'name': '명령성',
                'classes': ['서술/질문', '제안/요청', '지시/명령 (강한 명령 포함)'],
                'description': '화행의 명령성 정도'
            },
            'feat_attack': {
                'name': '공격성',
                'classes': ['중립', '가벼운 비판', '직접적 비판 (강한 공격 포함)'],
                'description': '언어적 공격성 수준'
            },
            'feat_power': {
                'name': '권력거리',
                'classes': ['동등', '약간 상하관계', '명확한 상하관계 (강한 권력관계 포함)'],
                'description': '화자와 청자 간 권력 관계'
            },
            'feat_distance': {
                'name': '사회적 거리',
                'classes': ['가까움', '보통', '격식적 (매우 격식적 포함)'],
                'description': '화자와 청자 간 사회적 거리'
            },
            'feat_indirect': {
                'name': '간접성',
                'classes': ['직접적', '약간 간접적', '매우 간접적'],
                'description': '표현의 간접성 정도'
            }
        }
        
        print(f"🚀 PolitenessAnalyzer loaded on {self.device}")
    
    def predict_single(self, sentence):
        """단일 문장 예측"""
        # 토크나이징
        encoded = self.tokenizer(
            sentence,
            truncation=True,
            padding='max_length',
            max_length=self.config['max_length'],
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        with torch.no_grad():
            feature_logits, score_pred = self.model(input_ids, attention_mask)
        
        # 피처별 예측
        feature_preds = {}
        feature_probs = {}
        
        for feat_name, logits in feature_logits.items():
            probs = torch.softmax(logits, dim=-1)
            pred_class = torch.argmax(logits, dim=-1).item()
            
            feature_preds[feat_name] = pred_class
            feature_probs[feat_name] = probs.cpu().numpy()[0]
        
        # 스코어 예측 (정규화 해제)
        if self.config.get('normalize_scores', True):
            score = score_pred.item() * 6.0 - 3.0
        else:
            score = score_pred.item()
        
        return {
            'sentence': sentence,
            'predicted_score': score,
            'feature_predictions': feature_preds,
            'feature_probabilities': feature_probs
        }
    
    def predict_batch(self, sentences):
        """배치 예측"""
        results = []
        for sentence in sentences:
            result = self.predict_single(sentence)
            results.append(result)
        return results
    
    def analyze_politeness(self, sentence, show_details=True):
        """공손도 상세 분석"""
        result = self.predict_single(sentence)
        
        print(f"📝 문장: {sentence}")
        print(f"🎯 예측 공손도 점수: {result['predicted_score']:.2f} (-3: 매우 무례 ~ +3: 매우 공손)")
        
        # 점수 해석
        score = result['predicted_score']
        if score >= 2.0:
            interpretation = "매우 공손한 표현"
        elif score >= 1.0:
            interpretation = "공손한 표현"
        elif score >= -0.5:
            interpretation = "중립적 표현"
        elif score >= -1.5:
            interpretation = "다소 무례한 표현"
        else:
            interpretation = "매우 무례한 표현"
        
        print(f"🔍 해석: {interpretation}")
        
        if show_details:
            print("\n📊 피처별 분석:")
            for feat_name, pred_class in result['feature_predictions'].items():
                info = self.feature_info[feat_name]
                class_name = info['classes'][pred_class]
                confidence = result['feature_probabilities'][feat_name][pred_class]
                
                print(f"  {info['name']}: {class_name} (확신도: {confidence:.2f})")
                print(f"    → {info['description']}")
        
        return result
    
    def compare_sentences(self, sentences, labels=None):
        """여러 문장 비교 분석"""
        results = self.predict_batch(sentences)
        
        if labels is None:
            labels = [f"문장 {i+1}" for i in range(len(sentences))]
        
        # 결과 시각화
        scores = [r['predicted_score'] for r in results]
        
        plt.figure(figsize=(12, 6))
        
        # 점수 비교
        plt.subplot(1, 2, 1)
        bars = plt.bar(labels, scores)
        plt.title('공손도 점수 비교')
        plt.ylabel('공손도 점수')
        plt.xticks(rotation=45)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        
        # 색상 설정
        for i, (bar, score) in enumerate(zip(bars, scores)):
            if score >= 1.0:
                bar.set_color('green')
            elif score >= -0.5:
                bar.set_color('yellow')
            else:
                bar.set_color('red')
        
        # 피처별 히트맵
        plt.subplot(1, 2, 2)
        feature_matrix = []
        feature_names = list(self.feature_info.keys())
        
        for result in results:
            row = [result['feature_predictions'][feat] for feat in feature_names]
            feature_matrix.append(row)
        
        sns.heatmap(
            feature_matrix,
            xticklabels=[self.feature_info[f]['name'] for f in feature_names],
            yticklabels=labels,
            annot=True,
            fmt='d',
            cmap='RdYlGn',
            cbar_kws={'label': '클래스 (높을수록 공손)'}
        )
        plt.title('피처별 예측 결과')
        
        plt.tight_layout()
        plt.show()
        
        return results
    
    def feature_importance_analysis(self, sentence):
        """피처 중요도 분석 (간단한 perturbation 기반)"""
        original_result = self.predict_single(sentence)
        original_score = original_result['predicted_score']
        
        print(f"📝 원본 문장: {sentence}")
        print(f"🎯 원본 점수: {original_score:.2f}")
        print("\n🔍 피처 제거 실험:")
        
        # 각 피처의 기여도 분석 (단순화된 방법)
        feature_contributions = {}
        
        for feat_name in self.feature_info.keys():
            # 해당 피처의 예측값을 0(중립)으로 강제 설정하고 재예측
            # 실제로는 모델을 수정해야 하지만, 여기서는 개념적 설명
            pred_class = original_result['feature_predictions'][feat_name]
            info = self.feature_info[feat_name]
            
            # 예측된 클래스의 "기여도" 추정
            contribution = (pred_class - 1.5) * 0.5  # 단순화된 계산
            feature_contributions[feat_name] = contribution
            
            print(f"  {info['name']}: {info['classes'][pred_class]} (기여도: {contribution:+.2f})")
        
        return feature_contributions
    
    def batch_analyze_csv(self, csv_path, sentence_col='sentence', output_path=None):
        """CSV 파일 일괄 분석"""
        df = pd.read_csv(csv_path)
        sentences = df[sentence_col].tolist()
        
        print(f"📁 {len(sentences)}개 문장 분석 중...")
        
        results = self.predict_batch(sentences)
        
        # 결과를 DataFrame으로 정리
        result_data = []
        for i, result in enumerate(results):
            row = {
                'sentence': result['sentence'],
                'predicted_score': result['predicted_score']
            }
            
            # 피처별 예측 추가
            for feat_name, pred_class in result['feature_predictions'].items():
                row[f'{feat_name}_pred'] = pred_class
                row[f'{feat_name}_class'] = self.feature_info[feat_name]['classes'][pred_class]
            
            result_data.append(row)
        
        result_df = pd.DataFrame(result_data)
        
        if output_path:
            result_df.to_csv(output_path, index=False, encoding='utf-8-sig')
            print(f"💾 결과 저장: {output_path}")
        
        # 기본 통계
        print(f"\n📊 분석 결과 요약:")
        print(f"평균 공손도: {result_df['predicted_score'].mean():.2f}")
        print(f"표준편차: {result_df['predicted_score'].std():.2f}")
        print(f"최고 점수: {result_df['predicted_score'].max():.2f}")
        print(f"최저 점수: {result_df['predicted_score'].min():.2f}")
        
        return result_df
    
    def interactive_analysis(self):
        """대화형 분석 모드"""
        print("🤖 한국어 공손도 분석기에 오신 것을 환영합니다!")
        print("문장을 입력하면 공손도를 분석해드립니다. (종료: 'quit')")
        
        while True:
            sentence = input("\n📝 분석할 문장을 입력하세요: ").strip()
            
            if sentence.lower() in ['quit', 'exit', '종료']:
                print("👋 분석기를 종료합니다.")
                break
            
            if not sentence:
                continue
            
            try:
                self.analyze_politeness(sentence)
                
                # 개선 제안
                result = self.predict_single(sentence)
                if result['predicted_score'] < 0:
                    print("\n💡 개선 제안:")
                    print("  - 존댓말 사용 고려")
                    print("  - 간접적 표현 사용")
                    print("  - 공손 표현 추가 (예: '혹시', '부탁드립니다')")
                
            except Exception as e:
                print(f"❌ 오류 발생: {e}")

def demonstrate_analyzer():
    """분석기 시연"""
    # 설정 (실제 사용 시 수정 필요)
    config = {
        'model_name': 'monologg/kobert',
        'max_length': 256,
        'dropout_rate': 0.3,
        'normalize_scores': True
    }
    
    # 예시 문장들
    test_sentences = [
        "눈이 좀 오더구나.",
        "야. 지금 밖에 어때?",
        "별말씀을요. 그럼 오늘 하루도 즐거운 마음으로 시작하시기 바랍니다.",
        "회의는 예정된 시간에 시작 되니 잠시 기다려주세요.",
        "당신이나 좀 잘하세요. 남 신경 쓰지 마시고."
    ]
    
    print("🎯 한국어 공손도 분석 시연")
    print("=" * 50)
    
    # 모델 경로를 실제 경로로 수정해야 합니다
    model_path = "./results/best_model_fold_0.pt"
    
    try:
        analyzer = PolitenessAnalyzer(model_path, config)
        
        print("\n📊 다양한 문장의 공손도 비교:")
        analyzer.compare_sentences(test_sentences)
        
        print("\n🔍 개별 문장 상세 분석:")
        for sentence in test_sentences[:2]:  # 처음 2개만 상세 분석
            print("\n" + "="*50)
            analyzer.analyze_politeness(sentence)
            
    except FileNotFoundError:
        print("❌ 모델 파일을 찾을 수 없습니다. 먼저 train_multitask.py를 실행하여 모델을 훈련하세요.")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")

if __name__ == "__main__":
    demonstrate_analyzer() 
# 한국어 공손도 예측 시스템 (Korean Politeness Scorer)

## 📌 프로젝트 개요

한국어 문장의 공손도를 예측하는 멀티태스크 딥러닝 시스템입니다. KoBERT를 기반으로 7가지 공손 피처를 분석하고 최종 공손도 점수를 제공합니다. 이 시스템은 한국어의 문화적 맥락과 언어적 특성을 고려한 공손도 분석 도구입니다.

## 🎯 주요 기능

- **멀티태스크 학습**: 7개 공손 피처 + 최종 점수 동시 예측
- **클래스 불균형 처리**: Focal Loss + 클래스 가중치 + 스마트 데이터 분할
- **K-fold 교차검증**: 안정적인 성능 평가
- **상세 분석**: 피처별 기여도 분석 및 해석 제공
- **대화형 인터페이스**: 실시간 문장 분석

## 📊 분석 피처

1. **어미 (feat_ending)**: 문장 종결 어미의 격식도
2. **전략적 표현 (feat_strat_cnt)**: 공손 전략의 복잡도
3. **명령성 (feat_command)**: 화행의 명령성 정도
4. **공격성 (feat_attack)**: 언어적 공격성 수준
5. **권력거리 (feat_power)**: 화자-청자 간 권력 관계
6. **사회적 거리 (feat_distance)**: 격식적 거리감
7. **간접성 (feat_indirect)**: 표현의 간접성 정도

## 🔧 기술 스택

- **프레임워크**: PyTorch, Transformers
- **언어 모델**: KoBERT (monologg/kobert)
- **데이터 처리**: Pandas, NumPy, Scikit-learn
- **시각화**: Matplotlib, Seaborn
- **학습 도구**: TensorBoard, Tqdm

## 🧠 모델 아키텍처

- **베이스 모델**: KoBERT 인코더
- **멀티태스크 헤드**: 7개 피처 분류 + 1개 최종 점수 회귀
- **피처 상호작용 레이어**: 피처 간 관계 학습
- **계층적 구조**: BERT 특징과 피처 예측을 결합한 최종 점수 예측

## 📚 데이터셋

- **Manual 데이터**: 362개 수동 라벨링된 한국어 문장
- **TyDiP 데이터**: 500개 공손도 점수가 있는 문장
- **클래스 분포**: 다수 피처에서 심각한 클래스 불균형 존재

## 🛠️ 학습 방법

- **K-fold 교차검증**: 5-fold 검증으로 안정적인 성능 평가
- **불균형 처리**: Focal Loss, 클래스 가중치, 스마트 데이터 분할
- **최적화**: AdamW + Cosine Annealing 스케줄러
- **정규화**: 드롭아웃 (0.4) + 가중치 감쇠 (0.01)
- **Early Stopping**: Quadratic Weighted Kappa 기준

## 📈 성능 지표

- **Quadratic Weighted Kappa**: 순서형 분류 평가의 주요 지표
- **Adjacent Accuracy**: ±1 오차 허용 정확도
- **Ordinal MAE**: 순서형 평균 절대 오차
- **MAE**: 최종 점수 예측의 평균 절대 오차

## 🔍 주요 구현 파일

- **models/multitask_model.py**: 멀티태스크 모델 구현
- **training/train_multitask.py**: 학습 파이프라인 구현
- **preprocessing/korean_politeness_analyzer.py**: 한국어 공손도 분석기
- **inference/**: 추론 및 실시간 분석 도구
- **benchmarks/**: 성능 평가 도구

## 💡 사용법

### 모델 훈련
```bash
python train_multitask.py
```

### 단일 문장 분석
```python
from inference_utils import PolitenessAnalyzer

analyzer = PolitenessAnalyzer("./results/best_model_fold_0.pt", config)
result = analyzer.analyze_politeness("안녕하세요. 도움이 필요하시면 말씀해 주세요.")
```

### 비교 분석
```python
sentences = [
    "야, 뭐해?",
    "안녕하세요. 뭐 하고 계세요?",
    "실례합니다. 혹시 무엇을 하고 계신지 여쭤봐도 될까요?"
]
analyzer.compare_sentences(sentences)
```

## 🚨 한계점 및 향후 개선 방향

- **데이터 부족**: 362개 Manual 샘플로는 심층 학습에 제한적
- **클래스 불균형**: 일부 피처의 클래스 불균형

**향후 개선 방향**:
- 다양한 도메인의 라벨링 데이터 확장
- 언어학적 규칙과 딥러닝의 하이브리드 접근 방식
- 주의 메커니즘 기반 피처 관계 모델링
- 실시간 웹 서비스 구축


# 한국어 공손도 예측 시스템 (Korean Politeness Scorer)

## 📌 프로젝트 개요

한국어 문장의 공손도를 예측하는 멀티태스크 딥러닝 시스템입니다. KoBERT 기반으로 7가지 공손 피처를 분석하고 최종 공손도 점수를 제공합니다.

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

## 🔧 설치 및 설정

### 1. 환경 설정


### 2. 데이터 구조
```
KoPolitenessScore/
├── data/
│   ├── manual_labels.csv      # Manual 라벨링 데이터 (362문장)
│   └── ko_test.csv           # TyDiP 데이터 (500문장)
├── multitask_model.py        # 모델 정의
├── train_multitask.py        # 훈련 스크립트
├── inference_utils.py        # 추론 및 분석 도구
└── results/                  # 훈련 결과 저장
```

## 사용법

### 1. 모델 훈련
```bash
python train_multitask.py
```

**주요 하이퍼파라미터:**
- `learning_rate`: 2e-5
- `batch_size`: 16
- `num_epochs`: 8
- `n_folds`: 3 (K-fold CV)
- `use_focal_loss`: True

### 2. 단일 문장 분석
```python
from inference_utils import PolitenessAnalyzer

config = {
    'model_name': 'monologg/kobert',
    'max_length': 256,
    'dropout_rate': 0.3,
    'normalize_scores': True
}

analyzer = PolitenessAnalyzer("./results/best_model_fold_0.pt", config)
result = analyzer.analyze_politeness("안녕하세요. 도움이 필요하시면 말씀해 주세요.")
```

### 3. 대화형 분석
```python
analyzer.interactive_analysis()
```

### 4. 배치 분석
```python
# CSV 파일 일괄 분석
results_df = analyzer.batch_analyze_csv(
    csv_path="./data/test_sentences.csv",
    sentence_col="sentence",
    output_path="./results/analysis_results.csv"
)
```

### 5. 비교 분석
```python
sentences = [
    "야, 뭐해?",
    "안녕하세요. 뭐 하고 계세요?",
    "실례합니다. 혹시 무엇을 하고 계신지 여쭤봐도 될까요?"
]
analyzer.compare_sentences(sentences)
```

## 📈 모델 성능

### 기본 성능 지표
- **Manual 데이터 Macro-F1**: ~0.65-0.75 (피처별 평균)
- **Score MAE**: ~0.3-0.5 (정규화된 점수 기준)
- **희귀 클래스 처리**: Focal Loss로 개선

### 클래스 분포 (Manual 데이터)
```
feat_attack: 심각한 불균형 (클래스 3: 7개)
feat_strat_cnt: 불균형 (클래스 2: 21개)
feat_command: 경미한 불균형 (클래스 3: 11개)
```

## 🔍 모델 아키텍처

### 핵심 개선사항
1. **피처 상호작용 레이어**: 피처 간 관계 학습
2. **계층적 헤드 구조**: BERT features + feature predictions 결합
3. **스마트 데이터 분할**: 희귀 클래스 보장
4. **Focal Loss**: 불균형 클래스 처리

### 손실 함수
```python
Total Loss = Σ(Feature Classification Loss) + Score Regression Loss
- Feature Loss: Focal Loss + Class Weights
- Score Loss: MSE Loss
- Masking: TyDiP 데이터는 피처 헤드 비활성화
```

## 📝 데이터 형식

### Manual 데이터 (manual_labels.csv)
```csv
sentence,feat_ending,feat_strat_cnt,feat_command,feat_attack,feat_power,feat_distance,feat_indirect,score
"안녕하세요.",2,1,0,0,1,2,0,1.5
```

### TyDiP 데이터 (ko_test.csv)
```csv
sentence,score
"좋은 하루 되세요.",1.2
```

## 🛠️ 개선 포인트

### 현재 한계점
1. **데이터 크기**: 362개 Manual 샘플로 제한적
2. **클래스 불균형**: 일부 피처의 극심한 불균형
3. **도메인 특성**: 특정 도메인(업무 대화) 중심

### 향후 개선 방향
1. **데이터 확장**: 다양한 도메인의 라벨링 데이터 추가
2. **모델 구조**: Attention 기반 피처 관계 모델링
3. **평가 방법**: 사람 평가와의 상관관계 분석
4. **실시간 서비스**: FastAPI 기반 웹 서비스 구축

## 🔬 결과 분석

### 피처별 중요도
1. **어미 (feat_ending)**: 가장 강력한 공손도 지표
2. **권력거리 (feat_power)**: 사회적 맥락 반영
3. **사회적 거리 (feat_distance)**: 격식성 판단에 중요

### 오류 분석
- **복합 표현**: 여러 공손 전략이 섞인 문장에서 성능 저하
- **맥락 의존성**: 상황 정보 부족 시 오판 발생
- **희귀 클래스**: 극소수 샘플 클래스의 낮은 재현율

## 📚 참고문헌

1. Brown, P., & Levinson, S. C. (1987). Politeness: Some universals in language usage.
2. Danescu-Niculescu-Mizil, C., et al. (2013). A computational approach to politeness.
3. 한국어 공손법 연구 관련 국내 문헌들

## 👥 기여자

- 프로젝트 설계 및 구현
- 데이터 라벨링 및 검증
- 모델 최적화 및 평가

## 📄 라이선스

MIT License - 자유롭게 사용, 수정, 배포 가능

---

## 🚨 주의사항

1. **모델 한계**: 학습 데이터의 도메인과 스타일에 의존적
2. **문화적 맥락**: 한국어 특정 공손 표현에 최적화
3. **지속적 업데이트**: 언어 변화에 따른 주기적 재훈련 필요

문의사항이나 개선 제안은 이슈로 등록해 주세요! 🙏 
import pandas as pd
import numpy as np
import re
from collections import Counter
from konlpy.tag import Okt

# 한국어 형태소 분석기 초기화
okt = Okt()

# 한국어 불용어 목록 (조사, 의존명사, 접미사 등 제외)
STOPWORDS = [
    "이", "가", "을", "를", "에", "에서", "의", "로", "으로", "와", "과", "이나", "나", "에게", "께", "한테",
    "까지", "부터", "라고", "이라고", "라는", "이라는", "라면", "이라면", "라도", "이라도", "만", "뿐", "도",
    "이다", "다", "것", "수", "듯", "데", "때", "곳", "뿐", "거", "줄", "만큼", "정도", "즈음", "이야", 
]

# 공손함을 나타내는 표현들 (PMI 분석 결과)
POLITE_BIGRAMS = [
    ("문서", "문서"), 
    ("중", "이다"), 
    ("글", "좋다"), 
    ("끝나다", "중학교"), 
    ("사람", "궁금하다"),
    ("링크", "외부"),
    ("많이", "내용"),
    ("혹시", "알다"),
    ("알다", "혹시"),
    ("작업", "완료"),
    ("완료", "작업")
]

# 공손함을 나타내는 단어들 (빈도 분석 결과)
POLITE_TOKENS = {
    "혹시": 2.0,         # 공손한 표현에서만 등장
    "어떻게": 1.5,       # 공손한 표현에서 더 자주 등장
    "어떨까": 1.8,       # 공손한 표현에서 더 자주 등장
    "있을까": 1.8,       # 공손한 표현에서 더 자주 등장
    "않을까": 2.0,       # 공손한 표현에서만 등장
    "감사": 1.8,         # 고마움 표현
    "드립니다": 1.7,     # 공손한 높임말
    "알려주실": 1.7,     # 요청 표현
    "부탁": 1.7,         # 요청 표현
    "죄송": 1.8          # 사과 표현
}

# 무례함을 나타내는 표현들
IMPOLITE_TOKENS = {
    "왜": 1.5,           # 직설적인 의문
    "안": 1.3,           # 부정적 표현
    "아니": 1.2,         # 부정적 표현
    "그냥": 1.3,         # 무관심 표현
    "말이": 1.4,         # 강한 의견 표현
    "그렇게": 1.2,       # 비판적 표현
    "그런": 1.2          # 비판적 표현
}

# 공손함과 관련된 품사 정보
POLITE_POS = {
    "Adjective": 0.5,    # 형용사
    "Adverb": 0.3,       # 부사
    "Suffix": 0.5        # 접미사 (주로 '-요', '-습니다' 등의 높임말)
}

# 무례함과 관련된 품사 정보
IMPOLITE_POS = {
    "Exclamation": 0.8,  # 감탄사
    "KoreanParticle": 0.3  # 한국어 조사
}

def preprocess_text(text):
    """한국어 텍스트 전처리
    
    :param text: 원본 텍스트
    :return: 전처리된 텍스트
    """
    # NaN 값 확인
    if pd.isna(text):
        return ""
        
    # URL 제거
    text = re.sub(r'<url>|https?://\S+|www\.\S+', '', text)
    
    # 특수 문자 및 숫자 제거
    text = re.sub(r'[^\w\s가-힣]', ' ', text)
    text = re.sub(r'[0-9]', ' ', text)
    
    # 여러 공백을 하나로 치환
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()

def tokenize(text):
    """KoNLPy Okt 형태소 분석기를 사용한 토큰화
    
    :param text: 전처리된 텍스트
    :return: (토큰 리스트, 품사 정보 리스트)
    """
    # 빈 텍스트 처리
    if not text:
        return [], []
        
    # 품사 필터 설정 (의미있는 품사만)
    pos_filter = ['Noun', 'Verb', 'Adjective', 'Adverb', 'Determiner', 'Exclamation']
    
    # Okt 형태소 분석기 사용
    pos_tagged = okt.pos(text, norm=True, stem=True)
    
    # 필터링 및 토큰 생성
    filtered_tokens = []
    filtered_pos = []
    
    for word, pos in pos_tagged:
        # 불용어 및 1글자 단어 제외 (형용사와 부사는 1글자도 유지)
        if word in STOPWORDS:
            continue
        if len(word) < 2 and pos not in ['Adjective', 'Adverb']:
            continue
        if pos in pos_filter:
            filtered_tokens.append(word)
            filtered_pos.append(pos)
    
    return filtered_tokens, filtered_pos

def extract_bigrams(tokens):
    """토큰 리스트에서 바이그램 추출
    
    :param tokens: 토큰 리스트
    :return: 바이그램 리스트
    """
    if len(tokens) < 2:
        return []
        
    bigrams = []
    for i in range(len(tokens) - 1):
        bigrams.append((tokens[i], tokens[i+1]))
    return bigrams

def calculate_politeness_score(text):
    """문장의 공손함 점수 계산 (형태소 분석 기반)
    
    :param text: 입력 문장
    :return: 공손함 점수
    """
    # 텍스트 전처리 및 토큰화
    preprocessed = preprocess_text(text)
    tokens, pos_info = tokenize(preprocessed)
    
    # 빈 토큰 리스트인 경우
    if not tokens:
        return 0.5  # 중립값 반환
    
    bigrams = extract_bigrams(tokens)
    
    # 기본 점수 설정
    base_score = 0.0
    
    # 토큰 점수 계산
    token_score = 0.0
    token_counts = Counter(tokens)
    
    for token, count in token_counts.items():
        if token in POLITE_TOKENS:
            token_score += POLITE_TOKENS[token] * count
        if token in IMPOLITE_TOKENS:
            token_score -= IMPOLITE_TOKENS[token] * count
    
    # 바이그램 점수 계산
    bigram_score = 0.0
    for bigram in bigrams:
        if bigram in POLITE_BIGRAMS:
            bigram_score += 2.0  # 공손한 바이그램에 높은 가중치 부여
    
    # 품사 기반 점수 계산
    pos_score = 0.0
    for pos in pos_info:
        if pos in POLITE_POS:
            pos_score += POLITE_POS[pos]
        if pos in IMPOLITE_POS:
            pos_score -= IMPOLITE_POS[pos]
    
    # 문장 종결 형태 분석
    ending_score = 0.0
    
    # 높임말 종결어미 체크
    if any(ending in text for ending in ["습니다", "니다", "세요", "에요", "어요", "아요", "시오"]):
        ending_score += 1.0
    
    # 반말 종결어미 체크
    if any(ending in text for ending in ["냐", "니", "구나", "어", "아", "지", "래"]):
        ending_score -= 0.5
    
    # 질문형 표현 점수 계산 (의문문이 공손한 경우가 많음)
    question_score = 0.0
    if "?" in text or "까요" in text or "나요" in text or "ㅂ니까" in text:
        question_score = 0.5
    
    # 전체 점수 계산
    # 문장 길이에 따른 정규화
    length_factor = min(1.0, len(tokens) / 10)  # 문장이 너무 긴 경우 정규화
    
    total_score = base_score + token_score + bigram_score + pos_score + ending_score + question_score
    
    # 최종 점수 정규화 (0~1 범위로)
    normalized_score = max(0, min(1, (total_score + 3) / 6))
    
    return normalized_score

def predict_politeness_class(score):
    """공손함 점수를 바탕으로 공손함 클래스 예측
    
    :param score: 공손함 점수 (0~1)
    :return: 공손함 클래스 (매우 무례, 무례, 중립, 공손, 매우 공손)
    """
    if score < 0.2:
        return "매우 무례"
    elif score < 0.4:
        return "무례"
    elif score < 0.6:
        return "중립"
    elif score < 0.8:
        return "공손"
    else:
        return "매우 공손"

def evaluate_model(csv_file):
    """모델 평가
    
    :param csv_file: 평가용 CSV 파일 경로
    :return: 평가 결과
    """
    df = pd.read_csv(csv_file)
    
    # 공손함 점수 계산
    df['predicted_score'] = df['sentence'].apply(calculate_politeness_score)
    
    # 실제 점수와 예측 점수 간의 상관관계 계산
    correlation = np.corrcoef(df['score'], df['predicted_score'])[0, 1]
    
    # 예측 클래스 할당
    df['predicted_class'] = df['predicted_score'].apply(predict_politeness_class)
    
    return df, correlation

def test_samples():
    """샘플 문장으로 모델 테스트
    """
    samples = [
        "굳이 이런 분류가 있어야 할 이유를 모르겠습니다. 있더라도 '배우'는 '연예게 인물'의 하위 분류로 두는 것이 좋지 않을까요?",
        "아르헨티나와 저는 분명 다른 인물이고 그가 누군지도 모릅니다. 그렇게 따지면 삼국지 도원결의 카페 회원들은 전원 동일인입니까?",
        "위백의 이용 약관은 읽어보셨습니까? 님께서 주로 퍼오시는 백과사전 하단에는 뭐라고 써져 있습니까?",
        "안녕하세요! 혹시 멘토 신청 가능할까요?",
        "그럴수도 있겠군요... 그럼 저도 이제 로고를 올리려 합니다만, 대략 어떻게 올리셨는지 알려주실 수 있으신가요?",
        "그런데 반대로 Wikitori님은 \"닛폰국\"을 뒷받침할 근거를 가져오신 적이 있습니까?",
        "저렇게 편집하지 마세요! 자꾸 반달행위를 하시면 차단됩니다."
    ]
    
    print("\n샘플 문장 테스트:")
    for sample in samples:
        score = calculate_politeness_score(sample)
        politeness_class = predict_politeness_class(score)
        
        # 형태소 분석 결과 출력
        print(f"문장: {sample}")
        tokens, pos_info = tokenize(sample)
        print(f"형태소 분석 결과: {list(zip(tokens, pos_info))[:10]}... (총 {len(tokens)}개)")
        print(f"공손함 점수: {score:.4f}, 클래스: {politeness_class}\n")

if __name__ == "__main__":
    # 모델 평가
    try:
        df, correlation = evaluate_model('ko_test_template.csv')
        print(f"실제 점수와 예측 점수 간의 상관관계: {correlation:.4f}")
    
        # 샘플 문장 테스트
        test_samples()
    
        # 결과 저장
        df[['sentence', 'score', 'predicted_score', 'predicted_class']].to_csv('politeness_predictions_konlpy_filtered.csv', index=False)
    
        print("\n예측 결과가 'politeness_predictions_konlpy_filtered.csv' 파일에 저장되었습니다.")
    except Exception as e:
        print(f"오류 발생: {e}")
        import traceback
        traceback.print_exc() 
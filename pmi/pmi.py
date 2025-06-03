import pandas as pd
import numpy as np
import math
import re
from collections import defaultdict, Counter
from konlpy.tag import Okt

# 한국어 형태소 분석기 초기화
okt = Okt()

# 한국어 불용어 목록 (조사, 의존명사, 접미사 등 제외)
STOPWORDS = [
    "이", "가", "을", "를", "에", "에서", "의", "로", "으로", "와", "과", "이나", "나", "에게", "께", "한테",
    "까지", "부터", "라고", "이라고", "라는", "이라는", "라면", "이라면", "라도", "이라도", "만", "뿐", "도",
    "이다", "다", "것", "수", "듯", "데", "때", "곳", "뿐", "거", "줄", "만큼", "정도", "즈음", "이야", 
]

df = pd.read_csv('ko_test_template.csv')
# 점수 컬럼이 'score'라면
q1, q3 = df['score'].quantile([0.25,0.75])
impolite = df[df['score'] <= q1]
polite   = df[df['score'] >= q3]

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

def tokenize(text, use_pos=True, pos_filter=None):
    """KoNLPy Okt 형태소 분석기를 사용한 토큰화
    
    :param text: 전처리된 텍스트
    :param use_pos: 품사 정보를 토큰에 포함할지 여부
    :param pos_filter: 포함할 품사 목록 (None이면 모든 품사 포함)
    :return: 토큰 리스트
    """
    # 빈 텍스트 처리
    if not text:
        return []
        
    # 품사 필터 설정 (기본값: 내용어만 사용)
    if pos_filter is None:
        # 조사, 접속사, 특수문자 등 제외
        pos_filter = ['Noun', 'Verb', 'Adjective', 'Adverb', 'Determiner', 'Exclamation']
    
    # Okt 형태소 분석기 사용
    pos_tagged = okt.pos(text, norm=True, stem=True)
    
    # 필터링 및 토큰 생성
    filtered_pos = []
    for word, pos in pos_tagged:
        # 불용어 및 1글자 단어 제외 (단, 형용사와 부사는 1글자도 유지)
        if word in STOPWORDS:
            continue
        if len(word) < 2 and pos not in ['Adjective', 'Adverb']:
            continue
        if pos in pos_filter:
            filtered_pos.append((word, pos))
    
    # 토큰화
    if use_pos:
        # 품사 정보 포함 (예: '사과/Noun')
        tokens = [f"{word}/{pos}" for word, pos in filtered_pos]
    else:
        # 단어만 포함
        tokens = [word for word, pos in filtered_pos]
    
    return tokens

def create_co_matrix(sentences, vocab_size=None, window_size=1):
    """동시발생 행렬 생성
    
    :param sentences: 문장 리스트
    :param vocab_size: 어휘 수
    :param window_size: 윈도우 크기
    :return: 동시발생 행렬, token_to_id, id_to_token
    """
    token_to_id = {}
    id_to_token = {}
    token_counts = Counter()
    
    # 전처리 및 토큰화
    processed_sentences = []
    for sentence in sentences:
        preprocessed = preprocess_text(sentence)
        tokens = tokenize(preprocessed, use_pos=True, pos_filter=['Noun', 'Verb', 'Adjective', 'Adverb'])
        
        # 비어있는 토큰 리스트 건너뛰기
        if not tokens:
            continue
            
        processed_sentences.append(tokens)
        
        # 토큰 사전 구축 및 빈도 세기
        for token in tokens:
            token_counts[token] += 1
            if token not in token_to_id:
                id = len(token_to_id)
                token_to_id[token] = id
                id_to_token[id] = token
    
    # 토큰 수가 적으면 빈 행렬 반환
    if len(token_to_id) < 2:
        return np.zeros((0, 0)), {}, {}, Counter(), 0
        
    if vocab_size is None:
        vocab_size = len(token_to_id)
    
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)
    total_cooc = 0
    
    # 동시발생 행렬 생성
    for tokens in processed_sentences:
        token_ids = [token_to_id[token] for token in tokens if token in token_to_id]
        
        for idx, token_id in enumerate(token_ids):
            for i in range(1, window_size + 1):
                left_idx = idx - i
                right_idx = idx + i
                
                if left_idx >= 0:
                    left_token_id = token_ids[left_idx]
                    co_matrix[token_id, left_token_id] += 1
                    total_cooc += 1
                
                if right_idx < len(token_ids):
                    right_token_id = token_ids[right_idx]
                    co_matrix[token_id, right_token_id] += 1
                    total_cooc += 1
    
    return co_matrix, token_to_id, id_to_token, token_counts, total_cooc

def compute_pmi(pair, cooc, token_counts, total): 
    """PMI(점별 상호정보량) 계산
    
    :param pair: 토큰 쌍 (token_id1, token_id2)
    :param cooc: 동시발생 빈도
    :param token_counts: 토큰별 빈도 카운트
    :param total: 총 동시발생 수
    :return: PMI 값
    """
    if total == 0 or cooc == 0:
        return 0
        
    p_xy = cooc/total
    p_x = token_counts[pair[0]]/total
    p_y = token_counts[pair[1]]/total
    
    if p_x == 0 or p_y == 0:
        return 0
        
    return math.log2(p_xy/(p_x*p_y) + 1e-9)  # 0으로 나누는 것 방지

def create_pmi_df(sentences, window_size=1):
    """문장 리스트에서 PMI 데이터프레임 생성
    
    :param sentences: 문장 리스트
    :param window_size: 윈도우 크기
    :return: PMI 데이터프레임
    """
    co_matrix, token_to_id, id_to_token, token_counts, total = create_co_matrix(sentences, window_size=window_size)
    
    # 토큰 수가 너무 적으면 빈 데이터프레임 반환
    if len(token_to_id) < 2:
        return pd.DataFrame(columns=['token1', 'token2', 'pmi', 'cooc_freq'])
    
    # PMI 계산 및 데이터프레임 생성
    pmi_data = []
    for i in range(len(token_to_id)):
        for j in range(len(token_to_id)):
            if co_matrix[i, j] > 0:  # 동시발생이 있는 경우만
                pmi = compute_pmi((i, j), co_matrix[i, j], token_counts, total)
                if pmi > 0:  # 양의 PMI만 고려
                    pmi_data.append({
                        'token1': id_to_token[i],
                        'token2': id_to_token[j],
                        'pmi': pmi,
                        'cooc_freq': co_matrix[i, j]
                    })
    
    return pd.DataFrame(pmi_data)

# 공손한 표현과 무례한 표현에 대한 PMI 계산
polite_sentences = polite['sentence'].tolist()
impolite_sentences = impolite['sentence'].tolist()

print(f"공손한 문장 수: {len(polite_sentences)}")
print(f"무례한 문장 수: {len(impolite_sentences)}")

# 샘플 문장 출력
print("\n공손한 문장 샘플:")
for i in range(min(5, len(polite_sentences))):
    print(f"  {i+1}. {polite_sentences[i]}")
    tokens = tokenize(preprocess_text(polite_sentences[i]), use_pos=True)
    print(f"     형태소 분석: {tokens[:8] if tokens else '(없음)'}...")

print("\n무례한 문장 샘플:")
for i in range(min(5, len(impolite_sentences))):
    print(f"  {i+1}. {impolite_sentences[i]}")
    tokens = tokenize(preprocess_text(impolite_sentences[i]), use_pos=True)
    print(f"     형태소 분석: {tokens[:8] if tokens else '(없음)'}...")

# 윈도우 크기를 조정하여 더 많은 문맥 정보 캡처
polite_pmi_df = create_pmi_df(polite_sentences, window_size=3)
impolite_pmi_df = create_pmi_df(impolite_sentences, window_size=3)

# 결과 확인
print(f"\n공손한 표현 토큰 쌍 수: {len(polite_pmi_df)}")
print(f"무례한 표현 토큰 쌍 수: {len(impolite_pmi_df)}")

# 공손한 표현 후보 추출 (PMI 조건 완화)
if len(polite_pmi_df) > 0 and 'pmi' in polite_pmi_df.columns:
    # PMI > 1, 빈도 >= 2로 조건 완화
    candidates = polite_pmi_df[
        (polite_pmi_df['pmi'] > 1) & (polite_pmi_df['cooc_freq'] >= 2)
    ]
else:
    candidates = pd.DataFrame(columns=['token1', 'token2', 'pmi', 'cooc_freq'])
    print("공손한 표현 후보를 추출할 수 없습니다. 충분한 데이터가 없습니다.")

# impolite_pmi_df에 없는 토큰 쌍 필터링
if len(candidates) > 0 and len(impolite_pmi_df) > 0:
    polite_only_pairs = set(zip(polite_pmi_df['token1'], polite_pmi_df['token2']))
    impolite_pairs = set(zip(impolite_pmi_df['token1'], impolite_pmi_df['token2']))
    unique_polite_pairs = polite_only_pairs - impolite_pairs
    
    print(f"\n공손한 표현에만 나타나는 토큰 쌍 수: {len(unique_polite_pairs)}")
    
    # 공손 표현 후보 추출
    polite_markers = candidates[
        candidates.apply(lambda row: (row['token1'], row['token2']) in unique_polite_pairs, axis=1)
    ].sort_values('pmi', ascending=False)
else:
    unique_polite_pairs = set()
    polite_markers = pd.DataFrame(columns=['token1', 'token2', 'pmi', 'cooc_freq'])
    print("\n공손한 표현에만 나타나는 토큰 쌍이 없습니다.")

# 단어별 PMI 순위 출력
if len(polite_pmi_df) > 0:
    print("\n공손한 표현의 PMI 상위 20개:")
    print(polite_pmi_df.sort_values('pmi', ascending=False).head(20)[['token1', 'token2', 'pmi', 'cooc_freq']])

if len(impolite_pmi_df) > 0:
    print("\n무례한 표현의 PMI 상위 20개:")
    print(impolite_pmi_df.sort_values('pmi', ascending=False).head(20)[['token1', 'token2', 'pmi', 'cooc_freq']])

# 상위 20개 공손 표현 출력
print("\n상위 20개 공손 표현 후보:")
if len(polite_markers) > 0:
    for i, row in polite_markers.head(20).iterrows():
        print(f"{row['token1']} + {row['token2']}: PMI={row['pmi']:.4f}, 빈도={row['cooc_freq']}")
else:
    print("공손 표현 후보가 없습니다.")

# 품사별 분석 (품사 태그가 있는 경우)
polite_pos_freq = Counter()
impolite_pos_freq = Counter()

for sentence in polite_sentences:
    pos_tagged = okt.pos(preprocess_text(sentence), norm=True)
    polite_pos_freq.update([pos for _, pos in pos_tagged])

for sentence in impolite_sentences:
    pos_tagged = okt.pos(preprocess_text(sentence), norm=True)
    impolite_pos_freq.update([pos for _, pos in pos_tagged])

print("\n품사별 분포 비교:")
all_pos = set(list(polite_pos_freq.keys()) + list(impolite_pos_freq.keys()))
for pos in all_pos:
    polite_count = polite_pos_freq.get(pos, 0)
    impolite_count = impolite_pos_freq.get(pos, 0)
    
    # 공손한 표현에서 더 많이 나타나는 품사인 경우
    if polite_count > impolite_count and pos not in ['Josa', 'Suffix', 'Eomi', 'Punctuation']:
        print(f"{pos}: 공손 {polite_count}회, 무례 {impolite_count}회 (차이: +{polite_count - impolite_count})")

# 공손한 표현에서 자주 등장하는 토큰 분석
polite_token_freq = Counter()
for sentence in polite_sentences:
    tokens = tokenize(preprocess_text(sentence), use_pos=False)
    polite_token_freq.update(tokens)

impolite_token_freq = Counter()
for sentence in impolite_sentences:
    tokens = tokenize(preprocess_text(sentence), use_pos=False)
    impolite_token_freq.update(tokens)

# 공손한 표현에서 자주 등장하는 토큰
print("\n공손한 표현에서 자주 등장하는 토큰 (상위 20개):")
for token, freq in polite_token_freq.most_common(20):
    # 무례한 표현에서도 나타나는지 확인
    impolite_freq = impolite_token_freq.get(token, 0)
    polite_ratio = freq / max(1, impolite_freq)  # 공손/무례 비율
    
    if freq > impolite_freq:
        print(f"{token}: {freq}회 (무례한 표현: {impolite_freq}회, 비율: {polite_ratio:.2f}x)")

# 결과 저장
polite_pmi_df.to_csv('polite_pmi_konlpy_filtered.csv', index=False)
impolite_pmi_df.to_csv('impolite_pmi_konlpy_filtered.csv', index=False)

if len(polite_markers) > 0:
    polite_markers.to_csv('polite_markers_konlpy_filtered.csv', index=False)
else:
    pd.DataFrame(columns=['token1', 'token2', 'pmi', 'cooc_freq']).to_csv('polite_markers_konlpy_filtered.csv', index=False)

print(f"\n공손한 표현 PMI 계산 완료: {len(polite_pmi_df)}개")
print(f"무례한 표현 PMI 계산 완료: {len(impolite_pmi_df)}개")
print(f"고유한 공손 표현 후보: {len(polite_markers)}개")

import numpy as np
import sys
import os
import pandas as pd
from collections import Counter
from konlpy.tag import Okt
import re
import matplotlib.pyplot as plt

# 한국어 형태소 분석기 초기화
okt = Okt()

# 한국어 불용어 목록 (조사, 의존명사, 접미사 등 제외)
STOPWORDS = [
    "이", "가", "을", "를", "에", "에서", "의", "로", "으로", "와", "과", "이나", "나", "에게", "께", "한테",
    "까지", "부터", "라고", "이라고", "라는", "이라는", "라면", "이라면", "라도", "이라도", "만", "뿐", "도",
    "이다", "다", "것", "수", "듯", "데", "때", "곳", "뿐", "거", "줄", "만큼", "정도", "즈음", "이야"
]

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

def tokenize(text, use_pos=False, pos_filter=None):
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
    filtered_tokens = []
    
    for word, pos in pos_tagged:
        # 불용어 제외
        if word in STOPWORDS:
            continue
        # 1글자 단어 제외 (단, 형용사와 부사는 1글자도 유지)
        if len(word) < 2 and pos not in ['Adjective', 'Adverb']:
            continue
        if pos in pos_filter:
            if use_pos:
                filtered_tokens.append(f"{word}/{pos}")
            else:
                filtered_tokens.append(word)
    
    return filtered_tokens

def create_corpus(sentences):
    """문장 리스트에서 코퍼스 생성
    
    :param sentences: 문장 리스트
    :return: 코퍼스, word_to_id, id_to_word
    """
    word_to_id = {}
    id_to_word = {}
    
    # 모든 문장 토큰화
    corpus = []
    for sentence in sentences:
        preprocessed = preprocess_text(sentence)
        tokens = tokenize(preprocessed)
        
        for token in tokens:
            if token not in word_to_id:
                new_id = len(word_to_id)
                word_to_id[token] = new_id
                id_to_word[new_id] = token
            
            corpus.append(word_to_id[token])
    
    return np.array(corpus), word_to_id, id_to_word

def create_contexts_target(corpus, window_size=2):
    """맥락과 타깃 생성
    
    :param corpus: 코퍼스(단어 ID 리스트)
    :param window_size: 윈도우 크기
    :return: 맥락과 타깃
    """
    target = corpus[window_size:-window_size]
    contexts = []
    
    for idx in range(window_size, len(corpus)-window_size):
        cs = []
        for t in range(-window_size, window_size + 1):
            if t == 0:
                continue
            cs.append(corpus[idx + t])
        contexts.append(cs)
    
    return np.array(contexts), np.array(target)

def convert_one_hot(corpus, vocab_size):
    """원-핫 인코딩 변환
    
    :param corpus: 단어 ID 리스트
    :param vocab_size: 어휘 크기
    :return: 원-핫 인코딩된 배열
    """
    N = corpus.shape[0]
    
    if corpus.ndim == 1:
        one_hot = np.zeros((N, vocab_size), dtype=np.int32)
        for idx, word_id in enumerate(corpus):
            one_hot[idx, word_id] = 1
            
    elif corpus.ndim == 2:
        C = corpus.shape[1]
        one_hot = np.zeros((N, C, vocab_size), dtype=np.int32)
        for idx_0, word_ids in enumerate(corpus):
            for idx_1, word_id in enumerate(word_ids):
                one_hot[idx_0, idx_1, word_id] = 1
    
    return one_hot

def cos_similarity(x, y, eps=1e-8):
    """코사인 유사도 계산
    
    :param x: 벡터
    :param y: 벡터
    :param eps: 0으로 나누는 것을 방지하는 작은 값
    :return: 코사인 유사도
    """
    nx = x / (np.sqrt(np.sum(x ** 2)) + eps)
    ny = y / (np.sqrt(np.sum(y ** 2)) + eps)
    return np.dot(nx, ny)

def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
    """가장 유사한 단어 탐색
    
    :param query: 쿼리 단어
    :param word_to_id: 단어 -> ID 매핑
    :param id_to_word: ID -> 단어 매핑
    :param word_matrix: 단어 벡터 행렬
    :param top: 상위 N개 결과
    :return: 유사한 단어 리스트
    """
    if query not in word_to_id:
        print(f'단어 "{query}"를 찾을 수 없습니다.')
        return []
    
    print(f'\n[쿼리] {query}')
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]
    
    # 코사인 유사도 계산
    vocab_size = len(id_to_word)
    similarity = np.zeros(vocab_size)
    
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)
    
    # 결과 반환
    results = []
    count = 0
    for i in (-1 * similarity).argsort():
        if id_to_word[i] == query:
            continue
        results.append((id_to_word[i], similarity[i]))
        
        count += 1
        if count >= top:
            break
            
    return results

class MatMul:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.x = None

    def forward(self, x):
        W, = self.params
        out = np.dot(x, W)
        self.x = x
        return out

    def backward(self, dout):
        W, = self.params
        dx = np.dot(dout, W.T)
        dW = np.dot(self.x.T, dout)
        self.grads[0][...] = dW
        return dx

class SoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.y = None  # softmax의 출력
        self.t = None  # 정답 레이블

    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)

        # 정답 레이블이 원핫 벡터일 경우 정답의 인덱스로 변환
        if self.t.size == self.y.size:
            self.t = self.t.argmax(axis=1)

        loss = cross_entropy_error(self.y, self.t)
        return loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]

        dx = self.y.copy()
        dx[np.arange(batch_size), self.t] -= 1
        dx *= dout
        dx = dx / batch_size

        return dx

def softmax(x):
    if x.ndim == 2:
        x = x - x.max(axis=1, keepdims=True)
        x = np.exp(x)
        x /= x.sum(axis=1, keepdims=True)
    elif x.ndim == 1:
        x = x - np.max(x)
        x = np.exp(x)
        x /= np.sum(x)

    return x

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
        
    # 정답 데이터가 원핫 벡터일 경우 정답 인덱스로 변환
    if t.size == y.size:
        t = t.argmax(axis=1)
    
    batch_size = y.shape[0]
    
    # y[np.arange(batch_size), t]는 각 데이터의 정답 레이블에 해당하는 신경망의 출력
    # y가 0이면 -inf가 되지 않도록 아주 작은 값을 더해줌
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size

class SimpleCBOW:
    def __init__(self, vocab_size, hidden_size):
        V, H = vocab_size, hidden_size
        
        # 가중치 초기화
        W_in = 0.01 * np.random.randn(V, H).astype('f')
        W_out = 0.01 * np.random.randn(H, V).astype('f')
        
        # 계층 생성
        self.in_layers = []
        for i in range(2):  # 윈도우 크기가 1이면 좌우 문맥의 수는 2
            layer = MatMul(W_in)
            self.in_layers.append(layer)
        self.out_layer = MatMul(W_out)
        self.loss_layer = SoftmaxWithLoss()
        
        # 모든 가중치와 기울기를 리스트에 모은다.
        layers = self.in_layers + [self.out_layer]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads
            
        # 단어의 분산 표현
        self.word_vecs = W_in
        
    def forward(self, contexts, target):
        h = 0
        for i, layer in enumerate(self.in_layers):
            h += layer.forward(contexts[:, i])
        h *= 1 / len(self.in_layers)
        
        score = self.out_layer.forward(h)
        loss = self.loss_layer.forward(score, target)
        return loss
        
    def backward(self, dout=1):
        ds = self.loss_layer.backward(dout)
        da = self.out_layer.backward(ds)
        da *= 1 / len(self.in_layers)
        
        for layer in self.in_layers:
            layer.backward(da)
        return None

class Adam:
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.iter = 0
        self.m = None
        self.v = None
        
    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = [], []
            for param in params:
                self.m.append(np.zeros_like(param))
                self.v.append(np.zeros_like(param))
        
        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta2**self.iter) / (1.0 - self.beta1**self.iter)
        
        for i in range(len(params)):
            self.m[i] += (1 - self.beta1) * (grads[i] - self.m[i])
            self.v[i] += (1 - self.beta2) * (grads[i]**2 - self.v[i])
            
            params[i] -= lr_t * self.m[i] / (np.sqrt(self.v[i]) + 1e-7)

class Trainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.loss_list = []
        
    def fit(self, x, t, max_epoch=10, batch_size=32, eval_interval=20):
        data_size = len(x)
        max_iters = data_size // batch_size
        model, optimizer = self.model, self.optimizer
        total_loss = 0
        loss_count = 0
        
        for epoch in range(max_epoch):
            # 데이터 뒤섞기
            idx = np.random.permutation(data_size)
            x = x[idx]
            t = t[idx]
            
            for iters in range(max_iters):
                batch_x = x[iters * batch_size:(iters + 1) * batch_size]
                batch_t = t[iters * batch_size:(iters + 1) * batch_size]
                
                # 기울기 구해 매개변수 갱신
                loss = model.forward(batch_x, batch_t)
                model.backward()
                optimizer.update(model.params, model.grads)
                total_loss += loss
                loss_count += 1
                
                # 평가
                if (iters + 1) % eval_interval == 0:
                    avg_loss = total_loss / loss_count
                    print('| 에폭 %d |  반복 %d / %d | 손실 %.2f' % 
                         (epoch + 1, iters + 1, max_iters, avg_loss))
                    self.loss_list.append(avg_loss)
                    total_loss, loss_count = 0, 0
    
    def plot(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_list)
        plt.xlabel('반복 (x' + str(self.eval_interval) + ')')
        plt.ylabel('손실')
        plt.title('학습 곡선')
        plt.savefig('training_curve.png')
        plt.show()

def train_word2vec(sentences, hidden_size=100, batch_size=100, max_epoch=10, eval_interval=20):
    """Word2Vec 모델 학습
    
    :param sentences: 문장 리스트
    :param hidden_size: 은닉층 크기
    :param batch_size: 배치 크기
    :param max_epoch: 최대 에폭 수
    :param eval_interval: 평가 간격
    :return: 모델, word_to_id, id_to_word
    """
    # 코퍼스 생성
    corpus, word_to_id, id_to_word = create_corpus(sentences)
    
    # 맥락과 타깃 생성
    contexts, target = create_contexts_target(corpus)
    
    # 어휘 크기
    vocab_size = len(word_to_id)
    
    # 모델 및 최적화기 생성
    model = SimpleCBOW(vocab_size, hidden_size)
    optimizer = Adam()
    trainer = Trainer(model, optimizer)
    trainer.eval_interval = eval_interval
    
    # 원-핫 인코딩
    target = convert_one_hot(target, vocab_size)
    contexts = convert_one_hot(contexts, vocab_size)
    
    # 학습
    trainer.fit(contexts, target, max_epoch, batch_size)
    
    # 그래프 그리기
    trainer.plot()
    
    return model, word_to_id, id_to_word

def analyze_politeness(model, word_to_id, id_to_word, polite_examples, impolite_examples, top_n=10):
    """공손함 표현 분석
    
    :param model: Word2Vec 모델
    :param word_to_id: 단어 -> ID 매핑
    :param id_to_word: ID -> 단어 매핑
    :param polite_examples: 공손한 표현 예시 단어 리스트
    :param impolite_examples: 무례한 표현 예시 단어 리스트
    :param top_n: 상위 N개 결과
    :return: 분석 결과
    """
    word_matrix = model.word_vecs
    
    # 공손한 표현과 유사한 단어 찾기
    polite_similar_words = {}
    for word in polite_examples:
        if word in word_to_id:
            similar_words = most_similar(word, word_to_id, id_to_word, word_matrix, top_n)
            polite_similar_words[word] = similar_words
    
    # 무례한 표현과 유사한 단어 찾기
    impolite_similar_words = {}
    for word in impolite_examples:
        if word in word_to_id:
            similar_words = most_similar(word, word_to_id, id_to_word, word_matrix, top_n)
            impolite_similar_words[word] = similar_words
    
    return polite_similar_words, impolite_similar_words

def save_word_vectors(model, word_to_id, id_to_word, output_file='word_vectors.csv'):
    """단어 벡터 저장
    
    :param model: Word2Vec 모델
    :param word_to_id: 단어 -> ID 매핑
    :param id_to_word: ID -> 단어 매핑
    :param output_file: 출력 파일 이름
    """
    word_matrix = model.word_vecs
    words = []
    vectors = []
    
    for i in range(len(id_to_word)):
        words.append(id_to_word[i])
        vectors.append(word_matrix[i])
    
    df = pd.DataFrame({'word': words})
    for i in range(word_matrix.shape[1]):
        df[f'dim_{i}'] = [vec[i] for vec in vectors]
    
    df.to_csv(output_file, index=False)
    print(f'단어 벡터가 {output_file}에 저장되었습니다.')

if __name__ == "__main__":
    # 테스트
    print("Word2Vec 모델 테스트 중...")
    
    # 예시 문장
    sentences = [
        "안녕하세요! 혹시 멘토 신청 가능할까요?",
        "그럼 저도 이제 로고를 올리려 합니다만, 대략 어떻게 올리셨는지 알려주실 수 있으신가요?",
        "위백의 이용 약관은 읽어보셨습니까?",
        "저렇게 편집하지 마세요! 자꾸 반달행위를 하시면 차단됩니다."
    ]
    
    # 코퍼스 생성 테스트
    corpus, word_to_id, id_to_word = create_corpus(sentences)
    print(f"어휘 크기: {len(word_to_id)}")
    print(f"코퍼스 크기: {len(corpus)}")
    
    # 맥락과 타깃 생성 테스트
    contexts, target = create_contexts_target(corpus, window_size=1)
    print(f"맥락 크기: {contexts.shape}")
    print(f"타깃 크기: {target.shape}")
    
    # 샘플 문장 토큰화 테스트
    for sentence in sentences:
        print(f"\n원본 문장: {sentence}")
        tokens = tokenize(preprocess_text(sentence))
        print(f"토큰화 결과: {tokens}")
    
    print("\nWord2Vec 모델 테스트가 완료되었습니다.") 
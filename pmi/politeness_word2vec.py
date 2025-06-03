import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from word2vec_model import tokenize, preprocess_text, train_word2vec, analyze_politeness, save_word_vectors, most_similar
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def load_data(file_path='ko_test_template.csv'):
    """데이터 로드 및 전처리
    
    :param file_path: CSV 파일 경로
    :return: 데이터프레임, 공손한 문장 리스트, 무례한 문장 리스트
    """
    # 데이터 로드
    df = pd.read_csv(file_path)
    print(f"데이터 크기: {len(df)}")
    
    # 공손함 점수 기준으로 상위/하위 문장 추출
    q1, q3 = df['score'].quantile([0.25, 0.75])
    impolite = df[df['score'] <= q1]
    polite = df[df['score'] >= q3]
    
    print(f"공손한 문장 수: {len(polite)}")
    print(f"무례한 문장 수: {len(impolite)}")
    
    # 문장 리스트 추출
    polite_sentences = polite['sentence'].tolist()
    impolite_sentences = impolite['sentence'].tolist()
    
    # 샘플 문장 출력
    print("\n공손한 문장 샘플:")
    for i in range(min(3, len(polite_sentences))):
        print(f"  {i+1}. {polite_sentences[i]}")
        
    print("\n무례한 문장 샘플:")
    for i in range(min(3, len(impolite_sentences))):
        print(f"  {i+1}. {impolite_sentences[i]}")
    
    return df, polite_sentences, impolite_sentences

def extract_distinctive_tokens(polite_sentences, impolite_sentences):
    """공손한 표현과 무례한 표현에서 특징적인 토큰 추출
    
    :param polite_sentences: 공손한 문장 리스트
    :param impolite_sentences: 무례한 문장 리스트
    :return: 공손한 토큰 리스트, 무례한 토큰 리스트
    """
    # 토큰화
    polite_tokens = []
    for sentence in polite_sentences:
        tokens = tokenize(preprocess_text(sentence))
        polite_tokens.extend(tokens)
    
    impolite_tokens = []
    for sentence in impolite_sentences:
        tokens = tokenize(preprocess_text(sentence))
        impolite_tokens.extend(tokens)
    
    # 토큰 빈도 계산
    polite_freq = {}
    for token in polite_tokens:
        polite_freq[token] = polite_freq.get(token, 0) + 1
    
    impolite_freq = {}
    for token in impolite_tokens:
        impolite_freq[token] = impolite_freq.get(token, 0) + 1
    
    # 공손한 표현에서 특징적인 토큰 추출
    distinctive_polite = []
    for token, freq in sorted(polite_freq.items(), key=lambda x: x[1], reverse=True):
        impolite_count = impolite_freq.get(token, 0)
        if freq > impolite_count * 2 and freq >= 3:  # 무례한 표현보다 2배 이상 많고, 최소 3번 이상 등장
            distinctive_polite.append(token)
    
    # 무례한 표현에서 특징적인 토큰 추출
    distinctive_impolite = []
    for token, freq in sorted(impolite_freq.items(), key=lambda x: x[1], reverse=True):
        polite_count = polite_freq.get(token, 0)
        if freq > polite_count * 2 and freq >= 3:  # 공손한 표현보다 2배 이상 많고, 최소 3번 이상 등장
            distinctive_impolite.append(token)
    
    print(f"\n특징적인 공손 표현 토큰: {distinctive_polite[:10]}")
    print(f"특징적인 무례 표현 토큰: {distinctive_impolite[:10]}")
    
    return distinctive_polite[:20], distinctive_impolite[:20]

def visualize_word_vectors(model, word_to_id, id_to_word, distinctive_polite, distinctive_impolite):
    """단어 벡터 시각화
    
    :param model: Word2Vec 모델
    :param word_to_id: 단어 -> ID 매핑
    :param id_to_word: ID -> 단어 매핑
    :param distinctive_polite: 특징적인 공손 표현 토큰
    :param distinctive_impolite: 특징적인 무례 표현 토큰
    """
    # 단어 벡터 추출
    word_vecs = model.word_vecs
    
    # 시각화할 단어 선택
    target_words = []
    labels = []
    vectors = []
    
    for word in distinctive_polite:
        if word in word_to_id:
            target_words.append(word)
            labels.append(1)  # 공손
            vectors.append(word_vecs[word_to_id[word]])
    
    for word in distinctive_impolite:
        if word in word_to_id:
            target_words.append(word)
            labels.append(0)  # 무례
            vectors.append(word_vecs[word_to_id[word]])
    
    if not vectors:
        print("시각화할 단어 벡터가 없습니다.")
        return
    
    # PCA를 통한 차원 축소
    pca = PCA(n_components=2)
    vectors_2d = pca.fit_transform(vectors)
    
    # 시각화
    plt.figure(figsize=(12, 10))
    
    # 데이터 포인트 플로팅
    for i, word in enumerate(target_words):
        x, y = vectors_2d[i]
        plt.scatter(x, y, alpha=0.7, 
                   color='blue' if labels[i] == 1 else 'red')
        plt.annotate(word, (x, y), fontsize=9)
    
    plt.title('단어 벡터 PCA 시각화', fontsize=15)
    plt.legend(['공손', '무례'])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('word_vectors_pca.png', dpi=300)
    plt.show()
    
    # t-SNE를 통한 차원 축소
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(vectors)-1))
    vectors_tsne = tsne.fit_transform(vectors)
    
    # 시각화
    plt.figure(figsize=(12, 10))
    
    # 데이터 포인트 플로팅
    for i, word in enumerate(target_words):
        x, y = vectors_tsne[i]
        plt.scatter(x, y, alpha=0.7, 
                   color='blue' if labels[i] == 1 else 'red')
        plt.annotate(word, (x, y), fontsize=9)
    
    plt.title('단어 벡터 t-SNE 시각화', fontsize=15)
    plt.legend(['공손', '무례'])
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('word_vectors_tsne.png', dpi=300)
    plt.show()

def predict_politeness(model, word_to_id, text, polite_tokens, impolite_tokens):
    """문장의 공손함 점수 예측
    
    :param model: Word2Vec 모델
    :param word_to_id: 단어 -> ID 매핑
    :param text: 입력 문장
    :param polite_tokens: 공손한 표현 토큰 리스트
    :param impolite_tokens: 무례한 표현 토큰 리스트
    :return: 공손함 점수 (0~1)
    """
    # 전처리 및 토큰화
    preprocessed = preprocess_text(text)
    tokens = tokenize(preprocessed)
    
    # 단어 벡터
    word_vecs = model.word_vecs
    
    # 공손함 점수 계산
    polite_score = 0
    impolite_score = 0
    token_count = 0
    
    for token in tokens:
        if token in word_to_id:
            token_vec = word_vecs[word_to_id[token]]
            
            # 공손한 표현과의 유사도
            for p_token in polite_tokens:
                if p_token in word_to_id:
                    p_vec = word_vecs[word_to_id[p_token]]
                    similarity = np.dot(token_vec, p_vec) / (np.linalg.norm(token_vec) * np.linalg.norm(p_vec))
                    polite_score += max(0, similarity)
            
            # 무례한 표현과의 유사도
            for i_token in impolite_tokens:
                if i_token in word_to_id:
                    i_vec = word_vecs[word_to_id[i_token]]
                    similarity = np.dot(token_vec, i_vec) / (np.linalg.norm(token_vec) * np.linalg.norm(i_vec))
                    impolite_score += max(0, similarity)
            
            token_count += 1
    
    # 점수 정규화
    if token_count > 0:
        polite_score /= (token_count * len(polite_tokens))
        impolite_score /= (token_count * len(impolite_tokens))
        
        # 공손함 점수 계산 (0~1)
        score = polite_score / (polite_score + impolite_score + 1e-10)
    else:
        score = 0.5  # 중립
    
    return score

def evaluate_model(model, word_to_id, df, polite_tokens, impolite_tokens):
    """Word2Vec 모델 평가
    
    :param model: Word2Vec 모델
    :param word_to_id: 단어 -> ID 매핑
    :param df: 데이터프레임
    :param polite_tokens: 공손한 표현 토큰 리스트
    :param impolite_tokens: 무례한 표현 토큰 리스트
    """
    # 예측
    predictions = []
    for text in df['sentence']:
        score = predict_politeness(model, word_to_id, text, polite_tokens, impolite_tokens)
        predictions.append(score)
    
    # 결과 저장
    df['predicted_w2v'] = predictions
    
    # 상관관계 계산
    correlation = np.corrcoef(df['score'], df['predicted_w2v'])[0, 1]
    print(f"\n실제 점수와 Word2Vec 예측 점수 간의 상관관계: {correlation:.4f}")
    
    # 시각화
    plt.figure(figsize=(10, 6))
    plt.scatter(df['score'], df['predicted_w2v'], alpha=0.5)
    plt.xlabel('실제 공손함 점수')
    plt.ylabel('Word2Vec 예측 점수')
    plt.title('실제 점수 vs Word2Vec 예측 점수')
    
    # 회귀선
    z = np.polyfit(df['score'], df['predicted_w2v'], 1)
    p = np.poly1d(z)
    plt.plot(df['score'], p(df['score']), "r--")
    
    plt.savefig('word2vec_correlation.png', dpi=300)
    plt.show()
    
    # 결과 저장
    df[['sentence', 'score', 'predicted_w2v']].to_csv('politeness_predictions_word2vec.csv', index=False)
    print("\n예측 결과가 'politeness_predictions_word2vec.csv' 파일에 저장되었습니다.")
    
    return correlation

def test_samples(model, word_to_id, polite_tokens, impolite_tokens):
    """샘플 문장 테스트
    
    :param model: Word2Vec 모델
    :param word_to_id: 단어 -> ID 매핑
    :param polite_tokens: 공손한 표현 토큰 리스트
    :param impolite_tokens: 무례한 표현 토큰 리스트
    """
    samples = [
        "안녕하세요! 혹시 멘토 신청 가능할까요?",
        "그럼 저도 이제 로고를 올리려 합니다만, 대략 어떻게 올리셨는지 알려주실 수 있으신가요?",
        "다름이 아니라 문서를 삭제 신청을 하셨는데, 제 개인적인 생각으로는 삭제 요청 틀을 올리지 마시고, 다른 문서로 넘기는 것이 나을 듯 싶을 것 같습니다. 님의 생각은 어떠하신지요?",
        "저렇게 편집하지 마세요! 자꾸 반달행위를 하시면 차단됩니다.",
        "다른 말을 계속 반복하니까 문제인 거죠. 이건 사랑한다고 고백했는데 밥 먹었냐?",
        "너도 언녀처럼 칼로 배때기를 65번 쳐맞아보고 싶니? 응?"
    ]
    
    print("\n샘플 문장 테스트:")
    for sample in samples:
        score = predict_politeness(model, word_to_id, sample, polite_tokens, impolite_tokens)
        polite_class = "공손" if score > 0.6 else "무례" if score < 0.4 else "중립"
        
        # 토큰화 결과
        tokens = tokenize(preprocess_text(sample))
        
        print(f"\n문장: {sample}")
        print(f"토큰화 결과: {tokens}")
        print(f"공손함 점수: {score:.4f}, 클래스: {polite_class}")

if __name__ == "__main__":
    print("한국어 공손표현 Word2Vec 분석을 시작합니다.")
    
    # 데이터 로드
    df, polite_sentences, impolite_sentences = load_data()
    
    # 모든 문장 합치기
    all_sentences = df['sentence'].tolist()
    
    # 특징적인 토큰 추출
    distinctive_polite, distinctive_impolite = extract_distinctive_tokens(polite_sentences, impolite_sentences)
    
    # Word2Vec 모델 학습
    print("\nWord2Vec 모델 학습 중...")
    model, word_to_id, id_to_word = train_word2vec(
        all_sentences, 
        hidden_size=100,  # 은닉층 크기
        batch_size=32,    # 배치 크기
        max_epoch=20,     # 최대 에폭 수
        eval_interval=10  # 평가 간격
    )
    
    # 단어 벡터 시각화
    visualize_word_vectors(model, word_to_id, id_to_word, distinctive_polite, distinctive_impolite)
    
    # 공손/무례 표현 분석
    polite_similar, impolite_similar = analyze_politeness(
        model, word_to_id, id_to_word, 
        distinctive_polite[:5], distinctive_impolite[:5],
        top_n=5
    )
    
    # 단어 벡터 저장
    save_word_vectors(model, word_to_id, id_to_word, 'word_vectors.csv')
    
    # 모델 평가
    correlation = evaluate_model(model, word_to_id, df, distinctive_polite, distinctive_impolite)
    
    # 샘플 문장 테스트
    test_samples(model, word_to_id, distinctive_polite, distinctive_impolite)
    
    print("\n분석이 완료되었습니다.") 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

# 예측 결과 불러오기
df = pd.read_csv('politeness_predictions_konlpy_filtered.csv')

# 한글 폰트 설정 (Windows 기준)
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# 그림 크기 설정
plt.figure(figsize=(12, 10))

# 1. 실제 점수와 예측 점수 간의 산점도
plt.subplot(2, 2, 1)
plt.scatter(df['score'], df['predicted_score'], alpha=0.5)
plt.xlabel('실제 공손함 점수')
plt.ylabel('예측 공손함 점수')
plt.title('실제 점수 vs 예측 점수')

# 상관관계 선 추가
z = np.polyfit(df['score'], df['predicted_score'], 1)
p = np.poly1d(z)
plt.plot(df['score'], p(df['score']), "r--")
correlation = np.corrcoef(df['score'], df['predicted_score'])[0, 1]
plt.text(-2, 0.9, f'상관계수: {correlation:.4f}', fontsize=12)

# 2. 예측 클래스별 분포
plt.subplot(2, 2, 2)
class_counts = df['predicted_class'].value_counts()
plt.bar(class_counts.index, class_counts.values)
plt.xlabel('예측 클래스')
plt.ylabel('문장 수')
plt.title('예측 클래스별 분포')
plt.xticks(rotation=45)

# 3. 실제 점수 분포 (히스토그램)
plt.subplot(2, 2, 3)
plt.hist(df['score'], bins=20, alpha=0.7)
plt.xlabel('실제 공손함 점수')
plt.ylabel('문장 수')
plt.title('실제 공손함 점수 분포')

# 4. 예측 점수 분포 (히스토그램)
plt.subplot(2, 2, 4)
plt.hist(df['predicted_score'], bins=20, alpha=0.7)
plt.xlabel('예측 공손함 점수')
plt.ylabel('문장 수')
plt.title('예측 공손함 점수 분포')

plt.tight_layout()
plt.savefig('politeness_results_konlpy_filtered.png', dpi=300)
plt.close()

# 실제 공손함과 무례함에 따른 예측 결과 분석
# 실제 점수의 중앙값을 기준으로 공손/무례 구분
median_score = df['score'].median()
df['actual_polite'] = df['score'] > median_score

# 예측 점수의 중앙값을 기준으로 공손/무례 구분
df['predicted_polite'] = df['predicted_score'] > 0.6

# 혼동 행렬 계산
cm = confusion_matrix(df['actual_polite'], df['predicted_polite'])

# 혼동 행렬 시각화
plt.figure(figsize=(8, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['무례', '공손'])
disp.plot(cmap=plt.cm.Blues)
plt.title('공손함 분류 혼동 행렬 (KoNLPy 필터링)')
plt.savefig('confusion_matrix_konlpy_filtered.png', dpi=300)
plt.close()

# 공손함과 무례함의 대표적인 문장 예시 출력
polite_examples = df[df['actual_polite'] & df['predicted_polite']].nlargest(5, 'score')
impolite_examples = df[~df['actual_polite'] & ~df['predicted_polite']].nsmallest(5, 'score')

print("=== 공손한 문장 예시 (실제 & 예측 모두 공손) ===")
for i, row in polite_examples.iterrows():
    print(f"점수: {row['score']:.4f}, 예측: {row['predicted_score']:.4f}")
    print(f"문장: {row['sentence']}")
    print()

print("=== 무례한 문장 예시 (실제 & 예측 모두 무례) ===")
for i, row in impolite_examples.iterrows():
    print(f"점수: {row['score']:.4f}, 예측: {row['predicted_score']:.4f}")
    print(f"문장: {row['sentence']}")
    print()

# 잘못 예측된 문장 예시 출력
false_polite = df[~df['actual_polite'] & df['predicted_polite']].nsmallest(5, 'score')
false_impolite = df[df['actual_polite'] & ~df['predicted_polite']].nlargest(5, 'score')

print("=== 잘못 예측된 문장 예시 (실제는 무례하지만 공손하게 예측) ===")
for i, row in false_polite.iterrows():
    print(f"실제 점수: {row['score']:.4f}, 예측 점수: {row['predicted_score']:.4f}")
    print(f"문장: {row['sentence']}")
    print()

print("=== 잘못 예측된 문장 예시 (실제는 공손하지만 무례하게 예측) ===")
for i, row in false_impolite.iterrows():
    print(f"실제 점수: {row['score']:.4f}, 예측 점수: {row['predicted_score']:.4f}")
    print(f"문장: {row['sentence']}")
    print()

# 결과 요약
accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum()
precision = cm[1, 1] / (cm[0, 1] + cm[1, 1]) if (cm[0, 1] + cm[1, 1]) > 0 else 0
recall = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

print("=== 모델 성능 요약 (KoNLPy 필터링) ===")
print(f"정확도(Accuracy): {accuracy:.4f}")
print(f"정밀도(Precision): {precision:.4f}")
print(f"재현율(Recall): {recall:.4f}")
print(f"F1 점수: {f1:.4f}")
print(f"상관계수: {correlation:.4f}")

# 분석 결과 저장
with open('analysis_results_konlpy_filtered.txt', 'w', encoding='utf-8') as f:
    f.write("=== 모델 성능 요약 (KoNLPy 필터링) ===\n")
    f.write(f"정확도(Accuracy): {accuracy:.4f}\n")
    f.write(f"정밀도(Precision): {precision:.4f}\n")
    f.write(f"재현율(Recall): {recall:.4f}\n")
    f.write(f"F1 점수: {f1:.4f}\n")
    f.write(f"상관계수: {correlation:.4f}\n")
    
print("\n분석 결과가 'analysis_results_konlpy_filtered.txt'에 저장되었습니다.") 
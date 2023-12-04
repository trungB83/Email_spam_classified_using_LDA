import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Đọc dữ liệu từ file CSV
data = pd.read_csv('./data/spam.csv')

# Tách dữ liệu thành hai cột: Category (nhãn) và Message (nội dung email)
X = data['Message']
y = data['Category']

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tiền xử lý dữ liệu: chuyển đổi văn bản thành vector sử dụng CountVectorizer
vectorizer = CountVectorizer(stop_words='english')
X_train_vectorized = vectorizer.fit_transform(X_train)

# Xây dựng mô hình LDA
lda = LatentDirichletAllocation(n_components=10, random_state=42)
X_train_lda = lda.fit_transform(X_train_vectorized)
print('Ma trận huấn luyện x_train=',X_train_lda)

# Xây dựng mô hình phân loại
classifier = svm.SVC()
classifier.fit(X_train_lda, y_train)

# Tiền xử lý và đánh giá dữ liệu kiểm tra
X_test_vectorized = vectorizer.transform(X_test)
X_test_lda = lda.transform(X_test_vectorized)
predictions = classifier.predict(X_test_lda)
print('Ma trận giảm chiều x_train xác định được bởi LDA = ',X_test_lda)


# Đánh giá dữ liệu kiểm tra
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions, pos_label='spam')
recall = recall_score(y_test, predictions, pos_label='spam')
f1 = f1_score(y_test, predictions, pos_label='spam')

# In kết quả
print("Độ chính xác: {:.2f}%".format(accuracy * 100))
print("Precision (spam): {:.2f}%".format(precision * 100))
print("Recall (spam): {:.2f}%".format(recall * 100))
print("F1-score (spam): {:.2f}%".format(f1 * 100))
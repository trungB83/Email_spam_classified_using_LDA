import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.model_selection import train_test_split
from sklearn import svm, naive_bayes
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Đọc dữ liệu từ file CSV
data = pd.read_csv('./data/spam.csv')

# Tách dữ liệu thành hai cột: Category (nhãn) và Message (nội dung email)
X = data['Message']
y = data['Category']

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Chuyển đổi dữ liệu thành chuỗi
X_train = X_train.astype(str)
# Tiền xử lý dữ liệu: chuyển đổi văn bản thành vector sử dụng CountVectorizer
vectorizer = CountVectorizer(stop_words='english')
X_train_vectorized = vectorizer.fit_transform(X_train)

# Xây dựng mô hình LDA
lda = LatentDirichletAllocation(n_components=10, random_state=42)
X_train_lda = lda.fit_transform(X_train_vectorized)

# Xây dựng mô hình phân loại SVM
svm_classifier = svm.SVC()
svm_classifier.fit(X_train_vectorized, y_train)

# Xây dựng mô hình phân loại SVM + LDA
svm_lda_classifier = svm.SVC()
svm_lda_classifier.fit(X_train_lda, y_train)

# Xây dựng mô hình phân loại Naïve Bayes
nb_classifier = naive_bayes.MultinomialNB()
nb_classifier.fit(X_train_vectorized, y_train)

# Xây dựng mô hình phân loại Naïve Bayes + LDA
lda_nb_classifier = naive_bayes.MultinomialNB()
X_train_lda_vectorized = vectorizer.transform(X_train_lda)
lda_nb_classifier.fit(X_train_lda_vectorized, y_train)

# Tiền xử lý và đánh giá dữ liệu kiểm tra
X_test_vectorized = vectorizer.transform(X_test)
X_test_lda = lda.transform(X_test_vectorized)

# Dự đoán kết quả cho SVM
predictions_svm = svm_classifier.predict(X_test_vectorized)

# Dự đoán kết quả cho SVM + LDA
predictions_svm_lda = svm_lda_classifier.predict(X_test_lda)

# Dự đoán kết quả cho Naïve Bayes
predictions_nb = nb_classifier.predict(X_test_vectorized)

# Dự đoán kết quả cho Naïve Bayes + LDA
X_test_lda_vectorized = vectorizer.transform(X_test_lda)
predictions_lda_nb = lda_nb_classifier.predict(X_test_lda_vectorized)

# Tính toán độ đo Accuracy, Precision, Recall và F1-Score cho SVM
accuracy_svm = accuracy_score(y_test, predictions_svm)
precision_svm = precision_score(y_test, predictions_svm, pos_label='spam')
recall_svm = recall_score(y_test, predictions_svm, pos_label='spam')
f1_svm = f1_score(y_test, predictions_svm, pos_label='spam')

# Tính toán độ đo Accuracy, Precision, Recall và F1-Score cho SVM + LDA
accuracy_svm_lda = accuracy_score(y_test, predictions_svm_lda)
precision_svm_lda = precision_score(y_test, predictions_svm_lda, pos_label='spam')
recall_svm_lda = recall_score(y_test, predictions_svm_lda, pos_label='spam')
f1_svm_lda = f1_score(y_test, predictions_svm_lda, pos_label='spam')

# Tính toán độ đo Accuracy, Precision, Recall và F1-Score cho Naïve Bayes
accuracy_nb = accuracy_score(y_test, predictions_nb)
precision_nb = precision_score(y_test, predictions_nb, pos_label='spam')
recall_nb = recall_score(y_test, predictions_nb, pos_label='spam')
f1_nb = f1_score(y_test, predictions_nb, pos_label='spam')

# Tính toán độ đo Accuracy, Precision, Recall và F1-Score cho Naïve Bayes + LDA
accuracy_lda_nb = accuracy_score(y_test, predictions_lda_nb)
precision_lda_nb = precision_score(y_test, predictions_lda_nb, pos_label='spam')
recall_lda_nb = recall_score(y_test, predictions_lda_nb, pos_label='spam')
f1_lda_nb = f1_score(y_test, predictions_lda_nb, pos_label='spam')

# Tạo DataFrame chứa kết quả so sánh
data = {'Phương pháp': ['SVM', 'SVM + LDA', 'Naïve Bayes', 'Naïve Bayes + LDA'],
        'Accuracy': [accuracy_svm, accuracy_svm_lda, accuracy_nb, accuracy_lda_nb],
        'Precision': [precision_svm, precision_svm_lda, precision_nb, precision_lda_nb],
        'Recall': [recall_svm, recall_svm_lda, recall_nb, recall_lda_nb],
        'F1-Score': [f1_svm, f1_svm_lda, f1_nb, f1_lda_nb],
        'Nhãn': ['spam', 'spam', 'spam', 'spam']}
df = pd.DataFrame(data)

# Hiển thị bảng
print(df)
import pandas as pd

# Đọc dữ liệu từ tập tin csv
data = pd.read_csv("./data/spam.csv")

# Số trường dữ liệu
num_fields = data.shape[1]

# Số lớp trong tập dữ liệu
num_classes = len(data["Category"].unique())

# Số lượng email spam
num_spam = len(data[data["Category"] == "spam"])

# Số lượng email không phải spam
num_ham = len(data[data["Category"] == "ham"])

print("Tổng số trường dữ liệu:", num_fields)
print("Số lớp trong tập dữ liệu:", num_classes)
print("Số lượng email spam:", num_spam)
print("Số lượng email không phải spam:", num_ham)
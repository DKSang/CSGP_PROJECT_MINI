# Import các thư viện cần thiết
import pandas as pd  # Thư viện xử lý dữ liệu dạng bảng
from sklearn.model_selection import train_test_split, GridSearchCV  # Chia dữ liệu và tìm kiếm tham số tối ưu
from sklearn.impute import SimpleImputer  # Xử lý giá trị thiếu
from sklearn.preprocessing import StandardScaler, OneHotEncoder  # Chuẩn hóa dữ liệu số và mã hóa biến phân loại
from sklearn.pipeline import Pipeline  # Tạo pipeline để xử lý dữ liệu và mô hình
from sklearn.compose import ColumnTransformer  # Áp dụng các bước xử lý khác nhau cho các cột khác nhau
from sklearn.metrics import classification_report  # Đánh giá mô hình
from sklearn.ensemble import RandomForestClassifier  # Mô hình Random Forest
from imblearn.over_sampling import SMOTE  # Cân bằng dữ liệu bằng oversampling
from imblearn.pipeline import Pipeline as imPipeline  # Pipeline hỗ trợ SMOTE

# Đọc dữ liệu từ file CSV
# File 'csgo.csv' chứa dữ liệu về các trận đấu CSGO, với các cột như map, match_time_s, result, v.v.
data = pd.read_csv(r"csgo.csv")

# Xử lý dữ liệu
# Loại bỏ các cột không cần thiết cho việc dự đoán
# Các cột bị loại: day, month, year, date (thời gian), wait_time_s (thời gian chờ), 
# team_a_rounds, team_b_rounds (số vòng của đội), result (biến mục tiêu)
dlt = ["day", "month", "year", "date", "wait_time_s", "team_a_rounds", "team_b_rounds", "result"]
x = data.drop(dlt, axis=1)  # Đặc trưng (features)
y = data["result"]  # Biến mục tiêu (target), kết quả trận đấu

# Chia dữ liệu thành tập huấn luyện (80%) và tập kiểm tra (20%)
# random_state=42 đảm bảo kết quả chia dữ liệu có thể tái hiện
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Xử lý dữ liệu phân loại (categorical) và số (numerical)
# Cột 'map' là biến phân loại, cần mã hóa
map_feature = ["map"]
nor_trans = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),  # Điền giá trị thiếu bằng giá trị xuất hiện nhiều nhất
    ("one_hot", OneHotEncoder(sparse_output=False, handle_unknown='ignore'))  # Mã hóa one-hot, bỏ qua giá trị mới
])

# Các cột số cần chuẩn hóa
num_lfeature = ["match_time_s", "ping", "kills", "assists", "deaths", "mvps", "hs_percent", "points"]
num_trans = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),  # Điền giá trị thiếu bằng giá trị trung vị
    ("scaler", StandardScaler())  # Chuẩn hóa dữ liệu về mean=0, std=1
])

# Kết hợp xử lý cho các cột số và phân loại
preprocessor = ColumnTransformer(transformers=[
    ("num", num_trans, num_lfeature),  # Áp dụng pipeline số cho các cột số
    ("nor", nor_trans, map_feature)    # Áp dụng pipeline phân loại cho cột map
])

# Tạo pipeline với SMOTE và RandomForest
# SMOTE để xử lý mất cân bằng lớp (nếu có)
smote = SMOTE(sampling_strategy='auto', random_state=42)
rf = RandomForestClassifier(random_state=42)  # Mô hình Random Forest

# Pipeline kết hợp: tiền xử lý, cân bằng dữ liệu, huấn luyện mô hình
pipeline = imPipeline(steps=[
    ('preprocessor', preprocessor),
    ('smote', smote),
    ('model', rf)
])

# Định nghĩa lưới tham số cho GridSearchCV
# Tìm kiếm các tham số tối ưu cho RandomForest
param_grid = {
    'model__n_estimators': [100, 200, 300],  # Số cây trong rừng
    'model__class_weight': [None, 'balanced'],  # Cân bằng trọng số lớp
    'model__criterion': ['gini', 'entropy'],   # Tiêu chí phân tách
    'model__max_depth': [None, 10, 20]         # Độ sâu tối đa của cây
}

# Tạo GridSearchCV để tìm tham số tốt nhất
# Sử dụng f1_weighted làm metric để đánh giá (phù hợp với bài toán đa lớp)
model = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,  # 5-fold cross-validation
    verbose=1,  # In tiến trình
    scoring='f1_weighted'  # Tối ưu hóa điểm F1 trọng số
)

# Huấn luyện mô hình trên tập huấn luyện
model.fit(x_train, y_train)

# In kết quả tốt nhất từ GridSearchCV
print("Best parameters:", model.best_params_)  # Tham số tốt nhất
print("Best F1 score:", model.best_score_)    # Điểm F1 tốt nhất trên tập huấn luyện

# Dự đoán trên tập kiểm tra và đánh giá
y_pred = model.predict(x_test)
print(classification_report(y_test, y_pred))  # In báo cáo phân loại (precision, recall, f1-score)

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Import 5 thuật toán theo đề bài
from sklearn.linear_model import LinearRegression, Ridge # 
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor

# 1. Nạp dữ liệu Diabetes (dùng toàn bộ 10 features)
diabetes = load_diabetes()
X = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
y = diabetes.target

# 2. Chia tập Train / Test (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Khai báo danh sách các mô hình
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Decision Tree Regression": DecisionTreeRegressor(random_state=42),
    "Support Vector Regression (SVR)": SVR(kernel='rbf'),
    "Gradient Boosting Regression": GradientBoostingRegressor(random_state=42)
}

# 5. Huấn luyện và đánh giá từng mô hình
results = []

for name, model in models.items():
    # Huấn luyện mô hình
    model.fit(X_train_scaled, y_train)

    # Dự báo trên tập test
    y_pred = model.predict(X_test_scaled)

    # Tính các chỉ số
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    results.append({
        "Thuật toán": name,
        "MAE": round(mae, 2),
        "RMSE": round(rmse, 2),
        "R² Score": round(r2, 4)
    })

# 6. Hiển thị bảng so sánh
results_df = pd.DataFrame(results).sort_values(by="R² Score", ascending=False)
print("=== BẢNG SO SÁNH CÁC THUẬT TOÁN (BÀI 3) ===")
print(results_df.to_string(index=False))
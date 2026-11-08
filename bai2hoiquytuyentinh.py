import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 1. Nạp dữ liệu Diabetes
diabetes = load_diabetes()
df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
df['target'] = diabetes.target

# 2. Xác định Top 5 features theo độ tương quan tuyệt đối với target
top5_features = df.corr()['target'].abs().drop('target').nlargest(5).index.tolist()

# 3. Độ nghiệp 4 tập features theo yêu cầu đề bài
feature_sets = {
    "1. bmi": ['bmi'],
    "2. bmi + bp": ['bmi', 'bp'],
    f"3. Top 5 Tương quan ({', '.join(top5_features)})": top5_features,
    "4. Toàn bộ 10 features": diabetes.feature_names
}

results = []

# 4. Vòng lặp huấn luyện và đánh giá trên từng tập feature
for set_name, features in feature_sets.items():
    X = df[features]
    y = df['target']

    # Chia tập dữ liệu ngẫu nhiên (80% Train / 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Huấn luyện mô hình
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # Dự báo
    y_pred = model.predict(X_test_scaled)

    # Tính toán chỉ số MAE, RMSE, R2
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))  # RMSE = Căn bậc hai của MSE
    r2 = r2_score(y_test, y_pred)

    results.append({
        "Bộ Features": set_name,
        "Số lượng Features": len(features),
        "MAE": round(mae, 2), # sai số tuyệt đối trung bình
        "RMSE": round(rmse, 2), # căn bậc 2 của sai số tuyển đối trung bình
        "R² Score": round(r2, 4) # hệ số trung bình
    })

# 5. In bảng so sánh kết quả
results_df = pd.DataFrame(results)
print("=== BẢNG SO SÁNH KẾT QUẢ HUẤN LUYỆN BÀI 2 ===")
print(results_df.to_string(index=False))
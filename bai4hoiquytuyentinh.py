import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Import các thuật toán
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor

# 1. Nạp trực tiếp dữ liệu King County House Data
df = pd.read_csv("kc_house_data.csv")

# 2. FEATURE ENGINEERING (Tạo thêm thuộc tính mới)
if 'yr_built' in df.columns:
    df['house_age'] = 2015 - df['yr_built']

if 'yr_renovated' in df.columns:
    df['is_renovated'] = df['yr_renovated'].apply(lambda x: 1 if x > 0 else 0)

# 3. ĐỊNH NGHĨA 2 TẬP FEATURE
# Tập 1: Chọn bằng tay 5 đặc trưng quan trọng nhất
manual_top5 = ['sqft_living', 'grade', 'sqft_above', 'sqft_living15', 'bathrooms']

# Tập 2: Toàn bộ đặc trưng (loại bỏ các cột không phải thông số nhà)
drop_cols = ['id', 'date', 'price']
all_features = [col for col in df.columns if col not in drop_cols]

feature_sets = {
    "Top 5 Features tự chọn": manual_top5,
    "Toàn bộ đặc trưng": all_features
}

# 4. ĐỊNH NGHĨA CÁC MÔ HÌNH
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Decision Tree": DecisionTreeRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42)
}

# 5. HUẤN LUYỆN PIPELINE VÀ ĐÁNH GIÁ
results = []
y = df['price']

for set_name, cols in feature_sets.items():
    X = df[cols]

    # Chia tập Train/Test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    for model_name, model in models.items():
        # Pipeline: Chuẩn hóa dữ liệu (StandardScaler) -> Huấn luyện mô hình
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', model)
        ])

        # Huấn luyện
        pipeline.fit(X_train, y_train)

        # Dự báo
        y_pred = pipeline.predict(X_test)

        # Đánh giá chỉ số
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        results.append({
            "Bộ Feature": set_name,
            "Số lượng": len(cols),
            "Mô hình": model_name,
            "MAE ($)": round(mae, 2),
            "RMSE ($)": round(rmse, 2),
            "R² Score": round(r2, 4)
        })

# 6. IN BẢNG TỔNG HỢP KẾT QUẢ
results_df = pd.DataFrame(results).sort_values(by="R² Score", ascending=False)
print("=== BẢNG SO SÁNH CÁC MÔ HÌNH BÀI 4 ===")
print(results_df.to_string(index=False))
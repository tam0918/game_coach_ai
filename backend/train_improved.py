import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
# Sử dụng pipeline của scikit-learn vì không cần imblearn nữa
from sklearn.pipeline import Pipeline

# --- Cấu hình đường dẫn ---
DATA_PATH = "backend/clean_training_data.csv"
MODEL_DIR = "backend/models_3_classes_manual_balance" # Thư mục mới cho kỹ thuật này
MODEL_PATH = os.path.join(MODEL_DIR, "move_classifier_3_classes_manual_balance.joblib")

# Tạo thư mục lưu mô hình nếu chưa tồn tại
os.makedirs(MODEL_DIR, exist_ok=True)

def load_and_prepare_data():
    """
    Đọc dữ liệu, gộp 5 lớp thành 3 lớp, sau đó cân bằng thủ công tập huấn luyện.
    """
    print("--- Bước 1: Đang đọc và chuẩn bị dữ liệu ---")
    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=['llm_classification_value'])
    y_original = df['llm_classification_value'] - 1

    mapping = {0: 0, 1: 0, 2: 1, 3: 2, 4: 2} # Tốt: 0, Trung bình: 1, Kém: 2
    y_mapped = y_original.map(mapping)
    y_mapped.name = "target" # Đặt tên cho Series để dễ thao tác

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_mapped, test_size=0.2, random_state=42, stratify=y_mapped
    )

    # *** NÂNG CẤP CỐT LÕI: CÂN BẰNG DỮ LIỆU THỦ CÔNG ***
    print("\n--- Bước 1.1: Thực hiện cân bằng dữ liệu thủ công (100 mẫu mỗi lớp) ---")
    # Kết hợp lại X_train và y_train để dễ dàng lấy mẫu
    train_data = pd.concat([X_train, y_train], axis=1)
    
    # Lấy 100 mẫu cho mỗi lớp. `replace=True` đảm bảo lớp thiếu mẫu vẫn lấy đủ.
    balanced_train_df = train_data.groupby('target').apply(
        lambda x: x.sample(n=100, replace=True, random_state=42)
    ).reset_index(drop=True)

    print("Phân phối nhãn của tập huấn luyện sau khi cân bằng:")
    print(balanced_train_df['target'].value_counts().sort_index().rename({0: 'Tốt', 1: 'Trung bình', 2: 'Kém'}))

    # Tách lại thành X và y cho tập huấn luyện mới
    X_train_balanced = balanced_train_df.drop(columns=['target'])
    y_train_balanced = balanced_train_df['target']
    
    print("\n - Chuẩn bị dữ liệu hoàn tất.")
    return X_train_balanced, X_test, y_train_balanced, y_test

def tune_and_train_ensemble(X_train, y_train):
    """
    Sử dụng GridSearchCV để tối ưu và huấn luyện mô hình trên tập dữ liệu đã cân bằng.
    """
    print("\n--- Bước 2: Tối ưu hóa và huấn luyện mô hình Ensemble ---")

    # Pipeline giờ không cần bước resampler (SMOTE) nữa
    xgb_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', XGBClassifier(
            objective='multi:softmax', num_class=3, use_label_encoder=False, 
            eval_metric='mlogloss', random_state=42, n_jobs=-1
        ))
    ])

    mlp_pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', MLPClassifier(
            max_iter=1000, early_stopping=True, n_iter_no_change=20, random_state=42
        ))
    ])
    
    ensemble_model = VotingClassifier(
        estimators=[('xgb', xgb_pipe), ('mlp', mlp_pipe)],
        voting='soft'
    )

    param_grid = {
        'xgb__classifier__n_estimators': [300, 400],
        'xgb__classifier__max_depth': [7, 10],
        'mlp__classifier__hidden_layer_sizes': [(100, 50, 25), (128, 64)],
        'weights': [[0.6, 0.4], [0.5, 0.5]]
    }

    grid_search = GridSearchCV(
        estimator=ensemble_model,
        param_grid=param_grid,
        cv=3,
        scoring='f1_weighted',
        verbose=3,
        n_jobs=-1
    )

    print(" - Bắt đầu quá trình tìm kiếm tham số tối ưu (GridSearchCV)...")
    grid_search.fit(X_train, y_train)
    
    print("\n--- Kết quả tìm kiếm ---")
    print(f"Tham số tốt nhất tìm được: {grid_search.best_params_}")
    print(f"Điểm F1-weighted cross-validation tốt nhất: {grid_search.best_score_:.4f}")
    
    print("\n - Huấn luyện hoàn tất.")
    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test):
    """
    Đánh giá mô hình Ensemble trên tập test.
    """
    print("\n--- Bước 3: Đang đánh giá mô hình Ensemble đã được tối ưu hóa ---")
    
    y_pred = model.predict(X_test)
    
    accuracy = accuracy_score(y_test, y_pred)
    class_names = ['Tốt', 'Trung bình', 'Kém']
    
    print(f"Độ chính xác: {accuracy:.4f}")
    print("\nBáo cáo phân loại:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='cividis', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Dự đoán')
    plt.ylabel('Thực tế')
    plt.title('Ma trận nhầm lẫn - Mô hình với dữ liệu cân bằng thủ công')
    plt.savefig(os.path.join(MODEL_DIR, 'confusion_matrix_manual_balance.png'))
    print(f"\nĐã lưu ma trận nhầm lẫn tại '{MODEL_DIR}/confusion_matrix_manual_balance.png'")

def main():
    """
    Hàm chính điều phối toàn bộ quy trình.
    """
    print("====== BẮT ĐẦU QUY TRÌNH HUẤN LUYỆN VỚI DỮ LIỆU CÂN BẰNG THỦ CÔNG ======")
    
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    if X_train is None: return

    model = tune_and_train_ensemble(X_train, y_train)
    
    joblib.dump(model, MODEL_PATH)
    print(f"\nĐã lưu pipeline mô hình Ensemble tốt nhất tại '{MODEL_PATH}'")
    
    evaluate_model(model, X_test, y_test)
    
    print("\n====== HOÀN THÀNH QUY TRÌNH HUẤN LUYỆN ======")

if __name__ == "__main__":
    main()

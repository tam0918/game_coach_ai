import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils import class_weight
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb

# Đường dẫn đến file dữ liệu và thư mục lưu mô hình
DATA_PATH = "backend/clean_training_data.csv"
MODEL_DIR = "backend/models"
SCALER_PATH = os.path.join(MODEL_DIR, "scaler_advanced.joblib")
MODEL_PATH = os.path.join(MODEL_DIR, "move_classifier_advanced.joblib")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder_advanced.joblib")

# Tạo thư mục lưu mô hình nếu chưa tồn tại
os.makedirs(MODEL_DIR, exist_ok=True)

def load_and_prepare_data():
    """
    Đọc dữ liệu, chuẩn bị features và labels
    """
    print("Đang đọc dữ liệu từ", DATA_PATH)
    df = pd.read_csv(DATA_PATH)
    
    # Hiển thị thông tin về dữ liệu
    print(f"Dữ liệu có {df.shape[0]} dòng và {df.shape[1]} cột")
    print("\nPhân phối nhãn:")
    print(df['llm_classification_value'].value_counts().sort_index())
    
    # Tách features và label
    X = df.drop(columns=['llm_classification_value'])
    y = df['llm_classification_value']
    
    # Chuyển nhãn từ 1-5 thành 0-4 để dễ dàng one-hot encoding
    y = y - 1
    
    # Chia tập dữ liệu thành train và test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nSố lượng mẫu huấn luyện: {X_train.shape[0]}")
    print(f"Số lượng mẫu kiểm tra: {X_test.shape[0]}")
    
    return X_train, X_test, y_train, y_test

def engineer_features(X_train, X_test):
    """
    Tạo thêm các đặc trưng mới từ các đặc trưng hiện có
    """
    print("\nĐang tạo thêm đặc trưng...")
    
    # Tạo bản sao để không ảnh hưởng đến dữ liệu gốc
    X_train_new = X_train.copy()
    X_test_new = X_test.copy()
    
    # In kích thước để debug
    print(f"Kích thước X_train: {X_train.shape}")
    print(f"Kích thước X_train_new: {X_train_new.shape}")
    
    # 1. Tạo đặc trưng mới: Tỷ lệ matching_tiles_count trên từng layer
    for i in range(4):
        X_train_new[f'matching_ratio_layer_{i}'] = X_train['matching_tiles_layer_{}'.format(i)] / (X_train['matching_tiles_count'] + 1e-6)
        X_test_new[f'matching_ratio_layer_{i}'] = X_test['matching_tiles_layer_{}'.format(i)] / (X_test['matching_tiles_count'] + 1e-6)
    
    # 2. Tạo đặc trưng mới: Tỷ lệ matching_tiles trên layer được chọn
    X_train_new['matching_ratio_selected_layer'] = 0.0
    X_test_new['matching_ratio_selected_layer'] = 0.0
    
    for i in range(len(X_train_new)):
        layer = int(X_train.iloc[i]['tile_layer'])
        if layer < 4:  # Đảm bảo layer hợp lệ
            X_train_new.loc[X_train_new.index[i], 'matching_ratio_selected_layer'] = X_train.iloc[i][f'matching_tiles_layer_{layer}'] / (X_train.iloc[i]['matching_tiles_count'] + 1e-6)
    
    for i in range(len(X_test_new)):
        layer = int(X_test.iloc[i]['tile_layer'])
        if layer < 4:  # Đảm bảo layer hợp lệ
            X_test_new.loc[X_test_new.index[i], 'matching_ratio_selected_layer'] = X_test.iloc[i][f'matching_tiles_layer_{layer}'] / (X_test.iloc[i]['matching_tiles_count'] + 1e-6)
    
    # 3. Tạo đặc trưng mới: Tương tác giữa difficulty và matching_tiles_count
    X_train_new['diff_matching_interaction'] = X_train['difficulty_numeric'] * X_train['matching_tiles_count']
    X_test_new['diff_matching_interaction'] = X_test['difficulty_numeric'] * X_test['matching_tiles_count']
    
    # 4. Tạo đặc trưng mới: Tương tác giữa difficulty và collection_fullness
    X_train_new['diff_collection_interaction'] = X_train['difficulty_numeric'] * X_train['collection_fullness']
    X_test_new['diff_collection_interaction'] = X_test['difficulty_numeric'] * X_test['collection_fullness']
    
    # 5. Tạo đặc trưng mới: Tổng số matching tiles trên các layer trên layer hiện tại
    X_train_new['tiles_above_current'] = 0.0
    X_test_new['tiles_above_current'] = 0.0
    
    for i in range(len(X_train_new)):
        layer = int(X_train.iloc[i]['tile_layer'])
        if layer < 4:  # Đảm bảo layer hợp lệ
            X_train_new.loc[X_train_new.index[i], 'tiles_above_current'] = sum(X_train.iloc[i][f'matching_tiles_layer_{j}'] for j in range(layer))
    
    for i in range(len(X_test_new)):
        layer = int(X_test.iloc[i]['tile_layer'])
        if layer < 4:  # Đảm bảo layer hợp lệ
            X_test_new.loc[X_test_new.index[i], 'tiles_above_current'] = sum(X_test.iloc[i][f'matching_tiles_layer_{j}'] for j in range(layer))
    
    # 6. Tạo đặc trưng mới: Bình phương và căn bậc hai của một số đặc trưng quan trọng
    for feature in ['matching_tiles_count', 'unblocks_count']:
        X_train_new[f'{feature}_squared'] = X_train[feature] ** 2
        X_test_new[f'{feature}_squared'] = X_test[feature] ** 2
        X_train_new[f'{feature}_sqrt'] = np.sqrt(X_train[feature] + 1e-6)
        X_test_new[f'{feature}_sqrt'] = np.sqrt(X_test[feature] + 1e-6)
    
    print(f"Số lượng đặc trưng ban đầu: {X_train.shape[1]}")
    print(f"Số lượng đặc trưng sau khi tạo thêm: {X_train_new.shape[1]}")
    
    # Kiểm tra xem các cột của X_train_new và X_test_new có khớp nhau không
    train_cols = set(X_train_new.columns)
    test_cols = set(X_test_new.columns)
    
    if train_cols != test_cols:
        print("\nCẢNH BÁO: Các cột trong tập huấn luyện và kiểm tra không khớp nhau!")
        print("Các cột chỉ có trong tập huấn luyện:", train_cols - test_cols)
        print("Các cột chỉ có trong tập kiểm tra:", test_cols - train_cols)
    
    return X_train_new, X_test_new

def select_best_features(X_train, X_test, y_train):
    """
    Chọn các đặc trưng quan trọng nhất
    """
    print("\nĐang chọn các đặc trưng quan trọng...")
    
    # Đảm bảo X_train và X_test có cùng các cột
    common_cols = list(set(X_train.columns) & set(X_test.columns))
    if len(common_cols) != len(X_train.columns):
        print(f"Cảnh báo: Chỉ sử dụng {len(common_cols)} đặc trưng chung giữa tập huấn luyện và kiểm tra")
        X_train = X_train[common_cols]
        X_test = X_test[common_cols]
    
    # Sử dụng Random Forest để chọn đặc trưng
    selector = SelectFromModel(
        RandomForestClassifier(n_estimators=100, random_state=42),
        threshold="median"
    )
    
    # Huấn luyện và chọn đặc trưng
    selector.fit(X_train, y_train)
    
    # Áp dụng lên tập huấn luyện và kiểm tra
    X_train_selected = selector.transform(X_train)
    X_test_selected = selector.transform(X_test)
    
    # Lấy tên các đặc trưng được chọn
    selected_features = np.array(common_cols)[selector.get_support()]
    print(f"Số lượng đặc trưng được chọn: {len(selected_features)}")
    print("Các đặc trưng được chọn:")
    for feature in selected_features:
        print(f"- {feature}")
    
    # Chuyển về DataFrame để giữ tên cột
    X_train_selected = pd.DataFrame(X_train_selected, columns=selected_features)
    X_test_selected = pd.DataFrame(X_test_selected, columns=selected_features)
    
    return X_train_selected, X_test_selected

def train_xgboost(X_train_scaled, y_train):
    """
    Huấn luyện mô hình XGBoost
    """
    print("\nĐang huấn luyện XGBoost...")
    
    # Tính class weights để xử lý mất cân bằng dữ liệu
    class_weights = class_weight.compute_class_weight(
        'balanced', classes=np.unique(y_train), y=y_train
    )
    
    # Chuyển đổi class weights thành sample weights
    sample_weights = np.ones(len(y_train))
    for i, cls in enumerate(np.unique(y_train)):
        sample_weights[y_train == cls] = class_weights[i]
    
    # Tạo mô hình XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        objective='multi:softprob',
        num_class=5,
        eval_metric='mlogloss',
        use_label_encoder=False,
        random_state=42
    )
    
    # Huấn luyện mô hình
    xgb_model.fit(
        X_train_scaled, 
        y_train,
        sample_weight=sample_weights,
        verbose=True
    )
    
    # Đánh giá bằng cross-validation
    cv_scores = cross_val_score(xgb_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV score: {cv_scores.mean():.4f}")
    
    return xgb_model

def train_ensemble(X_train_scaled, y_train):
    """
    Huấn luyện mô hình ensemble (kết hợp nhiều mô hình)
    """
    print("\nĐang huấn luyện Ensemble model...")
    
    # Tính class weights để xử lý mất cân bằng dữ liệu
    class_weights = class_weight.compute_class_weight(
        'balanced', classes=np.unique(y_train), y=y_train
    )
    class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}
    
    # Tạo các mô hình thành phần
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight=class_weights_dict,
        random_state=42
    )
    
    gb = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42
    )
    
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        objective='multi:softprob',
        num_class=5,
        eval_metric='mlogloss',
        use_label_encoder=False,
        random_state=42
    )
    
    # Tạo mô hình ensemble
    ensemble = VotingClassifier(
        estimators=[
            ('rf', rf),
            ('gb', gb),
            ('xgb', xgb_model)
        ],
        voting='soft'
    )
    
    # Huấn luyện mô hình
    ensemble.fit(X_train_scaled, y_train)
    
    # Đánh giá bằng cross-validation
    cv_scores = cross_val_score(ensemble, X_train_scaled, y_train, cv=5, scoring='accuracy')
    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV score: {cv_scores.mean():.4f}")
    
    return ensemble

def create_and_train_model(X_train, X_test, y_train):
    """
    Tạo và huấn luyện mô hình
    """
    print("\nĐang tạo và huấn luyện mô hình...")
    
    # Tạo thêm đặc trưng
    X_train_engineered, X_test_engineered = engineer_features(X_train, X_test)
    
    # Chọn các đặc trưng quan trọng
    X_train_selected, X_test_selected = select_best_features(X_train_engineered, X_test_engineered, y_train)
    
    # Tạo bộ chuẩn hóa dữ liệu
    scaler = StandardScaler()
    
    # Tạo bộ mã hóa one-hot cho nhãn
    encoder = OneHotEncoder(sparse=False, categories=[[0, 1, 2, 3, 4]])
    
    # Chuẩn hóa dữ liệu
    X_train_scaled = scaler.fit_transform(X_train_selected)
    X_test_scaled = scaler.transform(X_test_selected)
    
    # Huấn luyện các mô hình
    xgb_model = train_xgboost(X_train_scaled, y_train)
    ensemble_model = train_ensemble(X_train_scaled, y_train)
    
    # Đánh giá từng mô hình trên tập huấn luyện
    models = {
        "XGBoost": xgb_model,
        "Ensemble": ensemble_model
    }
    
    best_model_name = None
    best_train_score = -1
    
    for name, model in models.items():
        train_score = model.score(X_train_scaled, y_train)
        print(f"\nĐộ chính xác trên tập huấn luyện của {name}: {train_score:.4f}")
        
        if train_score > best_train_score:
            best_train_score = train_score
            best_model_name = name
    
    print(f"\nMô hình tốt nhất trên tập huấn luyện: {best_model_name} (độ chính xác: {best_train_score:.4f})")
    
    # Chọn mô hình tốt nhất
    best_model = models[best_model_name]
    
    # Lưu mô hình và bộ chuẩn hóa
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(encoder, ENCODER_PATH)
    
    # Lưu thêm thông tin về các đặc trưng đã chọn
    feature_info = {
        'original_features': list(X_train.columns),
        'engineered_features': list(X_train_engineered.columns),
        'selected_features': list(X_train_selected.columns)
    }
    joblib.dump(feature_info, os.path.join(MODEL_DIR, "feature_info.joblib"))
    
    print(f"\nĐã lưu mô hình tại {MODEL_PATH}")
    print(f"Đã lưu bộ chuẩn hóa tại {SCALER_PATH}")
    print(f"Đã lưu bộ mã hóa nhãn tại {ENCODER_PATH}")
    
    return best_model, scaler, encoder, best_model_name, X_train_selected, X_test_selected

def evaluate_model(model, scaler, encoder, X_test, y_test, model_name):
    """
    Đánh giá mô hình
    """
    print(f"\nĐang đánh giá mô hình {model_name}...")
    
    # Chuẩn hóa dữ liệu test
    X_test_scaled = scaler.transform(X_test)
    
    # Dự đoán
    y_pred = model.predict(X_test_scaled)
    
    # Tính độ chính xác
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Độ chính xác: {accuracy:.4f}")
    
    # Báo cáo phân loại
    class_names = ['Genius', 'Good', 'Average', 'Inaccuracy', 'Mistake']
    print("\nBáo cáo phân loại:")
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Vẽ ma trận nhầm lẫn
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Dự đoán')
    plt.ylabel('Thực tế')
    plt.title(f'Ma trận nhầm lẫn - {model_name}')
    plt.savefig(os.path.join(MODEL_DIR, f'confusion_matrix_advanced.png'))
    
    # Vẽ biểu đồ phân phối nhãn
    plt.figure(figsize=(10, 6))
    ax = sns.countplot(x=y_test)
    plt.xticks(range(5), class_names)
    plt.title('Phân phối nhãn trong tập test')
    plt.savefig(os.path.join(MODEL_DIR, 'label_distribution_advanced.png'))
    
    print(f"\nĐã lưu các biểu đồ đánh giá trong thư mục {MODEL_DIR}")
    
    return accuracy, y_pred

def predict_with_model(model, scaler, encoder, input_features, feature_info):
    """
    Dự đoán với mô hình đã huấn luyện
    """
    # Kiểm tra và đảm bảo input_features có đúng định dạng
    if isinstance(input_features, np.ndarray):
        # Chuyển thành DataFrame với tên cột phù hợp
        input_df = pd.DataFrame([input_features], columns=feature_info['original_features'])
    else:
        input_df = pd.DataFrame([input_features], columns=feature_info['original_features'])
    
    # Tạo thêm đặc trưng mới giống như khi huấn luyện
    input_engineered = input_df.copy()
    
    # 1. Tạo đặc trưng mới: Tỷ lệ matching_tiles_count trên từng layer
    for i in range(4):
        input_engineered[f'matching_ratio_layer_{i}'] = input_df['matching_tiles_layer_{}'.format(i)] / (input_df['matching_tiles_count'] + 1e-6)
    
    # 2. Tạo đặc trưng mới: Tỷ lệ matching_tiles trên layer được chọn
    input_engineered['matching_ratio_selected_layer'] = 0
    layer = int(input_df.iloc[0]['tile_layer'])
    if layer < 4:  # Đảm bảo layer hợp lệ
        input_engineered.loc[0, 'matching_ratio_selected_layer'] = input_df.iloc[0][f'matching_tiles_layer_{layer}'] / (input_df.iloc[0]['matching_tiles_count'] + 1e-6)
    
    # 3. Tạo đặc trưng mới: Tương tác giữa difficulty và matching_tiles_count
    input_engineered['diff_matching_interaction'] = input_df['difficulty_numeric'] * input_df['matching_tiles_count']
    
    # 4. Tạo đặc trưng mới: Tương tác giữa difficulty và collection_fullness
    input_engineered['diff_collection_interaction'] = input_df['difficulty_numeric'] * input_df['collection_fullness']
    
    # 5. Tạo đặc trưng mới: Tổng số matching tiles trên các layer trên layer hiện tại
    input_engineered['tiles_above_current'] = 0
    if layer < 4:  # Đảm bảo layer hợp lệ
        input_engineered.loc[0, 'tiles_above_current'] = sum(input_df.iloc[0][f'matching_tiles_layer_{j}'] for j in range(layer))
    
    # 6. Tạo đặc trưng mới: Bình phương và căn bậc hai của một số đặc trưng quan trọng
    for feature in ['matching_tiles_count', 'unblocks_count']:
        input_engineered[f'{feature}_squared'] = input_df[feature] ** 2
        input_engineered[f'{feature}_sqrt'] = np.sqrt(input_df[feature] + 1e-6)
    
    # Chọn các đặc trưng đã được chọn trong quá trình huấn luyện
    selected_features = feature_info['selected_features']
    input_selected = input_engineered[selected_features]
    
    # Chuẩn hóa dữ liệu đầu vào
    input_scaled = scaler.transform(input_selected)
    
    # Dự đoán xác suất cho mỗi lớp
    probabilities = model.predict_proba(input_scaled)[0]
    
    # Dự đoán lớp
    prediction = model.predict(input_scaled)[0]
    
    # Chuyển từ 0-4 về 1-5
    prediction += 1
    
    # Mã hóa one-hot
    one_hot = np.zeros(5)
    one_hot[prediction-1] = 1
    
    return {
        'prediction': int(prediction),
        'probabilities': probabilities.tolist(),
        'one_hot': one_hot.tolist(),
        'class_name': ['Genius', 'Good', 'Average', 'Inaccuracy', 'Mistake'][prediction-1]
    }

def main():
    """
    Hàm chính
    """
    print("=== HUẤN LUYỆN MÔ HÌNH ĐÁNH GIÁ NƯỚC ĐI NÂNG CAO ===")
    
    # Đọc và chuẩn bị dữ liệu
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    
    # Tạo và huấn luyện mô hình
    model, scaler, encoder, model_name, X_train_selected, X_test_selected = create_and_train_model(X_train, X_test, y_train)
    
    # Đánh giá mô hình
    accuracy, y_pred = evaluate_model(model, scaler, encoder, X_test_selected, y_test, model_name)
    
    # Ví dụ dự đoán
    print("\nVí dụ dự đoán:")
    # Đọc thông tin về đặc trưng
    feature_info = joblib.load(os.path.join(MODEL_DIR, "feature_info.joblib"))
    
    sample_features = X_test.iloc[0].values
    result = predict_with_model(model, scaler, encoder, sample_features, feature_info)
    print(f"Đặc trưng đầu vào: {sample_features}")
    print(f"Dự đoán: {result['prediction']} ({result['class_name']})")
    print(f"Xác suất: {result['probabilities']}")
    print(f"One-hot encoding: {result['one_hot']}")
    
    print("\n=== HOÀN THÀNH HUẤN LUYỆN MÔ HÌNH NÂNG CAO ===")

if __name__ == "__main__":
    main() 
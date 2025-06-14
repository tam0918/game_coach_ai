import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import seaborn as sns

# Đường dẫn đến file dữ liệu và thư mục lưu mô hình
DATA_PATH = "backend/clean_training_data.csv"
MODEL_DIR = "backend/models"
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
MODEL_PATH = os.path.join(MODEL_DIR, "move_classifier.joblib")
ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.joblib")

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

def create_and_train_model(X_train, y_train):
    """
    Tạo và huấn luyện mô hình
    """
    print("\nĐang tạo và huấn luyện mô hình...")
    
    # Tạo bộ chuẩn hóa dữ liệu
    scaler = StandardScaler()
    
    # Tạo bộ mã hóa one-hot cho nhãn
    encoder = OneHotEncoder(sparse=False, categories=[[0, 1, 2, 3, 4]])
    
    # Chuẩn hóa dữ liệu
    X_train_scaled = scaler.fit_transform(X_train)
    
    # Tính class weights để xử lý mất cân bằng dữ liệu
    class_weights = class_weight.compute_class_weight(
        'balanced', classes=np.unique(y_train), y=y_train
    )
    class_weights_dict = {i: class_weights[i] for i in range(len(class_weights))}
    print(f"Class weights: {class_weights_dict}")
    
    # Thử nghiệm với nhiều mô hình khác nhau
    models = {
        "RandomForest": RandomForestClassifier(
            n_estimators=200, 
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight=class_weights_dict,
            random_state=42,
            n_jobs=-1
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        ),
        "NeuralNetwork": MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size=64,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=1000,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
            random_state=42,
            verbose=True
        )
    }
    
    # Đánh giá từng mô hình bằng cross-validation
    print("\nĐánh giá mô hình bằng cross-validation:")
    best_model_name = None
    best_cv_score = -1
    
    for name, model in models.items():
        print(f"\nĐánh giá mô hình {name}...")
        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
        mean_cv_score = cv_scores.mean()
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV score: {mean_cv_score:.4f}")
        
        if mean_cv_score > best_cv_score:
            best_cv_score = mean_cv_score
            best_model_name = name
    
    print(f"\nMô hình tốt nhất theo cross-validation: {best_model_name} (độ chính xác: {best_cv_score:.4f})")
    
    # Huấn luyện mô hình tốt nhất trên toàn bộ tập huấn luyện
    best_model = models[best_model_name]
    best_model.fit(X_train_scaled, y_train)
    
    # Lưu mô hình và bộ chuẩn hóa
    joblib.dump(scaler, SCALER_PATH)
    joblib.dump(best_model, MODEL_PATH)
    joblib.dump(encoder, ENCODER_PATH)
    
    print(f"\nĐã lưu mô hình tại {MODEL_PATH}")
    print(f"Đã lưu bộ chuẩn hóa tại {SCALER_PATH}")
    print(f"Đã lưu bộ mã hóa nhãn tại {ENCODER_PATH}")
    
    return best_model, scaler, encoder

def evaluate_model(model, scaler, encoder, X_test, y_test):
    """
    Đánh giá mô hình
    """
    print("\nĐang đánh giá mô hình...")
    
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
    plt.title('Ma trận nhầm lẫn')
    plt.savefig(os.path.join(MODEL_DIR, 'confusion_matrix.png'))
    
    # Vẽ biểu đồ phân phối nhãn
    plt.figure(figsize=(10, 6))
    sns.countplot(x=y_test, hue=y_test, palette='viridis', legend=False)
    plt.xticks(range(5), class_names)
    plt.title('Phân phối nhãn trong tập test')
    plt.savefig(os.path.join(MODEL_DIR, 'label_distribution.png'))
    
    # Vẽ biểu đồ độ quan trọng của các đặc trưng (nếu là RandomForest)
    if hasattr(model, 'feature_importances_'):
        feature_names = X_test.columns
        feature_importances = model.feature_importances_
        
        plt.figure(figsize=(12, 8))
        indices = np.argsort(feature_importances)[::-1]
        plt.title('Độ quan trọng của các đặc trưng')
        plt.bar(range(X_test.shape[1]), feature_importances[indices], align='center')
        plt.xticks(range(X_test.shape[1]), [feature_names[i] for i in indices], rotation=90)
        plt.tight_layout()
        plt.savefig(os.path.join(MODEL_DIR, 'feature_importance.png'))
    
    print(f"\nĐã lưu các biểu đồ đánh giá trong thư mục {MODEL_DIR}")
    
    return accuracy, y_pred

def predict_with_model(model, scaler, encoder, input_features):
    """
    Dự đoán với mô hình đã huấn luyện
    """
    # Chuẩn hóa dữ liệu đầu vào
    input_scaled = scaler.transform([input_features])
    
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

def analyze_feature_importance(model, X):
    """
    Phân tích độ quan trọng của các đặc trưng
    """
    if hasattr(model, 'feature_importances_'):
        feature_names = X.columns
        feature_importances = model.feature_importances_
        
        # Sắp xếp các đặc trưng theo độ quan trọng
        sorted_idx = np.argsort(feature_importances)[::-1]
        sorted_features = [(feature_names[i], feature_importances[i]) for i in sorted_idx]
        
        print("\nĐộ quan trọng của các đặc trưng:")
        for feature, importance in sorted_features:
            print(f"{feature}: {importance:.4f}")
        
        return sorted_features
    else:
        print("\nMô hình không hỗ trợ tính năng phân tích độ quan trọng của đặc trưng")
        return None

def main():
    """
    Hàm chính
    """
    print("=== HUẤN LUYỆN MÔ HÌNH ĐÁNH GIÁ NƯỚC ĐI ===")
    
    # Đọc và chuẩn bị dữ liệu
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    
    # Tạo và huấn luyện mô hình
    model, scaler, encoder = create_and_train_model(X_train, y_train)
    
    # Đánh giá mô hình
    accuracy, y_pred = evaluate_model(model, scaler, encoder, X_test, y_test)
    
    # Phân tích độ quan trọng của các đặc trưng
    analyze_feature_importance(model, X_test)
    
    # Ví dụ dự đoán
    print("\nVí dụ dự đoán:")
    sample_features = X_test.iloc[0].values
    result = predict_with_model(model, scaler, encoder, sample_features)
    print(f"Đặc trưng đầu vào: {sample_features}")
    print(f"Dự đoán: {result['prediction']} ({result['class_name']})")
    print(f"Xác suất: {result['probabilities']}")
    print(f"One-hot encoding: {result['one_hot']}")
    
    print("\n=== HOÀN THÀNH HUẤN LUYỆN MÔ HÌNH ===")

if __name__ == "__main__":
    main() 
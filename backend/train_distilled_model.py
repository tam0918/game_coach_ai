import os
import pandas as pd
import numpy as np
import joblib
import json
import ast
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator
# <<< NEW IMPORT: For multi-output regression on embeddings >>>
from sklearn.multioutput import MultiOutputRegressor
# <<< NEW IMPORT: For converting text rationale to vector embeddings >>>
from sentence_transformers import SentenceTransformer

# --- Configuration ---
DATA_PATH = "backend/training_data/llm_distilled_training_data.csv"
MODEL_DIR = "backend/distilled_model_v2_rationale" # New model version
MODEL_PATH = os.path.join(MODEL_DIR, "rationale_predictor.joblib")
SYMBOLS = ["🍎", "🍐", "🍊", "🍋", "🍌", "🍉", "🍇", "🍓", "🍒", "🍑", "🥭", "🍍"]
COLLECTION_MAX_SIZE = 7

# <<< NEW: Define the sentence transformer model for creating embeddings >>>
# Using a multilingual model is robust for various text inputs.
RATIONALE_MODEL_NAME = 'paraphrase-multilingual-MiniLM-L12-v2'


# --- Data Loading and Preparation with Rationale Embedding ---
def load_and_prepare_data():
    print(f"--- Loading data from {DATA_PATH} ---")
    df = pd.read_csv(DATA_PATH)
    df.columns = df.columns.str.strip()

    # --- Feature Engineering (Same as before) ---
    df['collection_slot_contents'] = df['collection_slot_contents'].apply(ast.literal_eval)
    df['board_tiles_by_symbol'] = df['board_tiles_by_symbol'].apply(ast.literal_eval)

    slot_count_cols, board_count_cols = [], []
    for symbol in SYMBOLS:
        slot_col, board_col = f'slot_count_{symbol}', f'board_count_{symbol}'
        df[slot_col] = df['collection_slot_contents'].apply(lambda x: x.count(symbol))
        df[board_col] = df['board_tiles_by_symbol'].apply(lambda s_dict: s_dict.get(symbol, 0))
        slot_count_cols.append(slot_col)
        board_count_cols.append(board_col)

    df['total_tiles_in_slots'] = df[slot_count_cols].sum(axis=1)
    df['total_tiles_on_board'] = df[board_count_cols].sum(axis=1)
    df['collection_fullness_ratio'] = df['total_tiles_in_slots'] / COLLECTION_MAX_SIZE

    # --- Rationale Distillation: Create Target Embeddings ---
    print(f"--- Creating Rationale Embeddings using '{RATIONALE_MODEL_NAME}' ---")
    # This might take a moment on the first run as it downloads the model.
    rationale_model = SentenceTransformer(RATIONALE_MODEL_NAME)

    # Ensure rationale is a list of strings
    rationales = df['rationale'].dropna().astype(str).tolist()

    # Generate embeddings
    rationale_embeddings = rationale_model.encode(rationales, show_progress_bar=True)

    # Create a new DataFrame for embeddings
    embedding_df = pd.DataFrame(rationale_embeddings, index=df['rationale'].dropna().index)
    embedding_df.columns = [f'emb_{i}' for i in range(embedding_df.shape[1])]

    # Merge embeddings back into the main DataFrame
    df = df.join(embedding_df)

    # --- Final Data Preparation ---
    # Define features (X) and targets (y)
    # The targets are now the score AND the rationale embeddings
    columns_to_drop = ['move_id', 'rationale', 'collection_slot_contents', 'board_tiles_by_symbol', 'category']
    if 'collection_fullness' in df.columns:
        columns_to_drop.append('collection_fullness')

    df = df.drop(columns=columns_to_drop)
    df = df.dropna(subset=embedding_df.columns) # Drop rows where embedding failed

    print("--- Feature Engineering & Embedding Complete ---")

    target_cols = ['score'] + embedding_df.columns.tolist()
    X = df.drop(columns=target_cols)
    y = df[target_cols]

    print("\n--- Columns and dtypes for Features (X) ---")
    X.info()

    return train_test_split(X, y, test_size=0.2, random_state=42)


# --- Model Training for Rationale Distillation ---
def train_model(X_train, y_train):
    print("\n--- Training Rationale Distillation Model ---")

    categorical_features = ['move_symbol']
    numerical_features = X_train.select_dtypes(include=np.number).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='drop'
    )

    # We use a single LGBMRegressor wrapped in MultiOutputRegressor
    # to predict all target columns (score + all embedding dimensions) simultaneously.
    # <<< CHANGE: Increased model complexity to try and learn deeper patterns >>>
    lgbm_reg = LGBMRegressor(
        random_state=42,
        n_estimators=800, # Increased from 500
        learning_rate=0.05,
        num_leaves=50,    # Increased from 40
        reg_alpha=0.1,
        reg_lambda=0.1,
        colsample_bytree=0.8
    )

    # The core of the new approach: predict multiple targets
    multi_output_model = MultiOutputRegressor(estimator=lgbm_reg, n_jobs=-1)

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('multi_output_regressor', multi_output_model)
    ])

    model_pipeline.fit(X_train, y_train)

    print("--- Model Training Complete ---")
    return model_pipeline

# --- Evaluation for Rationale Distillation ---
def evaluate_model(model, X_test, y_test):
    print("\n--- Evaluating Rationale Distillation Performance ---")

    y_pred = model.predict(X_test)

    # Convert predictions and true values to DataFrames for easier indexing
    y_pred_df = pd.DataFrame(y_pred, columns=y_test.columns, index=y_test.index)

    # --- Evaluate Score Prediction ---
    from sklearn.metrics import mean_squared_error, r2_score
    score_true = y_test['score']
    score_pred = y_pred_df['score']
    print("\n**Regression Performance (Score)**")
    print(f"Mean Squared Error: {mean_squared_error(score_true, score_pred):.4f}")
    print(f"R-squared: {r2_score(score_true, score_pred):.4f}")

    # --- Evaluate Rationale Embedding Prediction ---
    from sklearn.metrics.pairwise import cosine_similarity
    emb_cols = [col for col in y_test.columns if col.startswith('emb_')]
    emb_true = y_test[emb_cols].values
    emb_pred = y_pred_df[emb_cols].values

    # Calculate cosine similarity for each row
    similarities = []
    for i in range(len(emb_true)):
        sim = cosine_similarity(emb_true[i].reshape(1, -1), emb_pred[i].reshape(1, -1))
        similarities.append(sim[0, 0])

    avg_cosine_similarity = np.mean(similarities)
    print("\n**Embedding Prediction Performance (Rationale)**")
    print(f"Average Cosine Similarity: {avg_cosine_similarity:.4f}")
    print("(A value closer to 1.0 means the model's 'reasoning' is very similar to the LLM's)")


# --- Main Execution ---
def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    X_train, X_test, y_train, y_test = load_and_prepare_data()

    model = train_model(X_train, y_train)

    evaluate_model(model, X_test, y_test)

    joblib.dump(model, MODEL_PATH)
    print(f"\n✅ Rationale distillation model saved to: {MODEL_PATH}")

if __name__ == "__main__":
    main()

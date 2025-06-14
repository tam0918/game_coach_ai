import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple, Dict, Any
import numpy as np

class DataProcessor:
    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        self.feature_scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.columns_after_fit = None
        self.encoded_class_names = None

    def _preprocess_features(self, df: pd.DataFrame, is_fitting: bool) -> pd.DataFrame:
        """Applies all feature transformations in a robust manner."""
        # 0. Drop identifier columns that are not useful for training
        ids_to_drop = ['game_state_id', 'move_id', 'selected_tile_id', 'timestamp']
        df = df.drop(columns=[col for col in ids_to_drop if col in df.columns], errors='ignore')

        # 1. Encode difficulty levels numerically (1: easy, 2: medium, 3: hard, 4: hell)
        if 'difficulty' in df.columns:
            difficulty_mapping = {'easy': 1, 'medium': 2, 'hard': 3, 'hell': 4}
            df['difficulty_encoded'] = df['difficulty'].map(difficulty_mapping)
            # Drop the original difficulty column since we've encoded it
            df = df.drop(columns=['difficulty'])

        # 2. Handle the special JSON string column 'gs_accessible_symbols'
        if 'gs_accessible_symbols' in df.columns:
            df['gs_accessible_symbols'] = df['gs_accessible_symbols'].apply(json.loads)
            
            # Count similar tiles in each accessible symbols list
            # This helps the model understand matching potential
            symbol_counts = df['gs_accessible_symbols'].apply(pd.Series).stack().value_counts().index
            symbol_df = pd.DataFrame(0, index=df.index, columns=[f'sym_count_{s}' for s in symbol_counts])
            
            # Count how many tiles match the selected tile in the accessible set
            if 'move_tile_symbol' in df.columns:
                df['matching_tiles_count'] = df.apply(
                    lambda row: row['gs_accessible_symbols'].count(row['move_tile_symbol']), 
                    axis=1
                )
            
            for i, row in df.iterrows():
                for symbol in row['gs_accessible_symbols']:
                    col_name = f'sym_count_{symbol}'
                    if col_name in symbol_df.columns:
                        symbol_df.at[i, col_name] += 1
            
            df = pd.concat([df.drop('gs_accessible_symbols', axis=1), symbol_df], axis=1)

        # 3. Calculate queue fullness (0-7 scale)
        if 'gs_collection_fill_ratio' in df.columns:
            # Convert collection fill ratio (0-1) to a 0-7 scale
            df['collection_fullness'] = (df['gs_collection_fill_ratio'] * 7).round().astype(int)

        # 4. Calculate win/loss binary feature (0: loss, 1: win)
        # Based on move_outcome field - immediate_loss is 0, others are 1 (ongoing)
        if 'move_outcome' in df.columns:
            df['win_loss'] = df['move_outcome'].apply(lambda x: 0 if x == 'immediate_loss' else 1)

        # 5. Count unlockable tiles (tiles that become accessible after making this move)
        if 'move_unblocks_potential' in df.columns:
            df['unlocks_count'] = df['move_unblocks_potential']

        # 6. Extract layers information
        if 'move_tile_layer' in df.columns:
            # Create counts of matching tiles by layer
            for layer in range(4):  # Layers 0-3
                # Calculate the number of matching tiles in this layer
                df[f'matching_tiles_layer_{layer}'] = 0
                # We'd need to implement the actual counting logic based on available data

        # 7. Separate columns by data type for robust processing
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_cols = df.select_dtypes(include=np.number).columns.tolist()

        # 8. One-Hot Encode any remaining categorical features
        if categorical_cols:
            df_encoded = pd.get_dummies(df[categorical_cols], prefix=categorical_cols, dtype=float)
        else:
            df_encoded = pd.DataFrame(index=df.index)

        # 9. Scale numerical features
        df_scaled = df[numerical_cols].copy()
        if is_fitting:
            df_scaled[numerical_cols] = self.feature_scaler.fit_transform(df_scaled[numerical_cols])
        else:
            # Check if scaler is fitted
            if hasattr(self.feature_scaler, 'scale_'):
                df_scaled[numerical_cols] = self.feature_scaler.transform(df_scaled[numerical_cols])
        
        # 10. Combine processed columns
        processed_df = pd.concat([df_scaled, df_encoded], axis=1)

        # 11. Manage columns for consistency between training and prediction
        if is_fitting:
            self.columns_after_fit = processed_df.columns.tolist()
        else:
            if self.columns_after_fit:
                # Add missing columns that were in training but not in current data
                missing_cols = set(self.columns_after_fit) - set(processed_df.columns)
                for c in missing_cols:
                    processed_df[c] = 0
                # Ensure the order of columns is the same as in training
                processed_df = processed_df[self.columns_after_fit]
            
        return processed_df

    def _preprocess_targets(self, df: pd.DataFrame, is_fitting: bool) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocesses the target variables."""
        # 1. Classification target - Remap to numerical values 1-5
        # "Genius/Excellent": 1, "Good": 2, "Average": 3, "Inaccuracy": 4, "Blunder/Stupid": 5
        classification_map = {
            "Genius/Excellent": 1,
            "Good": 2,
            "Average": 3, 
            "Inaccuracy": 4,
            "Mistake": 5,
            "Blunder/Stupid": 5
        }
        
        # Map the classification target
        classification_target = df['llm_classification'].map(classification_map)
        
        if is_fitting:
            encoded_labels = self.label_encoder.fit_transform(classification_target)
            self.encoded_class_names = self.label_encoder.classes_
        else:
            encoded_labels = self.label_encoder.transform(classification_target)
        
        # 2. Regression targets
        regression_targets = [
            'llm_quality_score', 'llm_strategy_score', 'llm_risk_score',
            'llm_efficiency_score', 'llm_combo_potential_score'
        ]
        regression_values = df[regression_targets].values
        
        return encoded_labels, regression_values

    def get_processed_data(self) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Cleans the data and returns features and targets."""
        # Drop rows with missing target values
        target_cols = ['llm_classification', 'llm_quality_score']
        self.df.dropna(subset=target_cols, inplace=True)
        
        # Define feature columns (everything except the LLM's direct output)
        llm_output_cols = [col for col in self.df.columns if col.startswith('llm_')]
        feature_cols = self.df.drop(columns=llm_output_cols)
        
        # Process features and targets
        X = self._preprocess_features(feature_cols, is_fitting=True)
        y_class, y_reg = self._preprocess_targets(self.df, is_fitting=True)
        
        return train_test_split(X, y_class, y_reg, test_size=0.2, random_state=42)

if __name__ == '__main__':
    # Example usage:
    DATA_PATH = '../simulation_results.csv'
    processor = DataProcessor(DATA_PATH)
    X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = processor.get_processed_data()
    
    print("Data processing complete.")
    print("Training Features Shape:", X_train.shape)
    print("Testing Features Shape:", X_test.shape)
    print("Training Classification Labels Shape:", y_class_train.shape)
    print("Training Regression Labels Shape:", y_reg_train.shape)
    print("Encoded Classes:", processor.encoded_class_names) 
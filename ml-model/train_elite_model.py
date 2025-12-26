import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, roc_auc_score, 
                             confusion_matrix, classification_report)
import xgboost as xgb
import pickle
import json
import warnings
warnings.filterwarnings('ignore')


class EliteModelTrainer:
    """Train XGBoost model on elite features
    
    XGBoost Documentation Reference:
    - XGBClassifier.fit() signature: fit(X, y, *, sample_weight=None, base_margin=None, 
                                         eval_set=None, eval_metric=None, early_stopping_rounds=None, ...)
    - early_stopping_rounds: MUST be passed to constructor, NOT fit()
    - eval_set: Passed to fit() as list of (X, y) tuples for validation
    - eval_metric: Can be in constructor OR fit()
    
    Reference: https://xgboost.readthedocs.io/en/stable/python/python_intro.html
    """
    
    def __init__(self, features_csv):
        self.df = pd.read_csv(features_csv)
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.X_test = None
        self.y_test = None
        
        print(f"✓ Loaded {len(self.df)} samples from {features_csv}")
        print(f"✓ Columns in engineered CSV: {list(self.df.columns)}")
    
    def prepare_data(self):
        """Prepare features and target"""
        print("\n📊 Preparing data...")
        
        self.df = self.df.dropna()
        print(f"✓ Samples after dropping NaN: {len(self.df)}")
        
        # Exclude non-feature columns
        # These are the engineered features CSV columns, NOT the raw CSV columns
        exclude_cols = ['GAME_ID', 'GAME_DATE_EST', 'HOME_TEAM_ID', 'VISITOR_TEAM_ID', 
                       'PTS_home', 'PTS_away', 'TARGET', 'SEASON',
                       'GAME_STATUS_TEXT', 'TEAM_ID_home', 'TEAM_ID_away', 'FG_PCT_home',
                       'FT_PCT_home', 'FG3_PCT_home', 'AST_home', 'REB_home', 'FG_PCT_away',
                       'FT_PCT_away', 'FG3_PCT_away', 'AST_away', 'REB_away', 'HOME_TEAM_WINS']
        
        self.feature_columns = [col for col in self.df.columns 
                               if col not in exclude_cols and col != 'TARGET']
        
        print(f"\n✓ Feature columns ({len(self.feature_columns)}):")
        for i, col in enumerate(self.feature_columns, 1):
            print(f"    {i}. {col}")
        
        X = self.df[self.feature_columns].copy()
        y = self.df['TARGET'].copy()
        
        # Split: 80% train, 20% test (chronological for sports)
        split_idx = int(len(X) * 0.8)
        X_train = X[:split_idx]
        X_test = X[split_idx:]
        y_train = y[:split_idx]
        y_test = y[split_idx:]
        
        self.X_test = X_test
        self.y_test = y_test
        
        print(f"\n✓ Data split:")
        print(f"    Train: {len(X_train)} samples ({100*len(X_train)/len(X):.1f}%)")
        print(f"    Test:  {len(X_test)} samples ({100*len(X_test)/len(X):.1f}%)")
        print(f"    Train target: {y_train.value_counts().to_dict()}")
        print(f"    Test target:  {y_test.value_counts().to_dict()}")
        
        return X_train, X_test, y_train, y_test
    
    def train(self):
        """Train XGBoost with elite hyperparameters
        
        CRITICAL FIX: early_stopping_rounds MUST be in XGBClassifier constructor
        NOT in the fit() method. This was the source of the TypeError.
        
        Correct Pattern:
            model = XGBClassifier(..., early_stopping_rounds=15)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        
        Reference: https://stackoverflow.com/questions/78713048/
        """
        print("\n" + "="*70)
        print("🤖 TRAINING ELITE XGBOOST MODEL")
        print("="*70)
        
        X_train, X_test, y_train, y_test = self.prepare_data()
        
        # Scale features
        print("\n📍 Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Elite hyperparameters optimized for sports prediction
        print("\n⚙️  Initializing XGBoost with elite parameters...")
        self.model = xgb.XGBClassifier(
            # Tree building
            n_estimators=200,           # More trees for complex features
            max_depth=6,                # Slightly deeper for interactions
            learning_rate=0.08,         # Slower learning, more stable
            subsample=0.85,             # 85% samples per tree
            colsample_bytree=0.85,      # 85% features per tree
            colsample_bylevel=0.85,     # 85% features per level
            
            # Regularization
            reg_alpha=0.1,              # L1 regularization
            reg_lambda=0.1,             # L2 regularization
            gamma=1.0,                  # Min loss reduction
            
            # Early stopping (MUST be in constructor, not fit())
            early_stopping_rounds=15,   # Stop if no improvement for 15 rounds
            
            # Training
            eval_metric='logloss',      # Binary classification metric
            use_label_encoder=False,    # Modern sklearn compatibility
            random_state=42,            # Reproducibility
            n_jobs=-1,                  # Use all CPU cores
            verbose=1                   # Print training progress
        )
        
        # Train model with validation set
        print("\n🔥 Training...")
        self.model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_test_scaled, y_test)],  # Validation set for early stopping
            verbose=True
        )
        
        # Predict
        print("\n📈 Evaluating on test set...")
        y_pred = self.model.predict(X_test_scaled)
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        print("\n" + "="*70)
        print("✓ MODEL TRAINING COMPLETE")
        print("="*70)
        print(f"📊 Accuracy:      {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"📊 ROC-AUC:       {roc_auc:.4f}")
        print(f"📊 Sensitivity:   {sensitivity:.4f}")
        print(f"📊 Specificity:   {specificity:.4f}")
        print(f"📊 Best iteration: {self.model.best_iteration}")
        print(f"📊 Best score:    {self.model.best_score:.4f}")
        print("="*70 + "\n")
        
        # Feature importance
        self.print_feature_importance()
        
        # Save metrics
        self.save_metrics(accuracy, roc_auc, sensitivity, specificity)
        
        return accuracy, roc_auc
    
    def print_feature_importance(self):
        """Display top features by importance"""
        print("\n🏆 FEATURE IMPORTANCE (Top 10)")
        print("-" * 50)
        
        importance = self.model.feature_importances_
        features_importance = sorted(
            zip(self.feature_columns, importance),
            key=lambda x: x[1],
            reverse=True
        )
        
        total_importance = sum(importance)
        
        for i, (feat, imp) in enumerate(features_importance[:10], 1):
            pct = (imp / total_importance) * 100
            bar = "█" * int(pct / 2)
            print(f"{i:2d}. {feat:25s} {pct:6.2f}% {bar}")
    
    def save_metrics(self, accuracy, roc_auc, sensitivity, specificity):
        """Save training metrics to JSON"""
        metrics = {
            'accuracy': float(accuracy),
            'roc_auc': float(roc_auc),
            'sensitivity': float(sensitivity),
            'specificity': float(specificity),
            'train_size': int(len(self.df) * 0.8),
            'test_size': int(len(self.df) * 0.2),
            'features': self.feature_columns,
            'feature_count': len(self.feature_columns),
            'best_iteration': int(self.model.best_iteration) if hasattr(self.model, 'best_iteration') else 0,
            'best_score': float(self.model.best_score) if hasattr(self.model, 'best_score') else 0.0,
            'model_type': 'XGBoost Elite',
            'model_version': '2.1',
            'early_stopping_rounds': 15
        }
        
        with open('models/metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        print(f"💾 Metrics saved to models/metrics.json")
    
    def save_model(self):
        """Save trained model and scaler"""
        print("\n💾 Saving model artifacts...")
        
        # Model
        with open('models/xgboost_elite_model.pkl', 'wb') as f:
            pickle.dump(self.model, f)
        
        # Scaler
        with open('models/scaler.pkl', 'wb') as f:
            pickle.dump(self.scaler, f)
        
        # Features list
        with open('models/feature_columns.json', 'w') as f:
            json.dump(self.feature_columns, f, indent=2)
        
        print(f"✓ Model saved: models/xgboost_elite_model.pkl")
        print(f"✓ Scaler saved: models/scaler.pkl")
        print(f"✓ Features saved: models/feature_columns.json")


# EXECUTION
if __name__ == '__main__':
    trainer = EliteModelTrainer('data/nba_games_elite.csv')
    trainer.train()
    trainer.save_model()
    
    print("\n" + "="*70)
    print("🎉 ALL DONE! Model ready for predictions")
    print("="*70)
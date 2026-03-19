import numpy as np
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_squared_error

def rmse(y_true, y_pred):
    """Root Mean Squared Error — competition metric."""
    return np.sqrt(mean_squared_error(y_true, np.maximum(y_pred, 0)))

def train_lgbm_twostep(X_tr, y_tr, y_tr_clf, X_val, y_val, y_val_clf, X_test, lgbm_reg_params, lgbm_clf_params):
    """
    Two-Step LightGBM Strategy:
    1. Classify buyer vs non-buyer.
    2. Regress revenue for predicted buyers.
    """
    # Step 1: Classification
    clf = lgb.LGBMClassifier(**lgbm_clf_params)
    clf.fit(X_tr, y_tr_clf, eval_set=[(X_val, y_val_clf)], 
            callbacks=[lgb.early_stopping(50, verbose=False)])

    # Step 2: Regression (on buyers only)
    buyer_mask_tr = y_tr_clf == 1
    reg = lgb.LGBMRegressor(**lgbm_reg_params)
    reg.fit(X_tr[buyer_mask_tr], y_tr[buyer_mask_tr], 
            eval_set=[(X_val[y_val_clf == 1], y_val[y_val_clf == 1])],
            callbacks=[lgb.early_stopping(50, verbose=False)])

    # Combined Prediction
    val_prob   = clf.predict_proba(X_val)[:, 1]
    val_amount = np.maximum(reg.predict(X_val), 0)
    val_pred   = val_prob * val_amount

    test_prob   = clf.predict_proba(X_test)[:, 1]
    test_amount = np.maximum(reg.predict(X_test), 0)
    test_pred   = test_prob * test_amount

    return (clf, reg), val_pred, test_pred

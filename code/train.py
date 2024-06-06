def find_optimal_threshold(model, x, y):
    import numpy as np
    import scipy
    from sklearn.metrics import accuracy_score, f1_score
    
    def thr_to_accuracy(thr, y_true, y_pred):
        return -f1_score(y_true, np.array(y_pred>thr, dtype=np.int64))
    
    y_pred = model.predict_proba(x)[:,1]
    best_thr = scipy.optimize.fmin(thr_to_accuracy, args=(y, y_pred), x0=0.5)
    return best_thr[0]
    

def train_model(df_input, features, target_col):
    import logging

    import numpy as np
    import pandas as pd

    from lightgbm import LGBMClassifier

    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    
    x, y = df_input[features], df_input[target_col]
    
    model = LGBMClassifier(is_unbalance = True)

    # Use evaluation metrics suited for imbalanced classification.
    scoring = {"PRECISION": "precision", "RECALL":"recall", "F1": "f1", "AUC": "roc_auc"}

    # Train the model using the optimal parameters identified during the training process.
    # In this case, i use a minimal set of parameter combinations to reduce training time.
    param_grid = {'learning_rate': [0.05], 'n_estimators': [150], 'num_leaves': [5]}

    model_gs = GridSearchCV(model,
                            param_grid = param_grid,
                            cv = StratifiedKFold(),
                            scoring = scoring,
                            refit="AUC",
                            return_train_score=True)

    logging.info("Start training Model ...")
    model_gs.fit(x, y)
    optimal_threshold = find_optimal_threshold(model_gs.best_estimator_, x, y)

    #Step 4: return trained model and preprocessing models that will be stored later.
    train_models = {"MODEL_GS":model_gs,
                    "METADATA": {"TARGET": target_col,
                                 "FEATURES": features,
                                 "OPTIMAL_THRESHOLD":optimal_threshold}
                    }
    
    logging.info("Finished training process.")
    return train_models



def save_trained_model(train_models, model_folder):
    import os
    import logging
    from pickle import dump

    # Save ml model
    model_path = os.path.join(model_folder, 'model_grid_search.pkl')
    logging.info("Saving model in %s...", model_path)
    dump(train_models["MODEL_GS"], open(model_path, 'wb'))

    # Save metadata
    metadata_path = os.path.join(model_folder, 'metadata.pkl')
    logging.info("Saving metadata in %s...", metadata_path)
    dump(train_models["METADATA"], open(metadata_path, 'wb'))

    return None



    
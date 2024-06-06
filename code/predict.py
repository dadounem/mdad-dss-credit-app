def make_prediction(df, model_path, f_return_input = True, f_return_proba = True):
    import logging
    import os
    import pandas as pd
    from pickle import load
    

    logging.info("Loading model ...")
    model_gs = load(open(os.path.join(model_path, "model_grid_search.pkl"), 'rb'))
    metadata = load(open(os.path.join(model_path, "metadata.pkl"), 'rb'))
    optimal_threshold = metadata["OPTIMAL_THRESHOLD"]
    #Extract the optimal model from grid search
    model = model_gs.best_estimator_
    features = model.feature_name_
    
    df_prediction = pd.DataFrame()
    if f_return_input:
        df_prediction = df.copy()
    
    proba = model.predict_proba(df[features])[:,1]
    df_prediction["PREDICTION"] = (proba > optimal_threshold)*1

    if f_return_proba:
        df_prediction["PREDICTION_PROBA_0"] = model.predict_proba(df[features])[:,0]
        df_prediction["PREDICTION_PROBA_1"] = model.predict_proba(df[features])[:,1]

    return df_prediction
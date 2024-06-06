def list_models(root_folder):
    import os
    models = [m for m in os.listdir(root_folder) if "model" in m]
    models = sorted(models,reverse = True)
    return models

def extract_scores(model_gs):
    import numpy as np
    
    #Get evaluation metrics used during training process.
    score_metrics = list(model_gs.scoring.keys())

    #Calculate mean cv score for every metric. 
    cv_results = model_gs.cv_results_
    scores = {s:np.mean(cv_results["mean_test_{}".format(s)]) for s in score_metrics} 
    return scores

def get_hist_model_scores(models_folder):
    import os
    import pandas as pd
    from pickle import load
    
    models = list_models(models_folder)
    scores = {}
    for m in models:
        try:
            model_gs = load(open(os.path.join(models_folder, m, "model_grid_search.pkl"), 'rb'))
            m_scores = extract_scores(model_gs)
            scores[m] = m_scores
        except:
            pass
        
    df_scores = pd.DataFrame(scores).T
    return df_scores



def get_features_from_last_model(models_folder):
    import os
    from pickle import load

    models = list_models(models_folder)
    if len(models) > 0:
        last_model = models[0]

        # Extract metadata
        metadata_path = os.path.join(models_folder, last_model, "metadata.pkl")
        metadata = load(open(metadata_path, 'rb'))
        return metadata
    
    return None

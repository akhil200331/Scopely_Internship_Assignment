import pandas as pd
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix

def update(Y_test, Y_pred, data_frame, model_name):
    # Create a new DataFrame with the new metrics
    new_rows = [[model_name, accuracy_score(Y_test, Y_pred), recall_score(Y_test, Y_pred), precision_score(Y_test, Y_pred), f1_score(Y_test, Y_pred)]]
    new_df = pd.DataFrame(new_rows, columns=list(data_frame.columns))
    
    # Concatenate the new DataFrame with the existing one and return the updated DataFrame
    updated_df = pd.concat([data_frame, new_df], ignore_index=True)
    
    return updated_df
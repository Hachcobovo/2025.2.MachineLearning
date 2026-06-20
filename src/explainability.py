import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

def plot_feature_importance(model, feature_names, top_n=20):
    importances = model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False)
    
    print(f"\nTop {top_n} Most Important Features:")
    print(importance_df.head(top_n).to_string(index=False))
    
    plt.figure(figsize=(12, 8))
    plt.barh(importance_df['Feature'].head(top_n)[::-1], importance_df['Importance'].head(top_n)[::-1], color='skyblue')
    plt.xlabel('Importance Score')
    plt.title(f'Top {top_n} Features for Attack Detection')
    plt.tight_layout()
    plt.show()

def plot_shap_summary(model, X_sample, feature_names):
    if hasattr(X_sample, "toarray"):
        X_dense = X_sample.toarray()
    else:
        X_dense = X_sample
        
    X_df = pd.DataFrame(X_dense, columns=feature_names)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_df)

    if isinstance(shap_values, list) and len(shap_values) == 2:
        shap_values_to_plot = shap_values[1]
    else:
        shap_values_to_plot = shap_values
        
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_to_plot, X_df, plot_type="dot", show=False)
    plt.title('SHAP Summary Plot (Impact on Attack Prediction)')
    plt.tight_layout()
    plt.show()

def analyze_false_positives_negatives(model, X_test, y_test, raw_texts=None, num_samples=5):
    y_pred = model.predict(X_test)
    fp_indices = np.where((y_pred == 1) & (y_test == 0))[0]
    fn_indices = np.where((y_pred == 0) & (y_test == 1))[0]

    print(f"\n[FALSE POSITIVES] Total: {len(fp_indices)} (Predicted Attack, Actually Normal)")
    for i, idx in enumerate(fp_indices[:num_samples]):
        print(f"\n--- FP Example {i+1} (Index: {idx}) ---")
        if raw_texts is not None:
            text = raw_texts.iloc[idx] if hasattr(raw_texts, 'iloc') else raw_texts[idx]
            print(f"Request Text:\n{text}")
        else:
            print("Raw text not provided")

    print(f"\n\n[FALSE NEGATIVES] Total: {len(fn_indices)} (Predicted Normal, Actually Attack)")
    for i, idx in enumerate(fn_indices[:num_samples]):
        print(f"\n--- FN Example {i+1} (Index: {idx}) ---")
        if raw_texts is not None:
            text = raw_texts.iloc[idx] if hasattr(raw_texts, 'iloc') else raw_texts[idx]
            print(f"Request Text:\n{text}")
        else:
            print("Raw text not provided")
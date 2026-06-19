import pandas as pd
import matplotlib.pyplot as plt

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
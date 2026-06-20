"""
evaluation.py
"""
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score,precision_score,recall_score,f1_score,roc_auc_score,confusion_matrix,classification_report,roc_curve,ConfusionMatrixDisplay)
# Evaluate classifier
def evaluate_model(model,X_test,y_test,model_name="Model",training_time=None):
    # Inference time
    start=time.time()
    y_pred=model.predict(X_test)
    inference_time=time.time()-start
    # Probability
    if hasattr(model,"predict_proba"):
        y_prob=model.predict_proba(X_test)[:,1]
    else:
        y_prob=None
    # Metrics
    acc=accuracy_score(y_test,y_pred)
    precision=precision_score(y_test,y_pred,zero_division=0)
    recall=recall_score(y_test,y_pred,zero_division=0)
    f1=f1_score(y_test,y_pred,zero_division=0)
    if y_prob is not None:
        auc=roc_auc_score(y_test,y_prob)
    else:
        auc=None
    # Confusion matrix metrics:false negative rate and false positive rate
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    results={"Model":model_name,"Accuracy":acc,"Precision":precision,"Recall":recall,"F1":f1,"ROC-AUC":auc,"False positive rate":fpr,"False negative rate":fnr,"Training_time(s)":training_time,"Inference_time(s)":inference_time}
    # Add number of trees if the model supports it
    if hasattr(model,"n_estimators"):
        results["Trees"]=model.n_estimators
    return results
# Confusion Matrix
def plot_confusion_matrix(model, X_test, y_test, title="Confusion Matrix"):
    y_pred = model.predict(X_test)
    ConfusionMatrixDisplay.from_predictions(y_test,y_pred,display_labels=["Normal", "Attack"],cmap="Blues",values_format="d")
    plt.title(title)
    plt.show()
# ROC Curve
def plot_roc_curve(models,X_test,y_test):
    plt.figure(figsize=(7,5))
    for name,model in models.items():
        if not hasattr(model,"predict_proba"):
            continue
        prob=model.predict_proba(X_test)[:,1]
        fpr,tpr,_=roc_curve(y_test,prob)
        auc=roc_auc_score(y_test,prob)
        plt.plot(fpr,tpr,label=f"{name} AUC={auc:.3f}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()
    plt.show()
# Precision Recall Curve
def plot_pr_curve(models, X_test, y_test):
    plt.figure(figsize=(7,5))
    for name, model in models.items():
        if not hasattr(model, "predict_proba"):
            continue
        prob = model.predict_proba(X_test)[:,1]
        precision, recall, _ = precision_recall_curve(y_test,prob)
        plt.plot(recall,precision,label=name)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.legend()
    plt.show()
# Full report
def print_report(model,X_test,y_test):
    y_pred=model.predict(X_test)
    print(classification_report(y_test,y_pred,target_names=["Normal","Malicious"],zero_division=0))
# Save result
def save_results(results,path="results.csv"):
    pd.DataFrame([results]).to_csv(path,index=False)
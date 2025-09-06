# Figure 6 – Hyper-parameter tuning + 10-fold ROC
# Adapt paths or mount your drive accordingly.
import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

plt.rcParams.update({"savefig.dpi": 300, "axes.grid": False})

# ---- Inputs ----
rf   = pd.read_csv("data/rf_numtrees_varspl.csv").rename(
        columns={"numberOfTrees":"n_trees", "variablesPerSplit":"mtry"})
svm_hyp = pd.read_csv("data/svm_kernel_cost.csv").rename(
          columns={"kernelType":"kernel"})
cart = pd.read_csv("data/cart_maxnodes.csv").rename(
         columns={"maxNodes":"max_nodes"})
pts  = pd.read_csv("data/sampled_normalized_parameters.csv")

META = ['landcover','random','system:index','.geo']
X = pts.drop(columns=[c for c in META if c in pts.columns if c in pts])
X = X.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')
mask = ~X.isna().any(axis=1)
X = X[mask].reset_index(drop=True)
y = pts.loc[mask, 'landcover'].reset_index(drop=True)

best_rf   = rf.loc[rf['accuracy'].idxmax()]
rf_model  = RandomForestClassifier(
              n_estimators=int(best_rf['n_trees']),
              max_features=int(best_rf['mtry']) if str(best_rf['mtry']).isdigit() else None,
              n_jobs=-1, random_state=42, class_weight="balanced")

best_svm  = svm_hyp.loc[svm_hyp['accuracy'].idxmax()]
svm_model = SVC(kernel=str(best_svm['kernel']).lower(),
                C=float(best_svm['cost']),
                probability=True, random_state=42, class_weight="balanced")

best_cart = cart.loc[cart['accuracy'].idxmax()]
cart_model= DecisionTreeClassifier(
              max_leaf_nodes=int(best_cart['max_nodes']),
              random_state=42, class_weight="balanced")

def draw_cv_mean_roc(ax, model, name):
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    xs = np.linspace(0, 1, 300)
    tprs, aucs = [], []
    classes = sorted(np.unique(y))
    for idx, (tr, te) in enumerate(cv.split(X, y), 1):
        model.fit(X.iloc[tr], y.iloc[tr])
        proba = model.predict_proba(X.iloc[te])
        fpr, tpr, _ = roc_curve(
            label_binarize(y.iloc[te], classes=classes).ravel(),
            proba.ravel())
        auc_fold = auc(fpr, tpr); aucs.append(auc_fold)
        tprs.append(np.interp(xs, fpr, tpr)); tprs[-1][0] = 0.0
        ax.plot(fpr, tpr, lw=0.7, alpha=0.25, label=f"Fold {idx}")
    mean_tpr = np.mean(tprs, axis=0); mean_tpr[-1] = 1.0
    std_tpr  = np.std(tprs,  axis=0)
    ax.plot(xs, mean_tpr, lw=2, label=f"Mean AUC={np.mean(aucs):.3f}")
    ax.fill_between(xs, mean_tpr-std_tpr, mean_tpr+std_tpr, alpha=0.2, label='±1 STD')
    ax.plot([0,1],[0,1],'--', lw=0.8, color='black', label='Chance (0.5)')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("False-Positive Rate"); ax.set_ylabel("True-Positive Rate")
    ax.set_title(name); ax.legend(fontsize=6, frameon=False)

fig, axs = plt.subplots(1, 3, figsize=(12,4))
draw_cv_mean_roc(axs[0], rf_model,   "RF | 10-fold ROC")
draw_cv_mean_roc(axs[1], svm_model,  "SVM | 10-fold ROC")
draw_cv_mean_roc(axs[2], cart_model, "CART | 10-fold ROC")
plt.tight_layout()
plt.savefig("fig/fig6_hyperparam_roc.png", dpi=300)
print("Saved: fig/fig6_hyperparam_roc.png")

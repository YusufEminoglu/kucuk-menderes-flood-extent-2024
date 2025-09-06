#!/usr/bin/env python3
import warnings, argparse, numpy as np, pandas as pd, seaborn as sns, matplotlib.pyplot as plt
import scipy.interpolate as si
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

def load_csv(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df

def prep_features(pts, label_col='landcover', drop_meta=('landcover','random','system:index','.geo')):
    X = pts.drop(columns=[c for c in drop_meta if c in pts.columns], errors='ignore')
    X = X.apply(pd.to_numeric, errors='coerce').dropna(axis=1, how='all')
    mask = ~X.isna().any(axis=1)
    X = X.loc[mask].reset_index(drop=True)
    y = pts.loc[mask, label_col].reset_index(drop=True)
    return X, y

def best_models(rf, svm, cart, X, y):
    rf = rf.rename(columns={'numberOfTrees':'n_trees','variablesPerSplit':'mtry','accuracy':'accuracy'})
    svm = svm.rename(columns={'kernelType':'kernel','accuracy':'accuracy'})
    cart = cart.rename(columns={'maxNodes':'max_nodes','accuracy':'accuracy'})
    brf  = rf.loc[rf['accuracy'].idxmax()]
    bsvm = svm.loc[svm['accuracy'].idxmax()]
    bcart= cart.loc[cart['accuracy'].idxmax()]
    rf_model  = RandomForestClassifier(
        n_estimators=int(brf['n_trees']), max_features=int(brf['mtry']),
        n_jobs=-1, random_state=42, class_weight='balanced')
    svm_model = SVC(kernel=str(bsvm['kernel']).lower(), C=float(bsvm['cost']),
                    probability=True, random_state=42, class_weight='balanced')
    cart_model= DecisionTreeClassifier(max_leaf_nodes=int(bcart['max_nodes']),
                                       random_state=42, class_weight='balanced')
    return (brf, bsvm, bcart), (rf_model, svm_model, cart_model)

def draw_cv_mean_roc(ax, model, name, X, y, classes):
    xs = np.linspace(0, 1, 400)
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    tprs, aucs = [], []
    for i, (tr, te) in enumerate(cv.split(X, y), 1):
        model.fit(X.iloc[tr], y.iloc[tr])
        proba = model.predict_proba(X.iloc[te])
        y_bin = label_binarize(y.iloc[te], classes=classes)
        fpr, tpr, _ = roc_curve(y_bin.ravel(), proba.ravel())
        aucs.append(auc(fpr, tpr))
        tprs.append(np.interp(xs, fpr, tpr)); tprs[-1][0] = 0.0
        if i == 1:
            ax.plot(fpr, tpr, lw=0.8, alpha=0.25, color='grey', label='CV folds')
        else:
            ax.plot(fpr, tpr, lw=0.8, alpha=0.25, color='grey')
    mean_tpr = np.mean(tprs, axis=0); mean_tpr[-1] = 1.0
    std_tpr  = np.std(tprs,  axis=0)
    ax.plot(xs, mean_tpr, color='brown', lw=2, label=f"Mean AUC={np.mean(aucs):.3f}")
    ax.fill_between(xs, mean_tpr-std_tpr, mean_tpr+std_tpr, color='brown', alpha=0.25, label='±1 SD')
    ax.plot([1e-3, 1], [1e-3, 1], '--', lw=0.8, color='black', label='Chance')
    ax.set_xscale('log'); ax.set_xlim(1e-3, 1.0); ax.set_ylim(0, 1.02)
    ax.set_xlabel("False-Positive Rate (log)"); ax.set_ylabel("True-Positive Rate")
    ax.set_title(name); ax.legend(frameon=False, fontsize=7, loc='lower right')
    return float(np.mean(aucs)), float(np.std(aucs))

def main(args):
    warnings.filterwarnings('ignore')
    sns.set_theme(style='whitegrid', font='DejaVu Sans')
    plt.rcParams.update({'savefig.dpi': 600, 'axes.grid': False})

    rf   = load_csv(args.rf)
    svm  = load_csv(args.svm)
    cart = load_csv(args.cart)
    pts  = load_csv(args.points)

    X, y = prep_features(pts, label_col=args.label)
    classes = np.sort(pd.unique(y))
    if len(classes) < 2:
        raise ValueError('Need at least 2 classes in labels.')

    (brf, bsvm, bcart), (rf_model, svm_model, cart_model) = best_models(rf, svm, cart, X, y)

    fig, axs = plt.subplots(2, 3, figsize=(15, 9))
    plt.subplots_adjust(wspace=0.30, hspace=0.40)
    ax_rf, ax_svm, ax_cart = axs[0]
    ax_roc_rf, ax_roc_svm, ax_roc_cart = axs[1]

    rf_plot = rf.rename(columns={'numberOfTrees':'n_trees','variablesPerSplit':'mtry','accuracy':'accuracy'})
    rf_plot[['n_trees','mtry','accuracy']] = rf_plot[['n_trees','mtry','accuracy']].apply(pd.to_numeric, errors='coerce')
    rf_piv = rf_plot.pivot_table(index='mtry', columns='n_trees', values='accuracy', aggfunc='max')
    sns.heatmap(rf_piv, cmap='PuOr', ax=ax_rf, cbar_kws={'label':'OA'})
    ax_rf.set_xlabel('# Trees'); ax_rf.set_ylabel('mtry'); ax_rf.set_title('Random Forest')
    x_opt = rf_piv.columns.get_loc(brf['n_trees']) + .5
    y_opt = rf_piv.index.get_loc(brf['mtry']) + .5
    ax_rf.plot(x_opt, y_opt, marker='*', color='brown', ms=16)
    ax_rf.annotate(f"{brf['accuracy']:.3f}", xy=(x_opt, y_opt), xytext=(0, -12),
                   textcoords='offset points', ha='center', fontsize=8, color='black')

    svm = svm.rename(columns={'kernelType':'kernel','accuracy':'accuracy'})
    svm['kernel'] = svm['kernel'].astype(str).str.upper()
    kernels = ['LINEAR','POLY','RBF','SIGMOID']
    palette = sns.color_palette('colorblind', 4)
    styles  = ['solid','dashed','dashdot','dotted']
    ymin, ymax = svm['accuracy'].min() - 0.02, min(1.0, svm['accuracy'].max() + 0.02)
    for k, col, ls in zip(kernels, palette, styles):
        g = svm.loc[svm['kernel']==k].copy()
        if g.empty: continue
        g = g.sort_values('cost')
        ax_svm.plot(g['cost'], g['accuracy'], color=col, linestyle=ls, marker='o', lw=2.0, label=k)
        top = g.loc[g['accuracy'].idxmax()]
        ax_svm.plot(top['cost'], top['accuracy'], marker='*', color='brown', ms=14)
        ax_svm.annotate(f"{top['accuracy']:.3f}", xy=(top['cost'], top['accuracy']),
                        xytext=(0,-12), textcoords='offset points', ha='center', fontsize=8, color='black')
    ax_svm.set_xscale('log'); ax_svm.set_ylim(ymin, ymax)
    ax_svm.set_xlabel('Cost (C, log)'); ax_svm.set_ylabel('Overall Accuracy'); ax_svm.set_title('SVM')
    ax_svm.legend(title='Kernel', frameon=False, fontsize=7, loc='lower left')

    cart = cart.rename(columns={'maxNodes':'max_nodes','accuracy':'accuracy'})
    x_c = cart['max_nodes'].astype(float).values
    y_c = cart['accuracy'].astype(float).values
    x_c_jit = x_c + np.linspace(0, 1e-6, len(x_c))
    spl = si.UnivariateSpline(x_c_jit, y_c, k=min(3, max(2, len(x_c)-1)), s=0.0001)
    xs = np.linspace(x_c.min(), x_c.max(), 400)
    ax_cart.plot(xs, spl(xs), color=sns.color_palette('cividis', as_cmap=True)(0.55), lw=2)
    sizes = 250*(y_c - y_c.min())/(max(1e-9, y_c.max()-y_c.min())) + 30
    ax_cart.scatter(x_c, y_c, s=sizes, color=sns.color_palette('cividis', as_cmap=True)(0.55),
                    alpha=.6, edgecolor='k')
    ax_cart.plot(bcart['max_nodes'], bcart['accuracy'], marker='*', color='brown', ms=16)
    ax_cart.annotate(f"{bcart['accuracy']:.3f}", xy=(bcart['max_nodes'], bcart['accuracy']),
                     xytext=(0,8), textcoords='offset points', ha='center', fontsize=8, color='black')
    ax_cart.set_ylim(max(0.5, y_c.min()-0.02), min(1.0, y_c.max()+0.02))
    ax_cart.set_xlabel('Max nodes'); ax_cart.set_ylabel('Overall Accuracy'); ax_cart.set_title('CART')

    rf_auc_m, rf_auc_s   = draw_cv_mean_roc(ax_roc_rf,   rf_model,  'RF · 10-fold ROC',   X, y, classes)
    svm_auc_m, svm_auc_s = draw_cv_mean_roc(ax_roc_svm,  svm_model, 'SVM · 10-fold ROC',  X, y, classes)
    ct_auc_m, ct_auc_s   = draw_cv_mean_roc(ax_roc_cart, cart_model,'CART · 10-fold ROC', X, y, classes)

    plt.tight_layout()
    plt.savefig(args.out_figure, bbox_inches='tight')
    if args.show: plt.show()

    summary = pd.DataFrame([
        {'model':'RF','param_1':'n_trees','val_1':int(brf['n_trees']),'param_2':'mtry','val_2':int(brf['mtry']),
         'cv_auc_mean':rf_auc_m,'cv_auc_sd':rf_auc_s},
        {'model':'SVM','param_1':'kernel','val_1':str(bsvm['kernel']),'param_2':'C','val_2':float(bsvm['cost']),
         'cv_auc_mean':svm_auc_m,'cv_auc_sd':svm_auc_s},
        {'model':'CART','param_1':'max_nodes','val_1':int(bcart['max_nodes']),'param_2':'','val_2':'',
         'cv_auc_mean':ct_auc_m,'cv_auc_sd':ct_auc_s},
    ])
    summary.to_csv(args.out_summary, index=False)

if __name__ == '__main__':
    p = argparse.ArgumentParser(description='Hyper-parameter tuning plots + 10-fold ROC (RF/SVM/CART).')
    p.add_argument('--rf',   default='rf_numtrees_varspl.csv')
    p.add_argument('--svm',  default='svm_kernel_cost.csv')
    p.add_argument('--cart', default='cart_maxnodes.csv')
    p.add_argument('--points', default='sampled_normalized_parameters.csv')
    p.add_argument('--label', default='landcover')
    p.add_argument('--out-figure',  default='hyperparam_roc.png')
    p.add_argument('--out-summary', default='hyperparam_summary.csv')
    p.add_argument('--show', action='store_true')
    main(p.parse_args())


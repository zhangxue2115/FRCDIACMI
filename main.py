import os
import torch
import numpy as np
import pandas as pd

from frcdiacmi import FRCDIACMI
from utils import generate_data, data_to_tensor

from sklearn.model_selection import KFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    auc,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    accuracy_score,
    confusion_matrix
)

# =========================
# 超参数（训练层面）
# =========================
SEED = 42
N_SPLITS = 5
EPOCHS = 1000
LR = 1e-4
NEG_RATIO = 1            # generate_data 里用，正:负 = 1:NEG_RATIO
EMBEDDING_DIM = 512
FRC_HEADS = 4
DIA_HEADS = 4
DIA_REPEAT = 1
DIA_DROPOUT = 0.1



ADJ_PATH = "20208/20208_interaction_matrix.csv"
MIR_FEAT_PATH = "20208/miRNA_feature.csv"
CIRC_FEAT_PATH = "20208/circRNA_feature.csv"
# 输出文件
LOG_FILE = "frcdiacmi_20208_1to10_pair5fold_epoch_metrics.txt"
AVG_FILE = "frcdiacmi_20208_1to10_pair5fold_epoch_mean_metrics.txt"
CURVE_DIR = "frcdiacmi_curve_data_20208_1to10_pair5fold"
CURVE_SUMMARY_FILE = os.path.join(CURVE_DIR, "curve_summary.csv")


def build_curve_record(fold_id, epoch, test_true, test_probs, test_auc, test_aupr):
    fpr, tpr, roc_thresholds = roc_curve(test_true, test_probs)
    pr_precision, pr_recall, pr_thresholds = precision_recall_curve(test_true, test_probs)
    return {
        "fold_id": fold_id,
        "epoch": epoch,
        "auc": float(test_auc),
        "aupr": float(test_aupr),
        "fpr": fpr,
        "tpr": tpr,
        "roc_thresholds": roc_thresholds,
        "pr_precision": pr_precision,
        "pr_recall": pr_recall,
        "pr_thresholds": pr_thresholds,
    }


def save_curve_records(records, mode_name, curve_dir):
    os.makedirs(curve_dir, exist_ok=True)
    summary_rows = []

    mean_fpr = np.linspace(0, 1, 1001)
    mean_recall = np.linspace(0, 1, 1001)
    interp_tprs = []
    interp_precisions = []

    for record in records:
        fold_id = record["fold_id"]
        epoch = record["epoch"]
        prefix = f"{mode_name}_fold{fold_id}_epoch{epoch}"
        roc_file = os.path.join(curve_dir, f"{prefix}_roc_curve.csv")
        pr_file = os.path.join(curve_dir, f"{prefix}_pr_curve.csv")

        pd.DataFrame(
            {
                "fpr": record["fpr"],
                "tpr": record["tpr"],
                "threshold": record["roc_thresholds"],
            }
        ).to_csv(roc_file, index=False)

        pr_thresholds = np.append(record["pr_thresholds"], np.nan)
        pd.DataFrame(
            {
                "recall": record["pr_recall"],
                "precision": record["pr_precision"],
                "threshold": pr_thresholds,
            }
        ).to_csv(pr_file, index=False)

        interp_tpr = np.interp(mean_fpr, record["fpr"], record["tpr"])
        interp_tpr[0] = 0.0
        interp_tpr[-1] = 1.0
        interp_tprs.append(interp_tpr)

        pr_order = np.argsort(record["pr_recall"])
        recall_sorted = record["pr_recall"][pr_order]
        precision_sorted = record["pr_precision"][pr_order]
        interp_precision = np.interp(mean_recall, recall_sorted, precision_sorted)
        interp_precisions.append(interp_precision)

        summary_rows.append(
            {
                "mode": mode_name,
                "fold": fold_id,
                "epoch": epoch,
                "auc": record["auc"],
                "aupr": record["aupr"],
                "roc_file": roc_file,
                "pr_file": pr_file,
            }
        )

    interp_tprs = np.array(interp_tprs)
    interp_precisions = np.array(interp_precisions)

    mean_tpr = np.mean(interp_tprs, axis=0)
    mean_precision = np.mean(interp_precisions, axis=0)

    roc_mean_df = pd.DataFrame({"fpr": mean_fpr, "mean_tpr": mean_tpr})
    for idx, interp_tpr in enumerate(interp_tprs, start=1):
        roc_mean_df[f"fold{idx}_tpr"] = interp_tpr
    mean_roc_file = os.path.join(curve_dir, f"{mode_name}_mean_roc_curve.csv")
    roc_mean_df.to_csv(mean_roc_file, index=False)

    pr_mean_df = pd.DataFrame({"recall": mean_recall, "mean_precision": mean_precision})
    for idx, interp_precision in enumerate(interp_precisions, start=1):
        pr_mean_df[f"fold{idx}_precision"] = interp_precision
    mean_pr_file = os.path.join(curve_dir, f"{mode_name}_mean_pr_curve.csv")
    pr_mean_df.to_csv(mean_pr_file, index=False)

    mean_summary = {
        "mode": mode_name,
        "fold": "mean",
        "epoch": "",
        "auc": float(np.mean([record["auc"] for record in records])),
        "aupr": float(np.mean([record["aupr"] for record in records])),
        "roc_file": mean_roc_file,
        "pr_file": mean_pr_file,
        "auc_of_mean_roc": float(auc(mean_fpr, mean_tpr)),
        "aupr_of_mean_pr": float(auc(mean_recall, mean_precision)),
    }
    summary_rows.append(mean_summary)
    return summary_rows


def save_mean_best_epoch_curve_records_from_probs(fold_epoch_prob_files, target_epoch, mode_name, curve_dir):
    records = []
    for fold_id, prob_file in enumerate(fold_epoch_prob_files, start=1):
        with np.load(prob_file) as data:
            epoch_probs = data["epoch_probs"]
            if target_epoch >= epoch_probs.shape[0]:
                raise ValueError(
                    f"target_epoch={target_epoch} is out of range for {prob_file}; "
                    f"available epochs={epoch_probs.shape[0]}"
                )
            test_probs = epoch_probs[target_epoch]
            test_true = data["test_true"]

        test_auc = roc_auc_score(test_true, test_probs)
        test_aupr = average_precision_score(test_true, test_probs)
        records.append(
            build_curve_record(fold_id, target_epoch, test_true, test_probs, test_auc, test_aupr)
        )

    return save_curve_records(records, mode_name, curve_dir)

# =========================
# 固定随机性 & 设备
# =========================
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
os.makedirs(CURVE_DIR, exist_ok=True)

# =========================
# 读取数据
# =========================
adj = pd.read_csv(ADJ_PATH, header=None).values.tolist()
mir_feat = pd.read_csv(MIR_FEAT_PATH, header=None).values.tolist()
circ_feat = pd.read_csv(CIRC_FEAT_PATH, header=None).values.tolist()

# =========================
# 生成正负样本（pair-level）
# =========================
pos_data, neg_data = generate_data(adj, mir_feat, circ_feat, NEG_RATIO)
dataset = np.array(pos_data + neg_data)
labels = np.array([1] * len(pos_data) + [0] * len(neg_data))

print("pos:", len(pos_data), "| neg:", len(neg_data), "| total:", len(dataset))

# =========================
# 五折交叉验证（pair-level）
# =========================
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

# 写日志头（要和后面写入列严格一致）
with open(LOG_FILE, "w") as f:
    f.write("fold\tepoch\ttrain_loss\tauc\taupr\tprec\trecall\tf1\tmcc\tspe\tacc\n")

all_folds_auc = []
all_folds_aupr = []
all_folds_prec = []
all_folds_recall = []
all_folds_f1 = []
all_folds_mcc = []
all_folds_spe = []
all_folds_acc = []
best_auc_each_fold = []
curve_records_final_epoch = []
curve_records_best_auc = []
curve_records_best_aupr = []
fold_epoch_prob_files = []

for fold_id, (train_idx, test_idx) in enumerate(kf.split(dataset), start=1):
    print(f"\n======== Fold {fold_id} / {N_SPLITS} ========\n")

    train_data = dataset[train_idx]
    test_data = dataset[test_idx]
    train_labels = labels[train_idx]
    test_labels = labels[test_idx]

    train_mirna, train_circrna = data_to_tensor(train_data)
    test_mirna, test_circrna = data_to_tensor(test_data)

    train_mirna = train_mirna.to(device)
    train_circrna = train_circrna.to(device)
    test_mirna = test_mirna.to(device)
    test_circrna = test_circrna.to(device)

    train_labels = torch.tensor(train_labels, dtype=torch.float32, device=device)
    test_labels = torch.tensor(test_labels, dtype=torch.float32, device=device)
    fold_test_true_np = test_labels.detach().cpu().numpy().astype(np.int8)

    model = FRCDIACMI(
        mirna_feature_dim=train_mirna.shape[1],
        circrna_feature_dim=train_circrna.shape[1],
        embedding_dim=EMBEDDING_DIM,
        frc_heads=FRC_HEADS,
        dia_heads=DIA_HEADS,
        dia_dropout=DIA_DROPOUT,
        dia_repeat=DIA_REPEAT,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = torch.nn.BCELoss().to(device)

    fold_auc = []
    fold_aupr = []
    fold_prec = []
    fold_recall = []
    fold_f1 = []
    fold_mcc = []
    fold_spe = []
    fold_acc = []

    best_auc = float("-inf")
    best_aupr = float("-inf")
    best_auc_curve_record = None
    best_aupr_curve_record = None
    final_epoch_curve_record = None
    fold_epoch_probs = []

    for epoch in range(EPOCHS):
        # ---- train ----
        model.train()
        opt.zero_grad()

        out = model(train_mirna, train_circrna).squeeze(-1)  # (N,)
        loss = loss_fn(out, train_labels)
        loss.backward()
        opt.step()

        # ---- eval ----
        model.eval()
        with torch.no_grad():
            test_out = model(test_mirna, test_circrna).squeeze(-1)  # (M,)
            test_probs = test_out.detach().cpu().numpy()
            test_true = test_labels.detach().cpu().numpy()
            fold_epoch_probs.append(test_probs.astype(np.float32))

        test_pred = (test_probs >= 0.5).astype(int)

        # AUC / AUPR
        test_auc = roc_auc_score(test_true, test_probs)
        test_aupr = average_precision_score(test_true, test_probs)
        current_curve_record = None

        # 其它指标
        precision = precision_score(test_true, test_pred, zero_division=0)
        recall = recall_score(test_true, test_pred, zero_division=0)
        f1v = f1_score(test_true, test_pred, zero_division=0)
        mcc = matthews_corrcoef(test_true, test_pred)
        acc = accuracy_score(test_true, test_pred)

        tn, fp, fn, tp = confusion_matrix(test_true, test_pred, labels=[0, 1]).ravel()
        spe = tn / (tn + fp) if (tn + fp) > 0 else 0.0

        # 保存本 epoch
        fold_auc.append(test_auc)
        fold_aupr.append(test_aupr)
        fold_prec.append(precision)
        fold_recall.append(recall)
        fold_f1.append(f1v)
        fold_mcc.append(mcc)
        fold_spe.append(spe)
        fold_acc.append(acc)

        # 写日志
        with open(LOG_FILE, "a") as f:
            f.write(
                f"{fold_id}\t{epoch}\t{loss.item():.6f}\t"
                f"{test_auc:.6f}\t{test_aupr:.6f}\t"
                f"{precision:.6f}\t{recall:.6f}\t{f1v:.6f}\t{mcc:.6f}\t{spe:.6f}\t{acc:.6f}\n"
            )

        if test_auc > best_auc:
            best_auc = test_auc
            current_curve_record = build_curve_record(
                fold_id, epoch, test_true, test_probs, test_auc, test_aupr
            )
            best_auc_curve_record = current_curve_record

        if test_aupr > best_aupr:
            best_aupr = test_aupr
            if current_curve_record is None:
                current_curve_record = build_curve_record(
                    fold_id, epoch, test_true, test_probs, test_auc, test_aupr
                )
            best_aupr_curve_record = current_curve_record

        if epoch == EPOCHS - 1:
            if current_curve_record is None:
                current_curve_record = build_curve_record(
                    fold_id, epoch, test_true, test_probs, test_auc, test_aupr
                )
            final_epoch_curve_record = current_curve_record

    best_auc_each_fold.append(best_auc)
    fold_epoch_probs_file = os.path.join(CURVE_DIR, f"fold{fold_id}_epoch_probs_for_mean_best.npz")
    np.savez_compressed(
        fold_epoch_probs_file,
        epoch_probs=np.asarray(fold_epoch_probs, dtype=np.float32),
        test_true=fold_test_true_np,
    )
    fold_epoch_prob_files.append(fold_epoch_probs_file)

    curve_records_best_auc.append(best_auc_curve_record)
    curve_records_best_aupr.append(best_aupr_curve_record)
    curve_records_final_epoch.append(final_epoch_curve_record)

    all_folds_auc.append(fold_auc)
    all_folds_aupr.append(fold_aupr)
    all_folds_prec.append(fold_prec)
    all_folds_recall.append(fold_recall)
    all_folds_f1.append(fold_f1)
    all_folds_mcc.append(fold_mcc)
    all_folds_spe.append(fold_spe)
    all_folds_acc.append(fold_acc)

# =========================
# 5 折平均
# =========================
all_folds_auc = np.array(all_folds_auc)
all_folds_aupr = np.array(all_folds_aupr)
all_folds_prec = np.array(all_folds_prec)
all_folds_recall = np.array(all_folds_recall)
all_folds_f1 = np.array(all_folds_f1)
all_folds_mcc = np.array(all_folds_mcc)
all_folds_spe = np.array(all_folds_spe)
all_folds_acc = np.array(all_folds_acc)

mean_auc = np.mean(all_folds_auc, axis=0)
mean_aupr = np.mean(all_folds_aupr, axis=0)
mean_prec = np.mean(all_folds_prec, axis=0)
mean_recall = np.mean(all_folds_recall, axis=0)
mean_f1 = np.mean(all_folds_f1, axis=0)
mean_mcc = np.mean(all_folds_mcc, axis=0)
mean_spe = np.mean(all_folds_spe, axis=0)
mean_acc = np.mean(all_folds_acc, axis=0)


# ====== 计算 best epoch（前500 & 前1000）======

# 防止 EPOCHS < 500 的情况
top500 = min(500, EPOCHS)
top1000 = min(1000, EPOCHS)

# 前 500
best_auc_epoch_500 = int(np.argmax(mean_auc[:top500]))
best_auc_value_500 = float(mean_auc[best_auc_epoch_500])

best_aupr_epoch_500 = int(np.argmax(mean_aupr[:top500]))
best_aupr_value_500 = float(mean_aupr[best_aupr_epoch_500])

# 前 1000（即全程，或最多1000）
best_auc_epoch_1000 = int(np.argmax(mean_auc[:top1000]))
best_auc_value_1000 = float(mean_auc[best_auc_epoch_1000])

best_aupr_epoch_1000 = int(np.argmax(mean_aupr[:top1000]))
best_aupr_value_1000 = float(mean_aupr[best_aupr_epoch_1000])

# ====== 写入文件 ======
with open(AVG_FILE, "w") as f:
    f.write("epoch\tmean_auc\tmean_aupr\tmean_prec\tmean_recall\tmean_f1\tmean_mcc\tmean_spe\tmean_acc\n")
    for ep in range(EPOCHS):
        f.write(
            f"{ep}\t{mean_auc[ep]:.6f}\t{mean_aupr[ep]:.6f}\t"
            f"{mean_prec[ep]:.6f}\t{mean_recall[ep]:.6f}\t{mean_f1[ep]:.6f}\t"
            f"{mean_mcc[ep]:.6f}\t{mean_spe[ep]:.6f}\t{mean_acc[ep]:.6f}\n"
        )

    f.write("\n# ====== BEST (first 500 epochs) ======\n")
    f.write(f"best_auc_epoch_500\t{best_auc_epoch_500}\tbest_auc_500\t{best_auc_value_500:.6f}\n")
    f.write(f"best_aupr_epoch_500\t{best_aupr_epoch_500}\tbest_aupr_500\t{best_aupr_value_500:.6f}\n")

    f.write("\n# ====== BEST (first 1000 epochs) ======\n")
    f.write(f"best_auc_epoch_1000\t{best_auc_epoch_1000}\tbest_auc_1000\t{best_auc_value_1000:.6f}\n")
    f.write(f"best_aupr_epoch_1000\t{best_aupr_epoch_1000}\tbest_aupr_1000\t{best_aupr_value_1000:.6f}\n")

# ====== 保存绘制 ROC / PR 曲线所需数据 ======
curve_summary_rows = []
curve_summary_rows.extend(save_curve_records(curve_records_final_epoch, "final_epoch", CURVE_DIR))
curve_summary_rows.extend(save_curve_records(curve_records_best_auc, "best_auc_epoch", CURVE_DIR))
curve_summary_rows.extend(save_curve_records(curve_records_best_aupr, "best_aupr_epoch", CURVE_DIR))
curve_summary_rows.extend(
    save_mean_best_epoch_curve_records_from_probs(
        fold_epoch_prob_files,
        best_auc_epoch_1000,
        f"fivefold_mean_best_auc_epoch_{best_auc_epoch_1000}",
        CURVE_DIR,
    )
)
curve_summary_rows.extend(
    save_mean_best_epoch_curve_records_from_probs(
        fold_epoch_prob_files,
        best_aupr_epoch_1000,
        f"fivefold_mean_best_aupr_epoch_{best_aupr_epoch_1000}",
        CURVE_DIR,
    )
)
pd.DataFrame(curve_summary_rows).to_csv(CURVE_SUMMARY_FILE, index=False)

print("\nDone.")
print("LOG_FILE:", LOG_FILE)
print("AVG_FILE:", AVG_FILE)
print("CURVE_DIR:", CURVE_DIR)
print("CURVE_SUMMARY_FILE:", CURVE_SUMMARY_FILE)
print("best_auc_each_fold:", best_auc_each_fold)
print(f"BEST@500  AUC={best_auc_value_500:.6f} (epoch {best_auc_epoch_500}), "
      f"AUPR={best_aupr_value_500:.6f} (epoch {best_aupr_epoch_500})")
print(f"BEST@1000 AUC={best_auc_value_1000:.6f} (epoch {best_auc_epoch_1000}), "
      f"AUPR={best_aupr_value_1000:.6f} (epoch {best_aupr_epoch_1000})")

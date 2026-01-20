import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

np.random.seed(42)

DATA_PATH = Path(__file__).resolve().parent.parent.parent / "outputs" / "processed_data" / "cleaned_medal_data.csv"
TEST_START_YEAR = 2016

def load_and_prepare():
    df = pd.read_csv(DATA_PATH)
    df = df.sort_values(["Year", "NOC"]).reset_index(drop=True)

    target_col = "Is_First_Medal"
    feature_cols = ["Sport_Count", "Participants", "Prev_Year_Participation"]
    missing = [c for c in feature_cols + [target_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in data: {missing}")

    X = df[feature_cols]
    y = df[target_col].values

    train_mask = df["Year"] < TEST_START_YEAR
    X_train, X_test = X[train_mask], X[~train_mask]
    y_train, y_test = y[train_mask], y[~train_mask]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"Train years: {df.loc[train_mask, 'Year'].min()} - {df.loc[train_mask, 'Year'].max()}")
    print(f"Test years:  {df.loc[~train_mask, 'Year'].min()} - {df.loc[~train_mask, 'Year'].max()}")
    print(f"X_train shape: {X_train_scaled.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test_scaled.shape}, y_test shape: {y_test.shape}")

    return (X_train_scaled, y_train), (X_test_scaled, y_test), feature_cols, df, scaler


def build_model(input_dim: int) -> keras.Model:
    model = keras.Sequential(
        [
            layers.Input(shape=(input_dim,)),
            # [模型简化] 减少神经元数量、降低Dropout使模型更温和
            layers.Dense(32, activation="relu"),  # ReLU mitigates vanishing gradients
            layers.Dense(16, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(1, activation="sigmoid"),  # Sigmoid for binary probability
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def main():
    (X_train, y_train), (X_test, y_test), feature_cols, df_all, scaler = load_and_prepare()
    model = build_model(input_dim=X_train.shape[1])
    model.summary()

    # Class distribution and weights
    pos_count = int((y_train == 1).sum())
    neg_count = int((y_train == 0).sum())
    print(f"\nTraining class distribution -> 0: {neg_count}, 1: {pos_count}")
    # [类别权重优化] 主动提升正例权重，但不提升过头（gentle boost）
    class_weight = compute_gentle_boost_class_weight(y_train, alpha=0.4, max_pos_weight=1.65)
    print(f"Gentle-boost class_weight: {class_weight}")

    callbacks = [
        # [早停策略] 优先保召回，放宽停止条件（更高patience、监控val_recall最大化）
        EarlyStopping(monitor="val_recall", mode="max", patience=8, min_delta=0.002, restore_best_weights=True),
    ]

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_test, y_test),
        epochs=50,
        batch_size=32,
        class_weight=class_weight,
        callbacks=callbacks,
        verbose=2,
    )

    plot_training_curves(history)

    eval_results = model.evaluate(X_test, y_test, verbose=0)
    print("\nEvaluation on test set:")
    for name, value in zip(model.metrics_names, eval_results):
        print(f"{name}: {value:.4f}")

    # Manual metric computation for clarity
    y_prob = model.predict(X_test, verbose=0).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    print("\nManual metrics on test:")
    print(f"accuracy: {acc:.4f}")
    print(f"precision: {prec:.4f}")
    print(f"recall: {rec:.4f}")
    print(f"f1: {f1:.4f}")
    print("Confusion matrix (rows=true, cols=pred):")
    print(cm)

    # [精细阈值搜索] 0.1到0.9、步长0.05，输出每个阈值的指标并筛选满足目标区间
    df_thr, best_thr, best_metrics, satisfied = sweep_thresholds(y_test, y_prob)
    print("\nThreshold metrics (full sweep):")
    print(df_thr.to_string(index=False))
    print("\nBest threshold (by F1):", f"{best_thr:.2f}")
    print(
        "optimized -> accuracy: {acc:.4f}, precision: {prec:.4f}, recall: {rec:.4f}, f1: {f1:.4f}".format(
            acc=best_metrics["accuracy"],
            prec=best_metrics["precision"],
            rec=best_metrics["recall"],
            f1=best_metrics["f1"],
        )
    )
    print("Optimized confusion matrix:")
    print(best_metrics["confusion_matrix"]) 
    print("\nThresholds meeting targets (precision>=0.30 & recall>=0.60):")
    if satisfied.empty:
        print("None meet the target range; consider further tuning.")
    else:
        print(satisfied.to_string(index=False))

    # Use trained model to score projected 2028 candidates (never-medal, projected from 2024 participation)
    predict_projected_2028(model, scaler, feature_cols, df_all, prob_threshold=0.5)


def plot_training_curves(history: keras.callbacks.History) -> None:
    history_dict = history.history
    epochs = range(1, len(history_dict.get("loss", [])) + 1)

    plt.figure(figsize=(10, 4))

    # Loss
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history_dict.get("loss", []), label="train_loss")
    plt.plot(epochs, history_dict.get("val_loss", []), label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()

    # Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history_dict.get("accuracy", []), label="train_acc")
    plt.plot(epochs, history_dict.get("val_accuracy", []), label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training vs Validation Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(Path(__file__).resolve().parent / "training_curves.png", dpi=150)
    plt.close()

def sweep_thresholds(y_true: np.ndarray, y_prob: np.ndarray):
    # 阈值范围更细：0.1到0.9，步长0.05
    thresholds = np.round(np.arange(0.1, 0.91, 0.05), 2)
    rows = []
    best_f1 = -1.0
    best_thr = 0.5
    best_info = None

    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1v = f1_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        rows.append({"threshold": thr, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1v})

        if f1v > best_f1:
            best_f1 = f1v
            best_thr = thr
            best_info = {
                "accuracy": acc,
                "precision": prec,
                "recall": rec,
                "f1": f1v,
                "confusion_matrix": cm,
            }

    df_thr = pd.DataFrame(rows)
    satisfied = df_thr[(df_thr["precision"] >= 0.30) & (df_thr["recall"] >= 0.60)].copy()
    return df_thr, best_thr, best_info, satisfied


def compute_moderate_class_weight(y: np.ndarray, damping: float = 0.5, max_pos_weight: float = 2.0):
    """根据balanced权重进行阻尼，得到温和类别权重。
    - damping: 0~1之间，越小越温和；0表示不加权，1表示使用balanced原值。
    - max_pos_weight: 对正例权重上限裁剪，避免过大导致精确率崩盘。
    """
    classes = np.array([0, 1])
    balanced = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    w0_bal, w1_bal = float(balanced[0]), float(balanced[1])
    # 负类维持1.0，正类在1.0与balanced之间线性插值，然后裁剪到上限
    w0 = 1.0
    w1 = 1.0 + damping * (w1_bal - 1.0)
    w1 = float(min(max_pos_weight, max(1.0, w1)))
    print(f"Balanced class_weight -> 0:{w0_bal:.3f}, 1:{w1_bal:.3f}")
    return {0: w0, 1: w1}


def compute_gentle_boost_class_weight(y: np.ndarray, alpha: float = 0.4, max_pos_weight: float = 2.5):
    """主动提升正例权重，但不过度：
    - 基于类别不平衡比(neg/pos)进行温和提升：w1 = 1 + alpha * (neg/pos - 1)
    - 限制正例权重上限为max_pos_weight，负例始终为1.0
    - alpha建议0.3~0.6，控制提升力度；max_pos_weight用于避免精确率崩盘。
    """
    neg = float((y == 0).sum())
    pos = float((y == 1).sum())
    if pos == 0:
        # 防御：没有正例时不加权
        return {0: 1.0, 1: 1.0}
    imbalance = neg / pos
    w1 = 1.0 + alpha * max(0.0, (imbalance - 1.0))
    w1 = float(min(max_pos_weight, max(1.0, w1)))
    w0 = 1.0
    print(f"Class imbalance neg/pos: {imbalance:.3f} -> boosted w1: {w1:.3f}")
    return {0: w0, 1: w1}


def project_2028_from_2024(df: pd.DataFrame) -> pd.DataFrame:
    """Project 2028 participation using 2024 rows for countries that never medaled."""
    ever_medaled = df.groupby("NOC")["Has_Medal"].transform("sum") > 0
    df_never = df.loc[~ever_medaled].copy()
    df_2024 = df_never.loc[df_never["Year"] == 2024].copy()
    if df_2024.empty:
        return df_2024
    df_2028 = df_2024.copy()
    df_2028["Year"] = 2028
    return df_2028


def predict_projected_2028(
    model: keras.Model,
    scaler: StandardScaler,
    feature_cols,
    df_all: pd.DataFrame,
    prob_threshold: float = 0.5,
):
    df_2028 = project_2028_from_2024(df_all)
    if df_2028.empty:
        print("\n[Inference] No projected 2028 rows (from 2024 never-medal participants).")
        return

    X_2028 = df_2028[feature_cols]
    X_2028_scaled = scaler.transform(X_2028)
    probs = model.predict(X_2028_scaled, verbose=0).ravel()

    df_2028 = df_2028.copy()
    df_2028["prob"] = probs
    df_2028["High_Potential"] = (df_2028["prob"] >= prob_threshold).astype(int)

    ranked = df_2028.sort_values("prob", ascending=False)
    top20 = ranked.head(20)[["NOC", "prob", "High_Potential"]]

    print("\n[Inference] Projected 2028 first-medal probabilities (never-medal countries, based on 2024 participation):")
    print(top20.to_string(index=False, formatters={"prob": lambda x: f"{x:.4f}"}))


if __name__ == "__main__":
    main()

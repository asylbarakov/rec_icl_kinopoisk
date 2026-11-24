"""
Шаг 5: простые бейзлайны на ТЕХ ЖЕ окнах, что и у модели (valid_small, test, warm/cold, без нейтральных).

Признаки окна (все доступны на момент предсказания, утечки нет):
  * item_mean  — доля положительных откликов на фильм-цель среди целей train
                 (для самого train — leave-one-out, чтобы метка цели не попадала в свой же признак)
  * recent_yes — доля Yes среди 4 примеров окна (недавняя «позитивность» автора)
  * hist_len   — длина истории положительных фильмов
Бейзлайны: item_mean; recent_yes; логистическая регрессия на трёх признаках (обучается на train).

Запуск:  python 05_baselines.py --data_dir data
"""
import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


def load(d, name):
    return pd.read_pickle(f"{d}/{name}_ood2.pkl")


def auc(y, s):
    y = np.asarray(y)
    if len(np.unique(y)) < 2:
        return float("nan")
    return roc_auc_score(y, s)


def logit(p):
    p = np.clip(p, 1e-3, 1 - 1e-3)
    return np.log(p / (1 - p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    args = ap.parse_args()

    train = load(args.data_dir, "train")
    glob = train["label"].mean()
    s = train.groupby("iid")["label"].sum()
    c = train.groupby("iid")["label"].count()

    def feats(df, loo=False):
        ssum = df["iid"].map(s)
        cnt = df["iid"].map(c)
        if loo:
            item_mean = ((ssum - df["label"]) / (cnt - 1)).where(cnt > 1, glob)
        else:
            item_mean = (ssum / cnt).where(cnt > 0, glob)
        item_mean = item_mean.fillna(glob).to_numpy()
        recent = np.array([np.mean(l[:-1]) for l in df["labels"]])
        hist = np.log1p(np.array([len(h) - 1 for h in df["his"]]))
        return item_mean, recent, hist

    im_tr, rc_tr, hl_tr = feats(train, loo=True)
    X_tr = np.c_[logit(im_tr), rc_tr, hl_tr]
    lr = LogisticRegression(max_iter=1000).fit(X_tr, train["label"])
    print("Коэффициенты LR [logit(item_mean), recent_yes, log1p(hist_len)]:", np.round(lr.coef_[0], 3))

    test = load(args.data_dir, "test")
    subsets = {
        "valid_small": load(args.data_dir, "valid_small"),
        "test (все)": test,
        "test warm": test[test["not_cold"] == 1],
        "test cold": test[test["not_cold"] == 0],
        "test без нейтральных": load(args.data_dir, "test_noneutral"),
    }

    rows = []
    for name, df in subsets.items():
        if len(df) == 0:
            continue
        im, rc, hl = feats(df)
        y = df["label"].to_numpy()
        rows.append({
            "срез": name, "окон": len(df), "доля 1": round(float(y.mean()), 3),
            "item-mean": round(auc(y, im), 4),
            "recent-yes": round(auc(y, rc), 4),
            "LR(3 признака)": round(auc(y, lr.predict_proba(np.c_[logit(im), rc, hl])[:, 1]), 4),
        })
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()

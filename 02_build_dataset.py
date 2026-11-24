"""
Шаг 2: строим данные для RecICL из blinoff/kinopoisk (формат, который читает AmazonOOData в ICL-режиме).

Логика авторов (dataset/processing_ood.ipynb), адаптированная под Кинопоиск:
  * пользователь = author, айтем = movie_name, время = date (+ review_id как tie-break)
  * label: Good -> 1, Bad -> 0; Neutral не попадает ни в train, ни в истории, ни в few-shot примеры
  * Neutral из тестового окна выгружается в neutral_to_label.csv для ручной разметки и
    используется только как ЦЕЛЬ тестового окна
  * история сэмпла = только предыдущие положительные (label == 1) фильмы пользователя, с ведущим 0 / ''
  * глобальное разбиение по времени: train | valid | test (flag = -1 | 0 | 1) по дате ЦЕЛИ
  * ICL-окно = K последних сэмплов пользователя: K-1 примеров с ответами Yes/No + цель (последняя).
    Колонки окна: uids, iids, titles, hiss, his_titles, labels, model (+ his, not_cold, flag, ...)

Запуск (Colab / локально):
    pip install huggingface_hub pandas scikit-learn
    python 02_build_dataset.py                                   # 1-й проход: без нейтральных
    # ... размечаете data/neutral_to_label.csv (колонка label: 1 или 0) ...
    python 02_build_dataset.py --neutral_labels data/neutral_to_label.csv   # 2-й проход
"""
import argparse
import json
import os

import numpy as np
import pandas as pd


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=None, help="локальный kinopoisk.jsonl (по умолчанию скачивается с HF)")
    ap.add_argument("--out_dir", default="data")
    ap.add_argument("--k", type=int, default=5, help="размер ICL-окна (incontext_sample_len): K-1 примеров + цель")
    ap.add_argument("--min_user", type=int, default=5, help="мин. число размеченных (Good/Bad) рецензий у автора")
    ap.add_argument("--min_item", type=int, default=20, help="мин. число размеченных рецензий у фильма")
    ap.add_argument("--val_frac", type=float, default=0.10, help="доля размеченных рецензий во valid (по времени, с конца)")
    ap.add_argument("--test_frac", type=float, default=0.10, help="доля размеченных рецензий в test (по времени, с конца)")
    ap.add_argument("--neutral_labels", default=None, help="CSV с вашей разметкой нейтральных (колонки review_id, label)")
    return ap.parse_args()


def load_raw(path):
    if path is None:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id="blinoff/kinopoisk", filename="kinopoisk.jsonl", repo_type="dataset")
    return pd.read_json(path, lines=True)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    K = args.k

    df = load_raw(args.input)
    print("Исходно строк:", len(df))

    # ---- чистка ----
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["rid"] = pd.to_numeric(df["review_id"], errors="coerce")
    df = df.dropna(subset=["author", "date", "movie_name", "rid"]).copy()
    df["author"] = df["author"].astype(str)
    df = df[df["grade3"].isin(["Good", "Bad", "Neutral"])].copy()
    print("После удаления пустых author/date/movie/review_id:", len(df))

    df = df.sort_values(["date", "rid"]).reset_index(drop=True)
    n0 = len(df)
    df = df.drop_duplicates(subset=["author", "movie_name"], keep="first").reset_index(drop=True)
    print("Удалено повторных рецензий автора на тот же фильм:", n0 - len(df))

    # timestamp: сутки + порядковый номер внутри дня => порядок строго по (date, review_id)
    day_secs = (df["date"] - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)
    df["timestamp"] = day_secs + df.groupby("date").cumcount()

    df["neutral"] = df["grade3"].eq("Neutral")
    df["label"] = (df["grade3"] == "Good").astype(int)  # для Neutral значение не используется

    # ---- фильтр активных пользователей/фильмов (по размеченным рецензиям, как в оригинале) ----
    lab = df[~df["neutral"]]
    ucnt = lab.groupby("author").size()
    icnt = lab.groupby("movie_name").size()
    active_u = ucnt[ucnt >= args.min_user].index
    active_i = icnt[icnt >= args.min_item].index
    df = df[df["author"].isin(active_u) & df["movie_name"].isin(active_i)].copy()
    print(f"Активных авторов (>= {args.min_user}): {len(active_u)}, фильмов (>= {args.min_item}): {len(active_i)}")
    print("Строк после фильтра:", len(df), "| из них Neutral:", int(df["neutral"].sum()))

    # ---- глобальное разбиение по времени (по размеченным рецензиям) ----
    lab_dates = np.sort(df.loc[~df["neutral"], "date"].values)
    n = len(lab_dates)
    test_start = pd.Timestamp(lab_dates[int(n * (1 - args.test_frac))])
    valid_start = pd.Timestamp(lab_dates[int(n * (1 - args.test_frac - args.val_frac))])
    print("valid с:", valid_start.date(), "| test с:", test_start.date(), "| конец данных:", pd.Timestamp(lab_dates[-1]).date())

    df["flag"] = np.where(df["date"] >= test_start, 1, np.where(df["date"] >= valid_start, 0, -1))

    # сколько «обычных» (Good/Bad) рецензий автор написал раньше: цель окна требует >= K-1 таких
    reg = (~df["neutral"]).astype(int)
    df["reg_before"] = reg.groupby(df["author"]).cumsum() - reg

    # Neutral допустим только в тестовом окне (иначе утечка из будущего относительно train)
    neutral_all = df[df["neutral"]]
    in_test = neutral_all[neutral_all["flag"] == 1]
    neutral_test = in_test[in_test["reg_before"] >= K - 1].copy()  # только те, что станут целями окна
    print(f"Neutral в тестовом окне: {len(in_test)} из {len(neutral_all)}; "
          f"из них с полным окном (их и нужно разметить): {len(neutral_test)}")

    csv_path = os.path.join(args.out_dir, "neutral_to_label.csv")
    if args.neutral_labels is None:
        already = False
        if os.path.exists(csv_path):
            try:
                already = pd.read_csv(csv_path)["label"].isin([0, 1]).any()
            except Exception:
                already = False
        if already:
            print(f"ВНИМАНИЕ: {csv_path} уже содержит вашу разметку, файл НЕ перезаписан. "
                  f"Запускайте с --neutral_labels {csv_path}")
        else:
            out = neutral_test[["rid", "author", "movie_name", "part", "date", "title", "content"]].copy()
            out = out.rename(columns={"rid": "review_id", "title": "review_title"})
            out["review_id"] = out["review_id"].astype(int)
            out["content"] = out["content"].astype(str).str.slice(0, 2000)
            out["label"] = ""
            out.to_csv(csv_path, index=False, encoding="utf-8-sig")
            print("Записан файл для ручной разметки:", csv_path, "(label: 1 = ближе к положительной, 0 = к отрицательной)")
        manual = {}
    else:
        m = pd.read_csv(args.neutral_labels)
        m = m[m["label"].isin([0, 1])]
        manual = dict(zip(m["review_id"].astype(int), m["label"].astype(int)))
        print("Загружено меток для нейтральных:", len(manual))

    neutral_ok = df["neutral"] & df["rid"].astype(int).isin(manual.keys()) & (df["flag"] == 1)
    df.loc[neutral_ok, "label"] = df.loc[neutral_ok, "rid"].astype(int).map(manual)
    use = df[(~df["neutral"]) | neutral_ok].copy()

    # ---- ID как в оригинале: с 1, 0 зарезервирован под паддинг истории ----
    users = np.sort(use["author"].unique())
    items = np.sort(use["movie_name"].unique())
    users_map = dict(zip(users, np.arange(len(users)) + 1))
    items_map = dict(zip(items, np.arange(len(items)) + 1))
    use["uid"] = use["author"].map(users_map)
    use["iid"] = use["movie_name"].map(items_map)

    # ---- по каждому пользователю: сэмплы -> окна из K элементов ----
    rows, skipped = [], 0
    for uid, g in use.sort_values(["uid", "timestamp"]).groupby("uid", sort=True):
        his, his_title = [0], [""]
        regular = []  # прошлые «обычные» сэмплы (Good/Bad) этого пользователя
        for r in g.itertuples():
            rec = dict(uid=int(uid), iid=int(r.iid), title=r.movie_name, label=int(r.label),
                       his=list(his), his_title=list(his_title), flag=int(r.flag), timestamp=int(r.timestamp))
            if len(regular) >= K - 1:
                w = regular[-(K - 1):] + [rec]
                rows.append({
                    "uids": [x["uid"] for x in w],
                    "iids": [x["iid"] for x in w],
                    "titles": [x["title"] for x in w],
                    "hiss": [x["his"] for x in w],
                    "his_titles": [x["his_title"] for x in w],
                    "labels": [x["label"] for x in w],
                    "model": 0,                       # индекс hash-модели; без CF не используется
                    "his": rec["his"],                # нужен, чтобы датасет включил режим с историей
                    "uid": rec["uid"], "iid": rec["iid"], "label": rec["label"],
                    "timestamp": rec["timestamp"], "flag": rec["flag"],
                    "neutral": bool(r.neutral),       # цель — нейтральная рецензия с вашей ручной меткой
                })
            else:
                skipped += 1
            if not r.neutral:                         # нейтральные не становятся примерами и не попадают в историю
                regular.append(rec)
                if r.label == 1:
                    his.append(int(r.iid))
                    his_title.append(r.movie_name)

    data = pd.DataFrame(rows)
    print(f"Окон (K={K}): {len(data)} | сэмплов без полного окна (пропущены как цели): {skipped} | "
          f"авторов с окнами: {data['uid'].nunique()}")

    train_ = data[data["flag"] == -1].copy()
    valid_ = data[data["flag"] == 0].copy()
    test_ = data[data["flag"] == 1].copy()

    tu, ti = set(train_["uid"]), set(train_["iid"])
    for d in (valid_, test_):
        d["not_cold"] = [int(u in tu and i in ti) for u, i in zip(d["uid"], d["iid"])]
    train_["not_cold"] = 1

    for name, d in (("train", train_), ("valid", valid_), ("test", test_)):
        if len(d) == 0:
            print(f"{name:5s}: пусто!")
            continue
        hl = d["his"].apply(len) - 1
        ex_yes = np.mean([np.mean(l[:-1]) for l in d["labels"]])
        print(f"{name:5s}: {len(d):6d} окон | доля label=1 у цели: {d['label'].mean():.3f} | "
              f"доля Yes среди примеров: {ex_yes:.3f} | ср. длина истории цели: {hl.mean():.2f} | "
              f"not_cold: {d['not_cold'].mean():.3f}")

    # ---- бейзлайн «средняя оценка фильма»: нижняя планка, которую ICL обязан превзойти ----
    try:
        from sklearn.metrics import roc_auc_score
        item_mean = train_.groupby("iid")["label"].mean()
        glob = train_["label"].mean()
        for name, d in (("valid", valid_), ("test", test_)):
            if len(d) and d["label"].nunique() == 2:
                score = d["iid"].map(item_mean).fillna(glob)
                print(f"Бейзлайн item-mean AUC на {name}: {roc_auc_score(d['label'], score):.4f}")
    except ImportError:
        print("scikit-learn не установлен: бейзлайн пропущен")

    # ---- сохранение (имена как читает AmazonOOData: <split>_ood2.pkl) ----
    train_.to_pickle(os.path.join(args.out_dir, "train_ood2.pkl"))
    valid_.to_pickle(os.path.join(args.out_dir, "valid_ood2.pkl"))
    test_.to_pickle(os.path.join(args.out_dir, "test_ood2.pkl"))
    # тот же тест без нейтральных целей: ручные метки субъективны, поэтому отчитываемся по обоим вариантам
    test_[~test_["neutral"]].to_pickle(os.path.join(args.out_dir, "test_noneutral_ood2.pkl"))
    print(f"В тесте целей с ручной меткой: {int(test_['neutral'].sum())} из {len(test_)}")
    valid_.sample(frac=0.25, random_state=2023).to_pickle(os.path.join(args.out_dir, "valid_small_ood2.pkl"))
    with open(os.path.join(args.out_dir, "maps.json"), "w", encoding="utf-8") as f:
        json.dump({"users": {k: int(v) for k, v in users_map.items()},
                   "items": {k: int(v) for k, v in items_map.items()}}, f, ensure_ascii=False)
    print("Готово ->", args.out_dir)


if __name__ == "__main__":
    main()

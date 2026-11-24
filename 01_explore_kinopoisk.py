"""
Шаг 1: проверяем, годится ли blinoff/kinopoisk для RecICL.
RecICL нужны пользовательские истории: author -> последовательность (фильм, оценка, дата).

Запуск в Colab:
    !pip -q install huggingface_hub pandas
    !python 01_explore_kinopoisk.py
"""
import pandas as pd
from huggingface_hub import hf_hub_download

path = hf_hub_download(repo_id="blinoff/kinopoisk", filename="kinopoisk.jsonl", repo_type="dataset")
df = pd.read_json(path, lines=True)

print("Строк:", len(df))
print("Колонки:", list(df.columns))
print("\nПример даты:", df["date"].head(3).tolist())

# --- даты ---
df["dt"] = pd.to_datetime(df["date"], errors="coerce")
print("\nДат не распарсилось:", int(df["dt"].isna().sum()))
print("Диапазон дат:", df["dt"].min(), "->", df["dt"].max())

# --- оценки ---
print("\ngrade3:\n", df["grade3"].value_counts(dropna=False))
df["g10"] = pd.to_numeric(df["grade10"], errors="coerce")
print("\ngrade10: пропусков =", int(df["g10"].isna().sum()), "из", len(df))
print(df["g10"].value_counts(dropna=False).sort_index())

# --- плотность по авторам: главный вопрос ---
per_author = df.groupby("author").size()
print("\nУникальных авторов:", per_author.shape[0])
print("Рецензий на автора (describe):\n", per_author.describe())
for k in (2, 3, 5, 10, 20):
    authors_k = per_author[per_author >= k].index
    rows_k = int(df["author"].isin(authors_k).sum())
    print(f"Авторов с >= {k:>2} рецензий: {len(authors_k):>6}  (строк у них: {rows_k})")

# --- сколько фильмов ---
print("\nУникальных фильмов:", df["movie_name"].nunique())
print("Строк с пустым/служебным автором:", int(df["author"].isna().sum()))

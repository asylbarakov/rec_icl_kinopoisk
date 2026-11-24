"""
Шаг 3: сколько токенов Qwen занимают наши ICL-промпты.

Повторяет сборку промпта из кода авторов (AmazonOOData.__getitem__ + prompt_based_encode_v2):
  * история каждого из K элементов окна: последние <=10 названий, формат "t1", "t2", ...
  * <Response i> = Yes/No для первых K-1 элементов
  * шаблон берётся из prompts/RecICL_kinopoisk.txt (первая строка; "\\n" остаются как два символа, как у авторов)

Запуск:
    python 03_token_stats.py --data_dir data --prompt prompts/RecICL_kinopoisk.txt
"""
import argparse
import numpy as np
import pandas as pd

K = 5
MAX_HIS = 10  # min(max длина истории, 10) в AmazonOOData


def fmt_titles(ts):
    return ", ".join('"' + t.strip(" ") + '"' for t in ts)


def render(row, template):
    hiss, his_titles, labels, titles = row["hiss"], row["his_titles"], row["labels"], row["titles"]
    text = template
    for i in range(1, K + 1):
        h, ht = list(hiss[i - 1]), list(his_titles[i - 1])
        num = len(h) - (1 if h and h[0] == 0 else 0)
        if len(h) > MAX_HIS:
            num = MAX_HIS
        chosen = ht[-num:] if num > 0 else ht  # как в оригинале: [-0:] возвращает весь список
        text = text.replace(f"<ItemTitleList{i}>", fmt_titles(chosen))
        text = text.replace(f"<TargetItemTitle{i}>", '"' + titles[i - 1].strip(" ") + '"')
    for i in range(1, K):
        text = text.replace(f"<Response{i}>", "Yes" if labels[i - 1] == 1 else "No")
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--prompt", default="prompts/RecICL_kinopoisk.txt")
    ap.add_argument("--tokenizer", default="Qwen/Qwen1.5-0.5B", help="'whitespace' — грубая проверка без загрузки")
    ap.add_argument("--n", type=int, default=1500, help="сколько окон на сплит")
    ap.add_argument("--limit", type=int, default=2048, help="max_txt_len из конфига")
    args = ap.parse_args()

    template = open(args.prompt, encoding="utf-8").read().splitlines()[0]
    if args.tokenizer == "whitespace":
        tok_len = lambda s: len(s.split())
    else:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True, use_fast=False)
        tok_len = lambda s: len(tok(s).input_ids)

    shown = False
    for name in ("train", "valid_small", "test"):
        df = pd.read_pickle(f"{args.data_dir}/{name}_ood2.pkl")
        df = df.sample(n=min(args.n, len(df)), random_state=0)
        lens = []
        for _, r in df.iterrows():
            text = render(r, template)
            if not shown:
                print("Пример промпта:\n", text, "\n")
                shown = True
            lens.append(tok_len(text))
        lens = np.array(lens)
        print(f"{name:11s} n={len(lens):5d} | mean={lens.mean():7.0f} p50={np.percentile(lens, 50):6.0f} "
              f"p90={np.percentile(lens, 90):6.0f} p99={np.percentile(lens, 99):6.0f} max={lens.max():6d} | "
              f"> {args.limit}: {np.mean(lens > args.limit) * 100:.1f}%")


if __name__ == "__main__":
    main()

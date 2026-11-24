"""
Патч по памяти для rec_icl/minigpt4/models/minigpt4rec_v2.py (код авторов не копируем в репозиторий, патчим клон).

Проблема: self.qwen_model(..., labels=targets) считает логиты по всему словарю (~152k) для КАЖДОЙ позиции
(в float32) и ещё CE-loss, который дальше не используется: авторский loss и AUC берутся только из
outputs.logits[:, -t_posi, :][:, pos_ans_id], где t_posi = (число токенов ответа) + 1 = 2.

Патч: прогоняем только тело модели, а lm_head применяем к последним t_posi позициям.
Индекс [:, -t_posi, :] указывает на ту же абсолютную позицию, поэтому значения логитов те же;
меняется только расход памяти. Затрагиваются только блоки, где в пределах ~60 строк определён t_posi и
в ближайших ~40 строках используется outputs.logits[:,-t_posi,:] (то есть forward_v2, forward_v3 и
generate_for_samples_v2). Остальные вызовы не трогаются.

Использование:
    python patch_memory.py rec_icl/minigpt4/models/minigpt4rec_v2.py          # применить
    python patch_memory.py rec_icl/minigpt4/models/minigpt4rec_v2.py --revert # вернуть оригинал
"""
import argparse
import os
import re
import shutil
import sys

MARK = "_forward_last_logits"

METHOD = '''    def _forward_last_logits(self, inputs_embeds, attention_mask, keep):
        # PATCH(memory): lm_head только для последних `keep` позиций; loss по токенам авторами не используется.
        from types import SimpleNamespace
        if not self.use_lora:
            body, head = self.qwen_model.model, self.qwen_model.lm_head
        else:
            inner = self.qwen_model.base_model.model
            body, head = inner.model, inner.lm_head
        hidden = body(inputs_embeds=inputs_embeds, attention_mask=attention_mask, return_dict=True).last_hidden_state
        return SimpleNamespace(logits=head(hidden[:, -keep:, :]).float())

'''

CALL = re.compile(
    r"^(?P<ind>[ \t]*)outputs = self\.qwen_model\(\n"
    r"[ \t]*inputs_embeds=inputs_embeds,\n"
    r"[ \t]*attention_mask=attention_mask,\n"
    r"[ \t]*return_dict=True,\n"
    r"[ \t]*labels=targets,\n"
    r"[ \t]*\)",
    re.MULTILINE,
)


def apply(path):
    src = open(path, encoding="utf-8").read()
    if MARK in src:
        print("Патч уже применён, ничего не делаю.")
        return
    backup = path + ".orig"
    if not os.path.exists(backup):
        shutil.copyfile(path, backup)

    # 1) метод класса — перед def forward_v2
    anchor = "    def forward_v2(self, samples):"
    if src.count(anchor) != 1:
        sys.exit(f"Не нашёл единственный якорь '{anchor.strip()}', патч не применён.")
    src = src.replace(anchor, METHOD + anchor)

    # 2) подмена подходящих вызовов
    out, last, patched, skipped = [], 0, 0, 0
    for m in CALL.finditer(src):
        before = src[max(0, m.start() - 6000):m.start()].splitlines()[-60:]
        after = src[m.end():m.end() + 6000].splitlines()[:40]
        has_def = any(re.match(r"\s*t_posi = to_regress_tokens\.input_ids\.shape\[-1\] \+ 1", l) for l in before)
        has_use = any(re.match(r"\s*logits_? = outputs\.logits\[:,-t_posi,:\]", l) for l in after)
        out.append(src[last:m.start()])
        if has_def and has_use:
            out.append(f"{m.group('ind')}outputs = self._forward_last_logits(inputs_embeds, attention_mask, t_posi)")
            patched += 1
        else:
            out.append(m.group(0))
            skipped += 1
        last = m.end()
    out.append(src[last:])
    new = "".join(out)

    if patched == 0:
        shutil.copyfile(backup, path)
        sys.exit("Не нашёл ни одного подходящего вызова, файл оставлен без изменений.")
    open(path, "w", encoding="utf-8").write(new)
    print(f"Готово: заменено вызовов {patched}, пропущено {skipped}. Оригинал: {backup}")


def revert(path):
    backup = path + ".orig"
    if not os.path.exists(backup):
        sys.exit("Нет файла .orig, нечего возвращать.")
    shutil.copyfile(backup, path)
    print("Оригинал восстановлен.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--revert", action="store_true")
    a = ap.parse_args()
    revert(a.path) if a.revert else apply(a.path)

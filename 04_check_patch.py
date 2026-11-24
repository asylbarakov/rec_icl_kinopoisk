"""
Проверка патча по памяти: те же логиты и градиенты, что и при полном проходе?

Строит крошечную случайную Qwen2 (CPU, секунды) и сравнивает:
  * оригинал авторов: model(inputs_embeds, attention_mask).logits[:, -t_posi, :][:, pos_ans_id]
  * патч:             lm_head(model.model(...).last_hidden_state[:, -t_posi:, :])[:, -t_posi, :][:, pos_ans_id]
и градиенты BCE-loss по параметрам модели. t_posi = 2 (ответ Yes/No = один токен).

Запуск в Colab / WSL:  python 04_check_patch.py
Ожидаемо: разница порядка 1e-6 и надпись OK.
"""
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

torch.manual_seed(0)
cfg = Qwen2Config(
    vocab_size=1000, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
    num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=256,
    tie_word_embeddings=True,
)
model = Qwen2ForCausalLM(cfg).float().eval()

B, S, keep, pos_id = 2, 40, 2, 5
emb = torch.randn(B, S, cfg.hidden_size)
mask = torch.ones(B, S, dtype=torch.long)  # без левого паддинга: тело модели у обоих вариантов одно и то же
target = torch.tensor([1.0, 0.0])


def full_path():
    out = model(inputs_embeds=emb, attention_mask=mask, return_dict=True)
    return out.logits[:, -keep, :][:, pos_id]


def patched_path():
    hidden = model.model(inputs_embeds=emb, attention_mask=mask, return_dict=True).last_hidden_state
    logits = model.lm_head(hidden[:, -keep:, :]).float()
    return logits[:, -keep, :][:, pos_id]


bce = torch.nn.functional.binary_cross_entropy_with_logits
params = [p for n, p in model.named_parameters() if "layers.0.mlp.up_proj" in n or n == "model.norm.weight"]

ref, new = full_path(), patched_path()
g_ref = torch.autograd.grad(bce(full_path(), target), params)
g_new = torch.autograd.grad(bce(patched_path(), target), params)

d_logit = (ref - new).abs().max().item()
d_grad = max((a - b).abs().max().item() for a, b in zip(g_ref, g_new))
print(f"max |logit_full - logit_patched| = {d_logit:.3e}")
print(f"max |grad_full  - grad_patched | = {d_grad:.3e}")
print("OK" if d_logit < 1e-5 and d_grad < 1e-5 else "РАСХОЖДЕНИЕ: патч менять нельзя, пришлите вывод")

# -*- coding: utf-8 -*-
"""Смоук-тесты: structured output + фикс «оборванного на полуслове»."""
import json as _json
import sys
sys.path.insert(0, '/workspace')
from unittest import mock
from src.llm.llm_engine import LLMEngine

eng = LLMEngine.__new__(LLMEngine)
eng.base_url = "http://x"; eng.model = "m"; eng.timeout = 5
eng._ctx_cap = (0, 0)

md = "## T\n$\\frac{x^{3}}{3!}$ $\\mathbf{r}(t)$ $\\sqrt{x}$"
assert eng._extract_json_answer(_json.dumps({"answer": md})) == md
assert eng._extract_json_answer("```json\n" + _json.dumps({"answer": "abc"}) + "\n```") == "abc"
assert eng._extract_json_answer("plain markdown") is None
assert eng._extract_json_answer('{"other": 1}') is None
assert eng._extract_json_answer("") is None
print("T1 extract_json_answer OK")

class Resp:
    def __init__(self, content, dr="stop"):
        self._c = content; self.dr = dr
    def raise_for_status(self): pass
    def json(self):
        return {"message": {"content": self._c}, "done": True,
                "done_reason": self.dr, "metrics": {"eval_count": 10}}

calls = []
def fake_post(url, json=None, timeout=None):
    calls.append(json)
    return Resp(_json.dumps({"answer": "# H\n$\\to$ x"}))
with mock.patch.object(LLMEngine, "_ensure_available"), \
     mock.patch.object(LLMEngine, "_resolve_model", return_value="m"), \
     mock.patch.object(LLMEngine, "detect_model_window", return_value=16384), \
     mock.patch("src.llm.llm_engine.requests.post", fake_post):
    out = eng.chat([{"role": "user", "content": "q"}])
assert calls[0]["format"]["type"] == "json_schema", calls[0].get("format")
assert out == "# H\n$\\to$ x", repr(out)
print("T2 structured-output payload + unwrap OK")

calls.clear()
it = iter([Resp(""), Resp("plain markdown")])
def fake_post2(url, json=None, timeout=None):
    calls.append(dict(json)); return next(it)
with mock.patch.object(LLMEngine, "_ensure_available"), \
     mock.patch.object(LLMEngine, "_resolve_model", return_value="m"), \
     mock.patch.object(LLMEngine, "detect_model_window", return_value=16384), \
     mock.patch("src.llm.llm_engine.requests.post", fake_post2):
    out = eng.chat([{"role": "user", "content": "q"}])
assert "format" in calls[0] and "format" not in calls[1], [list(c) for c in calls]
assert out == "plain markdown"
print("T3 schema-fallback OK")

big_ctx = "Контекст главы. " * 4000
part1 = "А" * 9000
calls.clear(); it2 = iter([Resp(part1, dr="length"), Resp("", dr="length"), Resp("OK-END", dr="stop")])
def fake_post3(url, json=None, timeout=None):
    r = next(it2); calls.append((dict(json), r)); return r
with mock.patch.object(LLMEngine, "_ensure_available"), \
     mock.patch.object(LLMEngine, "_resolve_model", return_value="m"), \
     mock.patch.object(LLMEngine, "detect_model_window", return_value=16384), \
     mock.patch("src.llm.llm_engine.requests.post", fake_post3):
    out = eng.chat([{"role": "system", "content": big_ctx},
                    {"role": "user", "content": "вопрос"}])
# ожидалось: 1) основной ответ (length) -> 2) continuation-дозапроска.
# Если вторая вернула пустоту — должен был сработать одноразовый ретрай
# с уменьшенным окном (3-й вызов).
assert len(calls) >= 2
cont_call = None
for i in range(1, len(calls)):
    roles = [m["role"] for m in calls[i][0]["messages"]]
    if "assistant" in roles:
        cont_call = calls[i][0]; break
assert cont_call is not None, f"нет continuation-вызова среди {len(calls)}"
payload2 = cont_call
msgs2 = payload2["messages"]
asst = [m for m in msgs2 if m["role"] == "assistant"][-1]
last_user = msgs2[-1]
assert asst["content"] == part1[-1500:], "tail передан неверно"
assert "Твой ответ был прерван" in last_user["content"]
assert "Продолжи ровно с точки обрыва" in last_user["content"]
assert "обрезан" not in asst["content"], "хвост ответа был обрезан!"
assert payload2["options"]["num_ctx"] <= 16384
assert out.startswith(part1) and out.endswith("OK-END"), "склейка сломана"
print("T4 continuation tail-protected + window OK")

tr = eng._trim_history([{"role": "system", "content": big_ctx},
                        {"role": "user", "content": "малый запрос"}], 16384)
assert tr[-1]["content"] == "малый запрос"
print("T5 default protect_last OK")

assert eng._glue_parts("xxxABCDEFtail", "ABCDEFtail cont") == " cont"
print("T6 glue regression OK")
print("ALL TESTS PASSED")

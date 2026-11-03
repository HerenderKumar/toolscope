import numpy as np

from toolscope.diagnosis.rewrite import card_prompt, rewrite_messages
from toolscope.diagnosis.whatif import WhatIf, rank_of


def _card(tid, name, desc):
    return {"tool_id": tid, "server_name": "S", "server_description": "", "tool_name": name, "description": desc,
            "params": [], "search_text": f"S | {name} | {desc}"}


CARDS = [_card("s/a", "alpha", "weather forecast for a city"), _card("s/b", "beta", "获取菜谱"),
         _card("s/c", "gamma", "stock prices and quotes")]


def _fake_encode(texts):  # bag-of-words over a tiny vocabulary, L2-normalised
    vocab = ["weather", "recipe", "stock", "cook"]
    v = np.array([[t.lower().count(w) for w in vocab] for t in texts], dtype=np.float32) + 1e-3
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def test_rank_of_is_stable():
    assert rank_of(np.array([1.0, 3.0, 3.0]), 2) == 2


def test_rewrite_moves_tool_up_and_leaves_inputs_untouched():
    vectors = _fake_encode([c["search_text"] for c in CARDS])
    before = vectors.copy()
    editor = WhatIf(CARDS, vectors, encode_doc=_fake_encode, encode_query=_fake_encode)
    res = editor.whatif("s/b", "Find a recipe and cook instructions for a dish", ["recipe to cook dinner"])[0]
    assert res["bm25"][1] == 1 and res["bm25"][0] > 1
    assert res["semantic"][1] == 1
    assert np.array_equal(vectors, before) and CARDS[1]["description"] == "获取菜谱"


def test_rewrite_prompt_holds_only_the_card():
    card = dict(CARDS[1], params=[{"name": "dish", "type": "string", "description": "", "required": True}])
    msgs = rewrite_messages(card)
    assert msgs[1]["content"] == card_prompt(card)
    assert "获取菜谱" in msgs[1]["content"] and "dish (string, required)" in msgs[1]["content"]
    assert "task" not in msgs[1]["content"].lower() and "query" not in msgs[1]["content"].lower()

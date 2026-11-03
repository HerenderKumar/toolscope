import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
from make_sample_sheet import stratified_tasks  # noqa: E402


def test_stratified_sample_is_distinct_and_proportional():
    tasks = [{"task_id": f"a{i}", "category": "A"} for i in range(60)] + \
            [{"task_id": f"b{i}", "category": "B"} for i in range(30)]
    chosen = stratified_tasks(tasks, 30, random.Random(1))
    assert len({t["task_id"] for t in chosen}) == 30
    assert sum(t["category"] == "A" for t in chosen) == 20

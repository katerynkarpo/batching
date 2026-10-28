"""Prompt templates. One instruction per task, shared batch framing.

Batch framing (identical for b=1 and b>1 so that the only difference is the number of items):

    <instruction>
    <format rule>

    [1] <input 1>
    [2] <input 2>
    ...

Expected completion:

    [1] <answer 1>
    [2] <answer 2>

Few-shot demonstrations are given as one previous user/assistant exchange with `few_shot` items,
so the demo batch size is a constant and does not vary with b.
"""
from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class TaskSpec:
    instruction: str                 # what to do with each item
    answer_format: str               # how each answer line must look
    max_tokens_per_item: int         # generation budget per item
    render_input: str = "plain"      # plain | absa_atsc | wikitq
    reasoning_instruction: str = ""  # appended when reasoning=True


SPECS: dict[str, TaskSpec] = {
    "gec": TaskSpec(
        instruction=(
            "You are a grammatical error correction system. Correct the grammatical, spelling, "
            "punctuation and word-choice errors in each sentence using the minimal number of edits. "
            "Do not rewrite or paraphrase, and do not change the meaning or style. "
            "If a sentence has no errors, output it unchanged. Keep the original tokenisation "
            "(spaces around punctuation) exactly as in the input."
        ),
        answer_format="the corrected sentence",
        max_tokens_per_item=120,
    ),
    "gsm8k": TaskSpec(
        instruction="Solve each math word problem.",
        answer_format="a brief step-by-step solution ending with the line '#### <final number>'",
        max_tokens_per_item=350,
    ),
    "sst5": TaskSpec(
        instruction=(
            "Classify the sentiment of each movie review sentence into exactly one of: "
            "very negative, negative, neutral, positive, very positive."
        ),
        answer_format="one of the five labels, nothing else",
        max_tokens_per_item=8,
    ),
    "tweeteval": TaskSpec(
        instruction="Classify the sentiment of each tweet into exactly one of: negative, neutral, positive.",
        answer_format="one of the three labels, nothing else",
        max_tokens_per_item=8,
    ),
    "absa_atsc": TaskSpec(
        instruction=(
            "For each item you are given a review sentence and an aspect term from it. "
            "Classify the sentiment expressed towards that aspect as positive, negative or neutral."
        ),
        answer_format="one of the three labels, nothing else",
        max_tokens_per_item=8,
        render_input="absa_atsc",
    ),
    "absa_e2e": TaskSpec(
        instruction=(
            "Extract all aspect terms from each review sentence together with the sentiment "
            "expressed towards each (positive, negative, neutral or conflict). Aspect terms must be "
            "copied verbatim from the sentence. If the sentence contains no aspect terms, output []."
        ),
        answer_format='a JSON list like [{"term": "...", "polarity": "..."}] on one line',
        max_tokens_per_item=96,
    ),
    "strategyqa": TaskSpec(
        instruction="Answer each yes/no question. The questions require implicit multi-step reasoning.",
        answer_format="'yes' or 'no', nothing else",
        max_tokens_per_item=8,
        reasoning_instruction="Think step by step briefly, then end with the line 'Answer: yes' or 'Answer: no'.",
    ),
    "wikitq": TaskSpec(
        instruction="Answer each question using the given table.",
        answer_format="the answer only; if there are several answers, separate them with ' | '",
        max_tokens_per_item=32,
        render_input="wikitq",
    ),
}

FORMAT_RULE = (
    "There are {n} item(s), numbered [1] to [{n}]. Answer every item, in order, each on its own "
    "line starting with its number in square brackets, e.g. '[1] ...'. For each item output {fmt}. "
    "Do not add anything else."
)


def render_table(table: dict, max_rows: int | None = None) -> str:
    header = table["header"]
    rows = table["rows"] if max_rows is None else table["rows"][:max_rows]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(c.replace("\n", " ") for c in r) + " |" for r in rows]
    return "\n".join(lines)


def render_input(task: str, rec: dict) -> str:
    kind = SPECS[task].render_input
    if kind == "absa_atsc":
        return f"Sentence: {rec['input']}\nAspect: {rec['meta']['aspect']}"
    if kind == "wikitq":
        return f"Question: {rec['input']}\nTable:\n{render_table(rec['meta']['table'])}"
    return rec["input"]


def render_demo_answer(task: str, rec: dict, reasoning: bool) -> str:
    if task == "gsm8k":
        return f"{rec['meta']['solution']}\n#### {rec['target']}"
    if task == "absa_e2e":
        return json.dumps(rec["target"], ensure_ascii=False)
    if task == "wikitq":
        return " | ".join(rec["target"])
    if task == "strategyqa" and reasoning:
        return f"{rec['meta']['facts']}\nAnswer: {rec['target']}"
    return rec["target"]


def items_block(task: str, recs: list[dict]) -> str:
    return "\n".join(f"[{i + 1}] {render_input(task, r)}" for i, r in enumerate(recs))


def answers_block(task: str, recs: list[dict], reasoning: bool) -> str:
    return "\n".join(f"[{i + 1}] {render_demo_answer(task, r, reasoning)}" for i, r in enumerate(recs))


def build_messages(task: str, batch: list[dict], demos: list[dict] | None = None,
                   reasoning: bool = False) -> list[dict]:
    """Chat messages for one batch. `batch` and `demos` are unified records."""
    spec = SPECS[task]
    instr = spec.instruction
    if reasoning and spec.reasoning_instruction:
        instr += " " + spec.reasoning_instruction
    system = instr

    def user_turn(recs: list[dict]) -> str:
        rule = FORMAT_RULE.format(n=len(recs), fmt=spec.answer_format)
        return f"{rule}\n\n{items_block(task, recs)}"

    messages = [{"role": "system", "content": system}]
    if demos:
        messages.append({"role": "user", "content": user_turn(demos)})
        messages.append({"role": "assistant", "content": answers_block(task, demos, reasoning)})
    messages.append({"role": "user", "content": user_turn(batch)})
    return messages


def max_new_tokens(task: str, batch_size: int, reasoning: bool = False) -> int:
    per_item = SPECS[task].max_tokens_per_item
    if reasoning and task == "strategyqa":
        per_item = 160
    return per_item * batch_size + 32

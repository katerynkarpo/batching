"""GEC prompts and few-shot demonstrations ported verbatim from the GEC paper repository
(github.com/katerynkarpo/llm-en-gec, commit 81fefec). Generated, do not edit by hand:
the texts must stay byte-identical to the paper prompts.

  base           A.1  prompt_01_vanilla_zero_shot           (EN_BASE_GEC_PROMPT)
  minimal_edits  A.2  prompt_02_minimal_edits_zero_shot     (MINIMAL_EDITS_PROMPT)
  taxonomy       A.4  prompt_03_minimal_edits_taxonomy_...  (TAXONOMY_BASED_PROMPT)

Few-shot (A.3 / A.5 setup): 8 BEA-2019 train pairs sampled with seed 123, detokenised, appended
to the system prompt. The demonstrations are part of the system prompt, so they are identical
for every batch size.
"""
from __future__ import annotations

GEC_PROMPTS: dict[str, str] = {
    'base': (
        'Reply with a corrected version of the input sentence with all grammatical and spelling errors fixed. If there are no errors, reply with a copy of the original sentence.\n'
    ),
    'minimal_edits': (
        'You are a grammatical error correction system. Make MINIMAL, PRECISE edits to fix errors. DO NOT rewrite or paraphrase. Only fix clear grammatical and spelling errors.\n'
        '\n'
        'RULES:\n'
        '- Make the SMALLEST possible edit to fix each error\n'
        '- Change only what is grammatically or orthographically wrong\n'
        '- Preserve the original meaning and style\n'
        '- Do NOT improve fluency beyond fixing errors\n'
        '- Do NOT make stylistic changes\n'
        '- If no errors exist, return the original sentence unchanged'
    ),
    'taxonomy': (
        'You are a grammatical error correction system. Make MINIMAL, PRECISE edits to fix errors. DO NOT rewrite or paraphrase. Only fix clear grammatical and spelling errors.\n'
        '\n'
        'Focus on these 25 error types:\n'
        '\n'
        'WORD-LEVEL ERRORS:\n'
        '1. ADJ: Wrong adjective choice (big→wide)\n'
        '2. ADJ:FORM: Adjective form errors - comparatives/superlatives (goodest→best, more easy→easier)\n'
        '3. ADV: Wrong adverb choice (speedily→quickly)\n'
        '4. CONJ: Wrong conjunction (and→but)\n'
        "5. CONTR: Contraction errors (n't→not)\n"
        '6. DET: Wrong/missing/extra determiner (the→a, ∅→the, the→∅)\n'
        '7. NOUN: Wrong noun choice (person→people)\n'
        '8. NOUN:INFL: Count-mass noun errors (informations→information)\n'
        '9. NOUN:NUM: Noun number agreement (cat→cats)\n'
        "10. NOUN:POSS: Noun possessive errors (friends→friend's)\n"
        '11. PART: Wrong particle (look in→look at)\n'
        '12. PREP: Wrong/missing/extra preposition (of→at, ∅→at, at→∅)\n'
        '13. PRON: Wrong pronoun (ours→ourselves)\n'
        '14. VERB: Wrong verb choice (ambulate→walk)\n'
        '15. VERB:FORM: Verb form errors - infinitive/gerund/participle (to eat→eating, dancing→danced)\n'
        '16. VERB:INFL: Verb inflection errors (getted→got, fliped→flipped)\n'
        '17. VERB:SVA: Subject-verb agreement ((He) have→(He) has)\n'
        '18. VERB:TENSE: Verb tense errors including modals and passive (eats→ate, eats→can eat, eats→was eaten)\n'
        '\n'
        'MECHANICAL ERRORS:\n'
        '19. ORTH: Orthography - capitalization/whitespace (Bestfriend→best friend, THIS→this)\n'
        '20. PUNCT: Punctuation errors (!→., missing commas, extra periods)\n'
        '21. SPELL: Spelling errors (genectic→genetic, color→colour)\n'
        '22. WO: Word order errors (only can→can only)\n'
        '\n'
        'OTHER:\n'
        '23. MORPH: Morphology - same lemma, different part of speech (quick[adj]→quickly[adv])\n'
        '24. OTHER: Complex errors requiring minimal paraphrasing\n'
        '25. UNK: Leave unchanged if error is unclear\n'
        '\n'
        'RULES:\n'
        '- Make the SMALLEST possible edit to fix each error\n'
        '- Change only what is grammatically or orthographically wrong\n'
        '- Preserve the original meaning and style\n'
        '- Do NOT improve fluency beyond fixing errors\n'
        '- Do NOT make stylistic changes\n'
        '- If no errors exist, return the original sentence unchanged'
    ),
}

# (train_index is the 1-based line in bea-train.src of the paper repository)
FEW_SHOT_SEED123: list[dict] = [
    {"train_index": 3432, "source": "Alison put on her coat, close the door and went to his parents house by car.", "reference": "Alison put on her coat, closed the door and went to her parents ' house by car."},
    {"train_index": 17543, "source": "we can see the number of the projected sales of jeans of Jack&Jones Co. will be at 450 thousands of pairs in next Jan, it will be higher than the number of the projected sales of jeans of Mango Co. about 300 thousands of pairs.Then the number of Jack&Jones Co. will drop to 250 thousands of pairs in Feb next year.", "reference": "we can see the number of projected sales of jeans of Jack&Jones Co. will be at 450 thousand pairs next Jan, it will be higher than the number of the projected sales of jeans of Mango Co.; about 300 thousand pairs. Then the number of Jack&Jones Co. will drop to 250 thousand pairs in Feb next year."},
    {"train_index": 5714, "source": "The girl is 18 years old and men (as it later turns out) are vampires and they have over 100 years.", "reference": "The girl is 18 years old and the men (as it later turns out) are vampires and they are over 100 years old."},
    {"train_index": 26689, "source": "Like what was once, I have no doubt that this world of wonders will remain important in the coming years.", "reference": "Like what was once, I have no doubt that this world of wonders will remain important in the coming years."},
    {"train_index": 17469, "source": "The need of export and import items has increased in the last 10 years consequently the life of the people has raised in faster respond.", "reference": "The need for exporting and importing items has increased in the last 10 years. Consequently, the life of the people has been raised in rapid response."},
    {"train_index": 7059, "source": "In conclusion, it's very important to learn the concept than saying something stupid....", "reference": "In conclusion, it's more important to learn the concept than say saying something stupid...."},
    {"train_index": 2501, "source": "Abortion is an immoral act that people make as everyone has a chance to survive and nobody has the right to decide for anyone life.", "reference": "Abortion is an immoral act that people perform as everyone has the chance to survive and nobody has the right to decide about anyone's life."},
    {"train_index": 24847, "source": "Hi John:", "reference": "Hi John:"},
]


def few_shot_block(n: int) -> str:
    """Same text as `format_few_shot_prompt` in the paper repository, for the first n examples."""
    if n > len(FEW_SHOT_SEED123):
        raise ValueError(f"only {len(FEW_SHOT_SEED123)} paper few-shot examples are available, got n={n}")
    lines = [
        "Few-shot examples from BEA train:",
        "Follow the same input-to-correction style. Do not copy these examples; use them only as guidance.",
    ]
    for i, e in enumerate(FEW_SHOT_SEED123[:n], start=1):
        lines += ["", f"Example {i}:", f"Input: {e['source']}", f"Correction: {e['reference']}"]
    return "\n".join(lines)


def system_prompt(variant: str, few_shot: int = 0) -> str:
    """Paper system prompt; with few_shot > 0 the demonstrations are appended (delivery: system)."""
    if variant not in GEC_PROMPTS:
        raise ValueError(f"unknown GEC prompt {variant!r}; available: {sorted(GEC_PROMPTS)}")
    prompt = GEC_PROMPTS[variant]
    if few_shot > 0:
        return f"{prompt.rstrip()}\n\n{few_shot_block(few_shot).strip()}"
    return prompt

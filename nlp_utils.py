"""NLP engine for AI Study Buddy: summary, keywords, MCQs, flashcards (spaCy)."""
import random
import re
from collections import Counter

import spacy

MODEL = "en_core_web_sm"

# Generic, unrelated words: used as wrong options when the text has too few keywords.
FALLBACK_POOL = [
    "HTML", "Database", "Networking", "Compiler", "Spreadsheet", "Operating system",
    "Photosynthesis", "Gravity", "Algebra", "Democracy", "Hardware", "Browser",
    "Cell", "Energy", "Molecule", "Revolution", "Algorithm", "Protocol",
]

DEFINITION_VERBS = (" is ", " are ", " means ", " refers to ", " allows ", " uses ", " finds ")


def load_nlp():
    try:
        return spacy.load(MODEL)
    except OSError:  # model missing -> download once
        from spacy.cli import download
        download(MODEL)
        return spacy.load(MODEL)


def extract_keywords(doc, top_n=30):
    """Rank nouns / proper nouns / noun phrases by frequency (ties: first seen)."""
    counts, first_seen = Counter(), {}
    for chunk in doc.noun_chunks:
        toks = [t for t in chunk if not t.is_stop and not t.is_punct and t.pos_ in ("NOUN", "PROPN", "ADJ")]
        if not toks or toks[-1].pos_ not in ("NOUN", "PROPN"):
            continue
        phrase = " ".join(t.text for t in toks)
        if len(phrase) < 2:
            continue
        key = phrase.lower()
        counts[key] += 1
        first_seen.setdefault(key, (chunk.start, phrase))
    ranked = sorted(counts, key=lambda k: (-counts[k], first_seen[k][0]))
    return [first_seen[k][1] for k in ranked[:top_n]]


def summarize(doc, n_sentences=2):
    """Extractive summary: score sentences by keyword-word frequency."""
    sents = list(doc.sents)
    if len(sents) <= n_sentences:
        return " ".join(s.text.strip() for s in sents)
    freq = Counter(t.lemma_.lower() for t in doc if t.pos_ in ("NOUN", "PROPN") and not t.is_stop)
    scores = {}
    for i, s in enumerate(sents):
        words = [t for t in s if t.is_alpha]
        scores[i] = sum(freq[t.lemma_.lower()] for t in words) / (len(words) ** 0.6 + 1)
    best = sorted(sorted(scores, key=scores.get, reverse=True)[:n_sentences])
    return " ".join(sents[i].text.strip() for i in best)


def make_flashcards(doc, keywords, limit=15):
    """Flashcards: keyword on the front, the sentence that explains it on the back."""
    sents = [s.text.strip() for s in doc.sents if len(s.text.split()) >= 4]
    cards = []
    for k in keywords:
        pat = re.compile(rf"\b{re.escape(k)}\b", re.I)
        matches = [s for s in sents if pat.search(s)]
        if not matches:
            continue
        # prefer a sentence that looks like a definition ("X is ...", "X allows ...")
        matches.sort(key=lambda s: (not any(v in s for v in DEFINITION_VERBS), pat.search(s).start() > 40))
        cards.append({"term": k, "definition": matches[0]})
        if len(cards) >= limit:
            break
    return cards


def pick_distractors(answer, pool, fallback, difficulty, rng):
    """Choose 3 wrong options. Harder levels use more believable ones.

    Easy   -> unrelated words (clearly wrong)
    Medium -> mix of keywords from the text and unrelated words
    Hard   -> only similar keywords from the same text (same length / word count first)
    """
    pool, fallback = pool[:], fallback[:]
    rng.shuffle(pool); rng.shuffle(fallback)
    if difficulty == "Easy":
        cands = fallback + pool
    elif difficulty == "Hard":
        pool.sort(key=lambda k: (abs(len(k.split()) - len(answer.split())), abs(len(k) - len(answer))))
        cands = pool + fallback
    else:
        mixed = pool[:2] + fallback[:1]
        cands = mixed + pool[2:] + fallback[1:]
    return cands[:3]


def generate_mcqs(doc, keywords, n_questions=5, seed=None, difficulty="Medium"):
    """Cloze-style MCQs: blank a keyword in a sentence, add 3 wrong options.

    Works in rounds so one sentence can give several questions: round 1 takes
    the best keyword of every sentence, round 2 the next-best, and so on.
    """
    rng = random.Random(seed)
    kw_lower = [k.lower() for k in keywords]

    candidates = []
    for sent in doc.sents:
        text = sent.text.strip()
        if len(text.split()) < 4:
            continue
        in_sent = [k for k in keywords if re.search(rf"\b{re.escape(k)}\b", text, re.I)]
        in_sent.sort(key=lambda k: (re.search(re.escape(k), text, re.I).start() == 0,
                                    not k.isupper(), -len(k)))
        if in_sent:
            candidates.append((text, in_sent))

    questions, used = [], set()
    for rnd in range(max((len(c[1]) for c in candidates), default=0)):
        for text, in_sent in candidates:
            if len(questions) >= n_questions:
                return questions
            if rnd >= len(in_sent):
                continue
            answer = in_sent[rnd]
            if (text, answer.lower()) in used:
                continue
            used.add((text, answer.lower()))

            pool = [k for k in keywords if k.lower() != answer.lower()
                    and answer.lower() not in k.lower() and k.lower() not in answer.lower()
                    and not re.search(rf"\b{re.escape(k)}\b", text, re.I)]
            fallback = [w for w in FALLBACK_POOL if w.lower() != answer.lower() and w.lower() not in kw_lower]
            options = pick_distractors(answer, pool, fallback, difficulty, rng) + [answer]
            rng.shuffle(options)
            blanked = re.sub(rf"\b{re.escape(answer)}\b", "_____", text, count=1, flags=re.I)
            questions.append({
                "question": f"Fill in the blank: {blanked}",
                "options": options,
                "answer": answer,
                "explanation": f'The correct answer is "{answer}". Source sentence: {text}',
            })
    return questions

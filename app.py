import re
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

import nlp_utils as nu

BASE = Path(__file__).parent
HISTORY = BASE / "history.csv"
BANK = BASE / "question_bank.csv"
MAX_CHARS = 30000

SAMPLE = ("Machine Learning is a branch of AI that allows computers to learn from data. "
          "Supervised learning uses labeled data to train a model. "
          "Unsupervised learning finds hidden patterns in unlabeled data. "
          "Neural networks are inspired by the human brain and power deep learning.")

st.set_page_config(page_title="AI Study Buddy", page_icon="📚", layout="centered")


# ---------- helpers ----------
@st.cache_resource
def get_nlp():
    return nu.load_nlp()


@st.cache_data
def load_bank():
    return pd.read_csv(BANK)


def read_upload(f):
    """Extract plain text from an uploaded PDF or TXT file."""
    if f.name.lower().endswith(".pdf"):
        from pypdf import PdfReader
        text = "\n".join((p.extract_text() or "") for p in PdfReader(f).pages)
    else:
        text = f.read().decode("utf-8", errors="ignore")
    return re.sub(r"\s+", " ", text).strip()


def save_history(source, difficulty, score, total, seconds):
    row = pd.DataFrame([{
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": source, "difficulty": difficulty,
        "score": score, "total": total,
        "percent": round(score / total * 100, 1), "seconds": seconds,
    }])
    row.to_csv(HISTORY, mode="a", header=not HISTORY.exists(), index=False)


def grade(questions, answers):
    rows = []
    for i, q in enumerate(questions):
        chosen = answers.get(i)
        rows.append({
            "Question": q["question"], "Your answer": chosen, "Correct answer": q["answer"],
            "Result": "✅ Correct" if chosen == q["answer"] else "❌ Wrong",
            "Explanation": q["explanation"],
        })
    df = pd.DataFrame(rows)
    return df, int((df["Result"] == "✅ Correct").sum())


# ---------- quiz + timer ----------
def new_quiz(key, limit_sec):
    st.session_state[f"meta_{key}"] = {"id": time.time_ns(), "start": time.time(), "limit": limit_sec,
                                       "done": False, "saved": False, "taken": 0, "timed_out": False}


def finish(m, timed_out=False):
    if not m["done"]:
        m["done"], m["timed_out"] = True, timed_out
        m["taken"] = int(min(time.time() - m["start"], m["limit"] or 10 ** 9))


@st.fragment(run_every=1)
def timer_box(key):
    m = st.session_state[f"meta_{key}"]
    if m["done"] or not m["limit"]:
        return
    left = int(m["limit"] - (time.time() - m["start"]))
    if left <= 0:
        finish(m, timed_out=True)
        st.rerun()
    st.progress(left / m["limit"], text=f"⏱️ Time left: {left // 60:02d}:{left % 60:02d}")


def show_quiz(questions, key, source, difficulty):
    if f"meta_{key}" not in st.session_state:
        new_quiz(key, 0)
    m = st.session_state[f"meta_{key}"]
    if m["limit"]:
        timer_box(key)

    for i, q in enumerate(questions):
        st.radio(f"**Q{i + 1}. {q['question']}**", q["options"], index=None,
                 key=f"{key}_{m['id']}_{i}", disabled=m["done"])

    if not m["done"]:
        if st.button("Submit answers", key=f"submit_{key}_{m['id']}", type="primary"):
            finish(m)
            st.rerun()
        return

    answers = {i: st.session_state.get(f"{key}_{m['id']}_{i}") for i in range(len(questions))}
    df, score = grade(questions, answers)
    if not m["saved"]:
        save_history(source, difficulty, score, len(questions), m["taken"])
        m["saved"] = True

    pct = score / len(questions) * 100
    st.divider()
    if m["timed_out"]:
        st.warning("⏰ Time's up! Unanswered questions are counted as wrong.")
    st.subheader(f"📊 Score: {score}/{len(questions)} ({pct:.0f}%)")
    st.caption(f"Time taken: {m['taken'] // 60:02d}:{m['taken'] % 60:02d}  •  Level: {difficulty}")
    st.progress(pct / 100)
    if score == len(questions):
        st.success("Perfect score! 🎉")
    for _, r in df.iterrows():
        if r["Result"].startswith("✅"):
            st.success(f"{r['Result']} — {r['Correct answer']}")
        else:
            st.error(f"{r['Result']} — You chose: {r['Your answer'] or 'no answer'}")
            st.info(f"💡 {r['Explanation']}")
    st.download_button("Download results (CSV)", df.to_csv(index=False), "results.csv", "text/csv",
                       key=f"dl_{key}_{m['id']}")


# ---------- sidebar ----------
with st.sidebar:
    st.header("⚙️ Settings")
    difficulty = st.selectbox("Difficulty", ["Easy", "Medium", "Hard"], index=1,
                              help="Easy: obviously wrong options. Hard: believable options from your text.")
    timer_on = st.checkbox("Enable timer")
    minutes = st.number_input("Time limit (minutes)", 1, 60, 3) if timer_on else 0
    limit = int(minutes * 60) if timer_on else 0
    st.caption("Settings apply the next time you generate a quiz.")

st.title("📚 AI Study Buddy")
st.caption("Learn, revise and test yourself using NLP.")

tab1, tab2, tab3, tab4 = st.tabs(["📝 Study my text", "🃏 Flashcards", "🗂️ Question bank", "📊 Progress"])

# ---------- tab 1: study text ----------
with tab1:
    if "source_text" not in st.session_state:
        st.session_state["source_text"] = SAMPLE
    up = st.file_uploader("Upload a PDF or TXT file (optional)", type=["pdf", "txt"])
    if up is not None and st.session_state.get("last_upload") != (up.name, up.size):
        st.session_state["source_text"] = read_upload(up)[:MAX_CHARS]
        st.session_state["last_upload"] = (up.name, up.size)
        st.success(f"Loaded **{up.name}** — you can edit the text below before generating.")

    text = st.text_area("Enter a topic or paste study material", height=200, key="source_text")
    c1, c2 = st.columns(2)
    n_q = c1.slider("Number of questions", 1, 20, 8)
    n_s = c2.slider("Summary sentences", 1, 5, 2)

    if st.button("Generate study material", type="primary"):
        if len(text.split()) < 8:
            st.warning("Please enter a longer text (at least one full sentence).")
        else:
            doc = get_nlp()(text)
            kws = nu.extract_keywords(doc)
            st.session_state["study"] = {
                "summary": nu.summarize(doc, n_s),
                "keywords": kws,
                "mcqs": nu.generate_mcqs(doc, kws, n_q, difficulty=difficulty),
                "cards": nu.make_flashcards(doc, kws),
                "difficulty": difficulty,
            }
            new_quiz("quiz_text", limit)
            st.session_state["card_i"], st.session_state["card_show"] = 0, False

    study = st.session_state.get("study")
    if study:
        st.subheader("📖 Summary")
        st.write(study["summary"])
        st.subheader("🔑 Keywords")
        st.write(" ".join(f"`{k}`" for k in study["keywords"][:12]))
        st.subheader(f"❓ Quiz — {study['difficulty']}")
        if study["mcqs"]:
            show_quiz(study["mcqs"], "quiz_text", "Study text", study["difficulty"])
        else:
            st.info("Not enough content to make questions. Try a longer text.")

# ---------- tab 2: flashcards ----------
with tab2:
    cards = (st.session_state.get("study") or {}).get("cards")
    if not cards:
        st.info("Generate study material in the first tab to get flashcards.")
    else:
        i = st.session_state.get("card_i", 0) % len(cards)
        show = st.session_state.get("card_show", False)

        def go(step):
            st.session_state["card_i"] = (st.session_state.get("card_i", 0) + step) % len(cards)
            st.session_state["card_show"] = False

        def flip():
            st.session_state["card_show"] = not st.session_state.get("card_show", False)

        with st.container(border=True):
            st.caption(f"Card {i + 1} of {len(cards)}")
            st.markdown(f"## 🔑 {cards[i]['term']}")
            if show:
                st.write(cards[i]["definition"])
            else:
                st.write("*Think of the meaning, then press Flip.*")
        b1, b2, b3 = st.columns(3)
        b1.button("⬅️ Previous", on_click=go, args=(-1,), width="stretch")
        b2.button("🔄 Flip", on_click=flip, width="stretch")
        b3.button("Next ➡️", on_click=go, args=(1,), width="stretch")

# ---------- tab 3: question bank ----------
with tab3:
    bank = load_bank()
    st.write(f"Question bank: **{len(bank)}** questions (add more rows to `question_bank.csv`).")
    n = st.slider("Questions in this round", 1, min(10, len(bank)), min(5, len(bank)))
    if st.button("New round"):
        st.session_state["bank_q"] = bank.sample(n).to_dict("records")
        new_quiz("quiz_bank", limit)
    if "bank_q" in st.session_state:
        qs = []
        for r in st.session_state["bank_q"]:
            opts = sorted([r["A"], r["B"], r["C"], r["D"]])
            qs.append({"question": r["question"], "options": opts, "answer": r[r["answer"]],
                       "explanation": r["explanation"]})
        show_quiz(qs, "quiz_bank", "Question bank", "—")

# ---------- tab 4: progress ----------
with tab4:
    if not HISTORY.exists():
        st.info("No attempts yet. Finish a quiz and your score will appear here.")
    else:
        hist = pd.read_csv(HISTORY)
        a, b, c = st.columns(3)
        a.metric("Attempts", len(hist))
        b.metric("Average score", f"{hist['percent'].mean():.0f}%")
        c.metric("Best score", f"{hist['percent'].max():.0f}%")
        st.subheader("Score over time")
        st.line_chart(hist.reset_index().rename(columns={"index": "attempt"}).set_index("attempt")["percent"])
        st.subheader("History")
        st.dataframe(hist.iloc[::-1], width="stretch", hide_index=True)
        if st.button("🗑️ Clear history"):
            HISTORY.unlink()
            st.rerun()

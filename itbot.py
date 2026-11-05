# itbot: shared code for the Ubuntu IT support bot (data cleaning, retrieval, fine-tuning, evaluation, chat).
import html
import math
import re
import time
from functools import lru_cache

import numpy as np
import pandas as pd
from markdownify import markdownify

URL_RE = re.compile(r"https?://\S+|www\.\S+")


def parse_posts(xml_path):
    """Read Posts.xml of a Stack Exchange data dump into two tables: questions and answers (bodies still HTML)."""
    from lxml import etree
    q_rows, a_rows = [], []
    for _, el in etree.iterparse(str(xml_path), events=("end",), tag="row"):
        g = el.attrib.get
        kind = g("PostTypeId")
        if kind == "1":
            q_rows.append((int(g("Id")), g("CreationDate"), int(g("Score", 0)), int(g("ViewCount", 0) or 0),
                           g("Title", ""), g("Body", ""), g("Tags", ""), int(g("AnswerCount", 0) or 0),
                           int(g("CommentCount", 0) or 0), g("AcceptedAnswerId"), g("ClosedDate"), g("LastActivityDate"),
                           g("OwnerUserId")))
        elif kind == "2":
            a_rows.append((int(g("Id")), int(g("ParentId")), g("CreationDate"), int(g("Score", 0)), g("Body", ""),
                           int(g("CommentCount", 0) or 0), g("OwnerUserId")))
        el.clear()
        while el.getprevious() is not None:          # free the memory of rows already read
            del el.getparent()[0]
    q = pd.DataFrame(q_rows, columns=["question_id", "created", "score", "view_count", "title", "body_html", "tags",
                                      "answer_count", "comment_count", "accepted_answer_id", "closed_date", "last_activity",
                                      "owner_id"])
    a = pd.DataFrame(a_rows, columns=["answer_id", "question_id", "created", "score", "body_html", "comment_count", "owner_id"])
    for df, cols in ((q, ["created", "closed_date", "last_activity"]), (a, ["created"])):
        for c in cols:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    q["accepted_answer_id"] = pd.to_numeric(q["accepted_answer_id"], errors="coerce").astype("Int64")
    q["tags"] = q["tags"].fillna("").str.strip("|").str.split(r"\|").apply(lambda t: [x for x in t if x])
    return q, a


def parse_links(xml_path):
    """PostLinks.xml -> table of (question, related question, type). Type 3 = closed as a duplicate of."""
    from lxml import etree
    rows = []
    for _, el in etree.iterparse(str(xml_path), events=("end",), tag="row"):
        g = el.attrib.get
        rows.append((int(g("PostId")), int(g("RelatedPostId")), int(g("LinkTypeId"))))
        el.clear()
    return pd.DataFrame(rows, columns=["question_id", "related_id", "link_type"])


def html_to_text(raw):
    """Stack Exchange / docs HTML -> plain markdown text. Code blocks and inline code are kept,
    links keep only their text (no URLs) and images are dropped."""
    if not raw:
        return ""
    raw = re.sub(r"</?kbd>", "", raw)                             # <kbd>Ctrl</kbd>+<kbd>T</kbd> -> Ctrl+T
    text = markdownify(raw, heading_style="ATX", strip=["a", "img"], bullets="-")
    text = html.unescape(text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


# ---------------------------------------------------------------- cleaning
EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+(\.[\w-]+)+\b")
PROMPT_RE = re.compile(r"\b[a-z_][\w.-]*@[\w.-]+:(?=[~/])")       # john@john-laptop:~$  ->  user@host:~$
HOME_RE = re.compile(r"/home/(?!user\b)[A-Za-z0-9._-]+")
MENTION_RE = re.compile(r"(?<![\w`])@[A-Za-z][\w.-]{1,30}")
CODE_SPLIT_RE = re.compile(r"(```.*?```|`[^`\n]+`)", re.S)
CLOSING_RE = re.compile(r"(?im)^\s*(thanks|thank you|thx|any help|cheers|regards|best regards|help me please)\b[^\n]{0,60}$")
VERSION_TAG_RE = re.compile(r"^\d{2}\.\d{2}$")
FIRST_PERSON_RE = (r"(?i)^\s*(i (had|have had|found|solved|fixed|figured|got it|managed|ended|finally|think i|was able|resolved|did)"
                   r"|update\b|edit\b|solved\b|thanks\b|ok,? (i|so)|so i )")


def _outside_code(text, fn):
    """Apply fn only to the parts of text that are not code, so commands are never changed."""
    parts = CODE_SPLIT_RE.split(text)
    return "".join(p if p.startswith("`") else fn(p) for p in parts)


def _domain(url):
    m = re.match(r"(?:https?://)?(?:www\.)?([^/\s)\]]+)", url)
    return f"[link: {m.group(1)}]" if m else "[link]"


def tidy(text):
    """Privacy and noise clean-up of converted text: e-mails, user names in paths and prompts, @mentions,
    bare URLs (kept as their domain) and thank-you lines. Code is only touched for the privacy rules."""
    text = EMAIL_RE.sub("[email]", text)
    text = PROMPT_RE.sub("user@host:", text)
    text = HOME_RE.sub("/home/user", text)
    text = _outside_code(text, lambda p: MENTION_RE.sub("", URL_RE.sub(lambda m: _domain(m.group(0)), p)))
    text = _outside_code(text, lambda p: CLOSING_RE.sub("", p))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def text_stats(text):
    """Simple numbers about a post, used for the charts and the filters."""
    blocks = re.findall(r"```.*?```", text, re.S)
    inline = re.findall(r"`[^`\n]+`", re.sub(r"```.*?```", " ", text, flags=re.S))
    return {"words": len(text.split()), "code_blocks": len(blocks), "inline_code": len(inline),
            "has_code": bool(blocks or inline)}


def pick_reference(q, a):
    """The answer the bot should learn from: the accepted answer if the community did not vote it down
    (score >= 1), otherwise the best-voted answer if at least two people up-voted it."""
    a = a.merge(q[["question_id", "accepted_answer_id"]], on="question_id", how="inner")
    acc = a[(a.answer_id == a.accepted_answer_id) & (a.score >= 1)].assign(ref_type="accepted")
    top = (a[~a.question_id.isin(acc.question_id) & (a.score >= 2)]
           .sort_values(["question_id", "score", "created"], ascending=[True, False, True])
           .drop_duplicates("question_id").assign(ref_type="top_voted"))
    return pd.concat([acc, top]).drop(columns="accepted_answer_id")


def clean_qa(q, a, links, max_answer_words=800, max_question_words=600):
    """Raw dump tables -> one clean question/answer pair per question, plus the cleaning funnel."""
    funnel = {"all questions in the dump": len(q)}
    q = q[q.closed_date.isna()]
    funnel["not closed (closed, duplicate or off-topic removed)"] = len(q)
    q = q[q.score >= 0]
    funnel["not voted down (score >= 0)"] = len(q)
    ref = pick_reference(q, a)
    q = q.merge(ref[["question_id", "answer_id", "created", "score", "body_html", "ref_type", "owner_id"]]
                .rename(columns={"created": "answer_created", "score": "a_score", "body_html": "answer_html", "owner_id": "a_owner"}),
                on="question_id", how="inner")
    funnel["has a trusted answer (accepted, or 2+ up-votes)"] = len(q)
    # the asker answered their own question: often written as "I found the fix ...". Only this flag is kept,
    # the user ids themselves are dropped
    q["self_answer"] = q.owner_id.notna() & (q.owner_id == q.a_owner)
    q = q.drop(columns=["owner_id", "a_owner"])

    q = q.rename(columns={"score": "q_score"})
    q["title"] = q.title.map(html.unescape).str.strip()
    q["question"] = [tidy(html_to_text(h)) for h in q.body_html]
    q["answer"] = [tidy(html_to_text(h)) for h in q.answer_html]
    q["answer_had_image"] = q.answer_html.str.contains("<img", regex=False)
    q = q.drop(columns=["body_html", "answer_html"])
    for side, col in (("q", "question"), ("a", "answer")):
        st = pd.DataFrame([text_stats(t) for t in q[col]], index=q.index)
        for c in st.columns:
            q[f"{side}_{c}"] = st[c]

    short = (q.a_words < 15) | (q.answer_had_image & (q.a_words < 40))
    q = q[~short]
    funnel["answer is more than a link or a screenshot (15+ words)"] = len(q)
    q = q[(q.a_words <= max_answer_words) & (q.q_words <= max_question_words) & (q.q_words >= 3)]
    funnel[f"lengths in range (question 3-{max_question_words} words, answer up to {max_answer_words})"] = len(q)
    q = q.sort_values("a_score", ascending=False)
    key = (q.title.str.lower().str.replace(r"\W+", " ", regex=True).str.strip() + "|" +
           q.question.str.lower().str.replace(r"\W+", " ", regex=True).str[:300])
    q = q[~key.duplicated()].sort_values("question_id")
    funnel["no exact duplicate questions"] = len(q)

    q["year"] = q.created.dt.year
    q["hours_to_answer"] = (q.answer_created - q.created).dt.total_seconds() / 3600
    q.loc[q.hours_to_answer < 0, "hours_to_answer"] = np.nan      # answer older than the question: merged posts
    q["versions"] = q.tags.apply(lambda t: [x for x in t if VERSION_TAG_RE.match(x)])
    n_dup = links[links.link_type == 3].groupby("related_id").size()
    q["n_duplicates"] = q.question_id.map(n_dup).fillna(0).astype(int)
    q["first_person"] = q.answer.str.match(FIRST_PERSON_RE)
    q["url"] = "https://askubuntu.com/q/" + q.question_id.astype(str)
    q["source"] = "dump"
    return q.reset_index(drop=True), funnel


def paraphrases(q_raw, links, qa):
    """Questions closed as duplicates are real re-wordings of an original question. Returns one row per
    duplicate whose original is in the clean data: (duplicate title + body, original question id)."""
    d = links[links.link_type == 3].rename(columns={"question_id": "dup_id", "related_id": "question_id"})
    d = d[d.question_id.isin(qa.question_id)]
    raw = q_raw.set_index("question_id")
    d = d[d.dup_id.isin(raw.index)].drop_duplicates("dup_id")
    d["dup_title"] = [html.unescape(raw.at[i, "title"]).strip() for i in d.dup_id]
    d["dup_question"] = [tidy(html_to_text(raw.at[i, "body_html"])) for i in d.dup_id]
    d["dup_created"] = [raw.at[i, "created"] for i in d.dup_id]
    return d.reset_index(drop=True)


def clean_api(rows):
    """Recent questions from the Stack Exchange API -> same columns as clean_qa (same answer rule)."""
    out = []
    for r in rows:
        if r.get("closed_reason") or r["score"] < 0:
            continue
        ans = r["answers"]
        acc = [x for x in ans if x["is_accepted"] and x["score"] >= 1]
        top = sorted([x for x in ans if x["score"] >= 2], key=lambda x: -x["score"])
        ref, kind = (acc[0], "accepted") if acc else ((top[0], "top_voted") if top else (None, None))
        if ref is None:
            continue
        answer, question = tidy(ref["body"]), tidy(r["body"])
        sa, sq = text_stats(answer), text_stats(question)
        if sa["words"] < 15 or sa["words"] > 800 or sq["words"] > 600:
            continue
        created = pd.to_datetime(r["creation_date"], unit="s")
        out.append({"question_id": r["question_id"], "created": created, "q_score": r["score"], "view_count": r["view_count"],
                    "title": html.unescape(r["title"]), "tags": r["tags"], "answer_count": r["answer_count"],
                    "answer_id": ref["answer_id"], "a_score": ref["score"], "ref_type": kind, "question": question,
                    "answer": answer, "q_words": sq["words"], "a_words": sa["words"], "a_has_code": sa["has_code"],
                    "q_has_code": sq["has_code"], "year": created.year,
                    "hours_to_answer": max(ref["creation_date"] - r["creation_date"], 0) / 3600,
                    "versions": [x for x in r["tags"] if VERSION_TAG_RE.match(x)],
                    "self_answer": False, "first_person": bool(re.match(FIRST_PERSON_RE, answer)),
                    "url": f"https://askubuntu.com/q/{r['question_id']}", "source": "api"})
    return pd.DataFrame(out)


# ---------------------------------------------------------------- classic NLP pre-processing (for the keyword search)
_STOP, _LEMMA = None, None
KEEP_WORDS = {"not", "no", "can", "cannot", "won", "doesn", "isn", "how", "why", "what", "where", "up", "down", "off", "on", "out"}


def _nltk_ready():
    global _STOP, _LEMMA
    if _STOP is None:
        import os
        # NLTK 3.10 refuses downloads through a proxy unless told the proxy is trusted; Colab's egress proxy is
        os.environ.setdefault("NLTK_ALLOW_PROXIED_URLOPEN", "1")
        import nltk
        for pkg in ("stopwords", "wordnet", "omw-1.4"):
            nltk.download(pkg, quiet=True)
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        _STOP = set(stopwords.words("english")) - KEEP_WORDS
        _LEMMA = WordNetLemmatizer()


LEMMA_FIX = {"instal": "install", "installing": "install", "installed": "install", "windows": "windows",
             "doe": "do", "os": "os", "gnome": "gnome", "ubuntu": "ubuntu", "lts": "lts"}


@lru_cache(maxsize=500_000)
def _lemma(w):
    """WordNet lemma: verb form first ('does' -> 'do'), otherwise noun form. WordNet maps 'installing' to the
    British 'instal' and 'Windows' to 'window', so a small list of IT words is fixed by hand."""
    if w in LEMMA_FIX:
        return LEMMA_FIX[w]
    v = _LEMMA.lemmatize(w, "v")
    out = v if v != w else _LEMMA.lemmatize(w)
    return LEMMA_FIX.get(out, out)


def classic_preprocess(text, expand=True, lower=True, drop_urls=True, drop_digit_words=False, drop_stop=True, lemmatize=True):
    """The template's text pre-processing chain. Each step can be switched off, because some of them hurt in
    this domain (version numbers such as 22.04 or package names such as python3 contain digits)."""
    import contractions
    _nltk_ready()
    if expand:
        text = contractions.fix(text)
    if lower:
        text = text.lower()
    if drop_urls:
        text = URL_RE.sub(" ", re.sub(r"\[link: [^\]]+\]", " ", text))
    text = re.sub(r"[^\w.]+|(?<!\d)\.|\.(?!\d)", " ", text)            # punctuation out, keep dots inside 22.04
    words = text.split()
    if drop_digit_words:
        words = [w for w in words if not any(c.isdigit() for c in w)]
    if drop_stop:
        words = [w for w in words if w not in _STOP]
    if lemmatize:
        words = [_lemma(w) for w in words]
    return " ".join(words)


# ---------------------------------------------------------------- the bot's library and the search engines
def thread_items(qa, max_answer_words=90):
    """Solved threads -> library items. Searching uses the question side, the prompt shows the answer."""
    return pd.DataFrame({
        "item_id": "t" + qa.question_id.astype(str), "kind": "thread", "question_id": qa.question_id.values,
        "title": qa.title.values, "year": qa.year.values, "score": qa.a_score.values, "url": qa.url.values,
        "search_text": (qa.title + "\n" + qa.question.map(lambda t: " ".join(t.split()[:150]))).values,
        "show_text": qa.answer.map(lambda t: shorten(t, max_answer_words)).values})


def doc_items(docs, max_words=90):
    return pd.DataFrame({
        "item_id": "d" + docs.doc_id.astype(str), "kind": "doc", "question_id": -1, "title": docs.title.values,
        "year": 2026, "score": 0, "url": docs.url.values,
        "search_text": (docs.title + " - " + docs.section + "\n" + docs.text).values,
        "show_text": (docs.section.where(docs.section != docs.title, "") + "\n" + docs.text.map(lambda t: shorten(t, max_words))).str.strip().values})


def shorten(text, max_words):
    """Cut a text after max_words words without breaking a code block in the middle."""
    words = text.split(" ")
    if len(text.split()) <= max_words:
        return text
    out, n = [], 0
    for w in words:
        out.append(w)
        n += len(w.split())
        if n >= max_words:
            break
    cut = " ".join(out)
    if cut.count("```") % 2:
        cut += "\n```"
    return cut + " ..."


def topk(scores, k):
    """Row-wise top-k of a dense score matrix -> (indices, scores), best first."""
    k = min(k, scores.shape[1])
    idx = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    part = np.take_along_axis(scores, idx, axis=1)
    order = np.argsort(-part, axis=1)
    return np.take_along_axis(idx, order, axis=1), np.take_along_axis(part, order, axis=1)


class TfidfSearch:
    """Keyword search: TF-IDF vectors of the pre-processed text, cosine similarity."""
    def __init__(self, prep=None, **tfidf):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.prep = prep or {}
        self.vec = TfidfVectorizer(token_pattern=r"(?u)\b[\w.]+\b", dtype=np.float32, **tfidf)

    def _p(self, texts):
        return [classic_preprocess(t, **self.prep) for t in texts]

    def fit(self, texts):
        self.X = self.vec.fit_transform(self._p(texts))
        return self

    def search(self, queries, k=10, chunk=512):
        Q = self.vec.transform(self._p(queries))
        out_i, out_s = [], []
        for s in range(0, Q.shape[0], chunk):
            S = (Q[s:s + chunk] @ self.X.T).toarray()
            i, v = topk(S, k)
            out_i.append(i)
            out_s.append(v)
        return np.vstack(out_i), np.vstack(out_s)


class BM25Search(TfidfSearch):
    """Okapi BM25 on the same pre-processed text, computed as one sparse matrix so a search is a matrix product.
    fit() counts the words once; set_params() only recomputes the weights, which makes a grid search fast."""
    def __init__(self, prep=None, k1=1.2, b=0.75, **count):
        from sklearn.feature_extraction.text import CountVectorizer
        self.prep, self.k1, self.b = prep or {}, k1, b
        self.vec = CountVectorizer(token_pattern=r"(?u)\b[\w.]+\b", dtype=np.float32, **count)

    def fit(self, texts=None, prepped=None):
        self.C = self.vec.fit_transform(prepped if prepped is not None else self._p(texts)).tocsr().astype(np.float32)
        return self.set_params(self.k1, self.b)

    def set_params(self, k1, b):
        self.k1, self.b = k1, b
        X = self.C.copy()
        n = X.shape[0]
        df = np.bincount(X.indices, minlength=X.shape[1])
        idf = np.log1p((n - df + 0.5) / (df + 0.5)).astype(np.float32)
        dl = np.asarray(X.sum(1)).ravel()
        rows = np.repeat(np.arange(n), np.diff(X.indptr))
        tf = X.data
        X.data = idf[X.indices] * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl[rows] / dl.mean()))
        self.X = X
        return self

    def search(self, queries, k=10, chunk=512, prepped=None):
        Q = self.vec.transform(prepped if prepped is not None else self._p(queries))
        Q.data[:] = 1.0                                   # each query word counts once
        out_i, out_s = [], []
        for s in range(0, Q.shape[0], chunk):
            S = (Q[s:s + chunk] @ self.X.T).toarray()
            i, v = topk(S, k)
            out_i.append(i)
            out_s.append(v)
        return np.vstack(out_i), np.vstack(out_s)


class DenseSearch:
    """Semantic search: sentence embeddings (bge-small by default) and cosine similarity."""
    def __init__(self, model_name="BAAI/bge-small-en-v1.5", device=None, batch_size=256, max_length=256):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, device=device)
        self.model.max_seq_length = max_length
        self.batch_size = batch_size

    def encode(self, texts, show=False):
        return self.model.encode(list(texts), batch_size=self.batch_size, normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=show).astype(np.float32)

    def fit(self, texts=None, emb=None, show=False):
        self.E = emb if emb is not None else self.encode(texts, show)
        return self

    def search(self, queries, k=10, chunk=1024, q_emb=None):
        Q = q_emb if q_emb is not None else self.encode(queries)
        out_i, out_s = [], []
        for s in range(0, len(Q), chunk):
            i, v = topk(Q[s:s + chunk] @ self.E.T, k)
            out_i.append(i)
            out_s.append(v)
        out_s = np.vstack(out_s)
        self.last_dense_sim = out_s[:, 0]
        return np.vstack(out_i), out_s


class HybridSearch:
    """Reciprocal rank fusion of keyword search (BM25) and semantic search: an item ranked high by either wins."""
    def __init__(self, lexical, dense, k_rrf=60, depth=50, dense_weight=1.0):
        self.lexical, self.dense, self.k_rrf, self.depth, self.w = lexical, dense, k_rrf, depth, dense_weight

    def search(self, queries, k=10, q_emb=None):
        li, _ = self.lexical.search(queries, self.depth)
        di, ds = self.dense.search(queries, self.depth, q_emb=q_emb)
        out_i, out_s, best_sim = [], [], []
        for r in range(len(queries)):
            sc = {}
            for rank, j in enumerate(li[r]):
                sc[j] = sc.get(j, 0) + 1 / (self.k_rrf + rank + 1)
            for rank, j in enumerate(di[r]):
                sc[j] = sc.get(j, 0) + self.w / (self.k_rrf + rank + 1)
            top = sorted(sc.items(), key=lambda x: -x[1])[:k]
            out_i.append([j for j, _ in top])
            out_s.append([s for _, s in top])
            best_sim.append(ds[r, 0])
        self.last_dense_sim = np.array(best_sim)              # used by the bot to say "not sure"
        return np.array(out_i), np.array(out_s)


def retrieval_scores(found_ids, true_ids, ks=(1, 5, 10)):
    """found_ids: list of ranked id lists; true_ids: the right id per query. Recall@k and mean reciprocal rank."""
    res = {}
    ranks = []
    for f, t in zip(found_ids, true_ids):
        f = list(f)
        ranks.append(f.index(t) + 1 if t in f else np.inf)
    ranks = np.array(ranks)
    for k in ks:
        res[f"recall@{k}"] = float(np.mean(ranks <= k))
    res["MRR@10"] = float(np.mean(np.where(ranks <= 10, 1 / ranks, 0)))
    return res


# ---------------------------------------------------------------- prompts
SYSTEM_PROMPT = ("You are an IT support assistant for Ubuntu and Linux. Help the user like an experienced Ask Ubuntu "
                 "member: get to the point, give the exact commands in code blocks and briefly explain what they do. "
                 "If the question is not about Ubuntu, Linux or computers, say that you can only help with Ubuntu "
                 "and Linux questions.")
REFUSAL = ("Sorry, I can only help with Ubuntu and Linux questions, for example installing software, drivers, "
           "networking, the terminal or system errors.")
REFUSAL_RE = re.compile(r"(?i)(only help with (ubuntu|linux|questions about)|can only (help|assist|answer)[^.]{0,40}(ubuntu|linux)|"
                        r"not (related to|about) (ubuntu|linux)|outside (of )?(my|the) (scope|expertise)|i can'?t help with that)")


def is_refusal(text):
    return bool(REFUSAL_RE.search(text[:400]))


# Teacher rewrite of a trusted answer into a clear help-desk reply (scripts/refine_answers.py). The rewrite keeps the
# answer's fix and commands; refine_ok() rejects rewrites that lost them or slipped into the asker's voice.
REFINE_SYSTEM = "You are a senior Ubuntu support engineer. You write the replies an IT help desk sends to its users."
REFINE_TASK = """A user asked this question:

<question>
{question}
</question>

This answer from an experienced Ask Ubuntu member solved the problem:

<trusted_answer>
{answer}
</trusted_answer>

Write the reply the help desk should send to the user. Rules:
1. Use the trusted answer's solution. Keep every command, package name, file path and setting it relies on, exactly as written there.
2. Do not add other commands, packages or claims. A one-line check that the fix worked is fine.
3. Start with one or two sentences that say what to do (and why, if the trusted answer explains it). Then give numbered steps, with commands in ``` code blocks.
4. Speak to the user as "you". Never write as the person who asked. No greeting, no thanks, no sign-off, no links.
5. Keep it under 220 words."""
_FIRST_PERSON_START = re.compile(r"^\s*(I|I'm|I've|I'd|My|We)\b")


def refine_prompt(tok, question, answer):
    return tok.apply_chat_template([{"role": "system", "content": REFINE_SYSTEM},
                                    {"role": "user", "content": REFINE_TASK.format(question=question, answer=answer)}],
                                   add_generation_prompt=True, tokenize=False)


def refine_ok(rewrite, answer):
    """Keep a rewrite only if it has a sensible length, speaks to the user, and keeps 60%+ of the answer's commands."""
    w = len(rewrite.split())
    if w < 25 or w > 320 or _FIRST_PERSON_START.search(rewrite) or is_refusal(rewrite) or "trusted answer" in rewrite.lower():
        return False
    return command_recall(rewrite, answer) >= 0.6 if command_words(answer) else True


def format_question(title, body, max_words=200):
    body = shorten(body, max_words) if body else ""
    return f"{title.strip()}\n\n{body}".strip()


def format_context(items):
    """Library rows -> the text block placed before the question."""
    lines = ["Related solved threads and documentation (use them only if they fit the problem):"]
    for n, r in enumerate(items.itertuples(), 1):
        src = f"Ask Ubuntu thread ({r.year})" if r.kind == "thread" else "Ubuntu documentation"
        lines.append(f"[{n}] {src}: {r.title}\n{r.show_text}")
    return "\n\n".join(lines)


def build_messages(question, context=None, history=None):
    """question: text of the user's question (already formatted); context: library rows or None;
    history: earlier (user, assistant) turns of the chat."""
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
    for u, a in (history or [])[-2:]:
        msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    user = question if context is None or len(context) == 0 else format_context(context) + "\n\nQuestion:\n" + question
    msgs.append({"role": "user", "content": user})
    return msgs


def chat_prompt(tok, msgs):
    return tok.apply_chat_template(msgs, add_generation_prompt=True, enable_thinking=False, tokenize=False)


# ---------------------------------------------------------------- the language model: loading, LoRA training, answers
def get_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_llm(name="Qwen/Qwen3-1.7B-Base", device=None, dtype=None, adapter=None):
    """Load the chat model (and optionally a LoRA adapter). fp16 on an NVIDIA GPU, fp32 on the CPU.
    The weights go straight to the device, so a model never takes up the machine's RAM on the way to the GPU."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    device = device or get_device()
    dtype = dtype or (torch.float16 if device == "cuda" else torch.float32)
    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, attn_implementation="sdpa", device_map=device)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
    model.eval()
    return model, tok


def add_lora(model, r=16, alpha=32, dropout=0.05, targets=("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")):
    """Wrap the model with LoRA adapters on all attention and MLP projections. Only the adapters train;
    they are kept in fp32 so mixed precision training is stable."""
    import torch
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=list(targets), task_type="CAUSAL_LM")
    model = get_peft_model(model, cfg)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    return model


def encode_example(tok, msgs, answer=None, max_len=1024):
    """Chat messages (+ the answer to learn) -> token ids of the prompt and of the completion. The answer ends with
    the tokenizer's own end token: <|im_end|> for an instruction-tuned Qwen, <|endoftext|> for a base model, which never
    learned <|im_end|> in pre-training (its output weights for that token are tiny, so it would not learn to stop)."""
    p = tok(chat_prompt(tok, msgs), add_special_tokens=False)["input_ids"]
    c = [] if answer is None else tok(answer, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
    if len(p) + len(c) > max_len:
        return None
    return {"prompt": p, "completion": c}


def make_batches(examples, batch_size, shuffle=True, seed=0):
    """Group examples of similar length into batches (less padding = faster training)."""
    rng = np.random.default_rng(seed)
    order = np.argsort([len(e["prompt"]) + len(e["completion"]) for e in examples], kind="stable")
    batches = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    if shuffle:
        rng.shuffle(batches)
    return batches


def collate(examples, idx, pad_id, device):
    """Left-pad a batch. Returns ids, mask, position ids and the completion mask of the last K positions."""
    import torch
    seqs = [examples[i]["prompt"] + examples[i]["completion"] for i in idx]
    comp = [len(examples[i]["completion"]) for i in idx]
    L, K = max(map(len, seqs)), max(comp)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), L), dtype=torch.long)
    cmask = torch.zeros((len(seqs), K), dtype=torch.bool)
    for r, (s, c) in enumerate(zip(seqs, comp)):
        ids[r, L - len(s):] = torch.tensor(s)
        mask[r, L - len(s):] = 1
        cmask[r, K - c:] = True
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    return ids.to(device), mask.to(device), pos.to(device), cmask.to(device), K


def completion_loss(model, ids, mask, pos, cmask, K, reduce=True):
    """Cross-entropy on the answer tokens only. Logits are computed for the last K+1 positions only,
    which saves a lot of memory with a 152k-word vocabulary."""
    import torch
    import torch.nn.functional as F
    out = model(input_ids=ids, attention_mask=mask, position_ids=pos, logits_to_keep=K + 1, use_cache=False)
    logits = out.logits[:, :-1].float()
    target = ids[:, -K:]
    nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1), reduction="none").view(target.shape)
    nll = nll * cmask
    if reduce:
        return nll.sum() / cmask.sum()
    return nll.sum(1), cmask.sum(1)


def train_lora(model, tok, train_ex, val_ex, lr=2e-4, epochs=2, batch_size=4, grad_accum=4, warmup=0.05,
               eval_every=50, patience=4, max_minutes=None, max_steps=None, log=print, seed=0):
    """Fine-tune the LoRA adapters with AdamW, a cosine schedule, fp16 mixed precision (on CUDA), early stopping
    on the validation loss and an optional time limit. Keeps the best adapter weights. Returns the history."""
    import torch
    from transformers import get_cosine_schedule_with_warmup
    device = next(model.parameters()).device
    cuda = device.type == "cuda"
    torch.manual_seed(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0, betas=(0.9, 0.999))
    n_batches = len(make_batches(train_ex, batch_size))
    steps_per_epoch = math.ceil(n_batches / grad_accum)
    total = steps_per_epoch * epochs if max_steps is None else min(max_steps, steps_per_epoch * epochs)
    sched = get_cosine_schedule_with_warmup(opt, max(1, int(warmup * total)), total)
    scaler = torch.amp.GradScaler("cuda", enabled=cuda)
    best, best_state, bad, step, hist = np.inf, None, 0, 0, []
    t0, tokens, run_loss, run_n = time.time(), 0, 0.0, 0
    stop = False
    for ep in range(epochs):
        model.train()
        for b_i, idx in enumerate(make_batches(train_ex, batch_size, seed=seed + ep)):
            ids, mask, pos, cmask, K = collate(train_ex, idx, tok.pad_token_id, device)
            with torch.autocast("cuda", dtype=torch.float16, enabled=cuda):
                loss = completion_loss(model, ids, mask, pos, cmask, K)
            scaler.scale(loss / grad_accum).backward()
            run_loss, run_n, tokens = run_loss + loss.item(), run_n + 1, tokens + int(mask.sum())
            if (b_i + 1) % grad_accum:
                continue
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
            step += 1
            if step % eval_every == 0 or step == total:
                vl = evaluate_loss(model, tok, val_ex, batch_size * 2)
                el = (time.time() - t0) / 60
                hist.append({"step": step, "epoch": ep + (b_i + 1) / n_batches,
                             "train_loss": run_loss / run_n, "val_loss": vl, "lr": sched.get_last_lr()[0],
                             "minutes": el, "tokens_per_s": tokens / (time.time() - t0)})
                log(f"step {step}/{total}  epoch {hist[-1]['epoch']:.2f}  train {run_loss / run_n:.4f}  val {vl:.4f}  "
                    f"{tokens / (time.time() - t0):.0f} tok/s  {el:.1f} min")
                run_loss, run_n = 0.0, 0
                if vl < best - 1e-4:
                    best, bad = vl, 0
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items() if "lora_" in k}
                else:
                    bad += 1
                    if bad >= patience:
                        log("early stopping: the validation loss stopped improving")
                        stop = True
                model.train()
            if (max_minutes and (time.time() - t0) / 60 > max_minutes) or step >= total:
                if max_minutes and (time.time() - t0) / 60 > max_minutes:
                    log(f"time limit of {max_minutes} min reached")
                stop = True
            if stop:
                break
        if stop:
            break
    if best_state is not None:
        model.load_state_dict(best_state, strict=False)
    model.eval()
    return pd.DataFrame(hist)


def evaluate_loss(model, tok, examples, batch_size=8):
    """Mean cross-entropy per answer token on a set of examples."""
    import torch
    device = next(model.parameters()).device
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
        for idx in make_batches(examples, batch_size, shuffle=False):
            ids, mask, pos, cmask, K = collate(examples, idx, tok.pad_token_id, device)
            s, c = completion_loss(model, ids, mask, pos, cmask, K, reduce=False)
            tot, n = tot + float(s.sum()), n + int(c.sum())
    return tot / n


def answer_nll(model, tok, examples, batch_size=8):
    """Per-example mean negative log-likelihood of the reference answer (perplexity = exp of it)."""
    import torch
    device = next(model.parameters()).device
    out = np.zeros(len(examples))
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16, enabled=device.type == "cuda"):
        for idx in make_batches(examples, batch_size, shuffle=False):
            ids, mask, pos, cmask, K = collate(examples, idx, tok.pad_token_id, device)
            s, c = completion_loss(model, ids, mask, pos, cmask, K, reduce=False)
            out[idx] = (s / c).float().cpu().numpy()
    return out


def generate(model, tok, prompts, max_new_tokens=384, batch_size=16, do_sample=False, temperature=0.7, top_p=0.8,
             top_k=20, repetition_penalty=1.05, log=None):
    """Batched answers for chat-formatted prompts. Long prompts go first so memory peaks early."""
    import torch
    device = next(model.parameters()).device
    tok.padding_side = "left"
    order = np.argsort([-len(p) for p in prompts], kind="stable")
    out = [None] * len(prompts)
    stop_ids = [tok.convert_tokens_to_ids("<|im_end|>"), tok.convert_tokens_to_ids("<|endoftext|>")]
    t0 = time.time()
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        kw = dict(do_sample=True, temperature=temperature, top_p=top_p, top_k=top_k) if do_sample else dict(do_sample=False)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=max_new_tokens, repetition_penalty=repetition_penalty,
                                 eos_token_id=stop_ids, pad_token_id=tok.pad_token_id, **kw)
        for j, i in enumerate(idx):
            text = tok.decode(gen[j, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            out[i] = re.sub(r"(?s)<think>.*?</think>", "", text).strip()
        if log and (s // batch_size) % 5 == 0:
            log(f"  {min(s + batch_size, len(order))}/{len(order)} answers, {time.time() - t0:.0f} s")
    return out


# ---------------------------------------------------------------- evaluation
CMD_RE = re.compile(r"[A-Za-z][\w.+/-]{1,60}")
COMMON = {"sudo", "the", "to", "a", "and", "your", "you", "file", "then", "run", "this", "is", "in", "of", "for", "it",
          "or", "with", "be", "if", "on", "that", "can", "use", "will", "from", "as", "by", "are", "an", "at", "not"}


def command_words(text, code_only=True):
    """Commands, package names, options and paths: the words inside code blocks and inline code."""
    if code_only:
        text = " ".join(re.findall(r"```(.*?)```", text, re.S) + re.findall(r"`([^`\n]+)`", text))
    return {w.lower().strip("./") for w in CMD_RE.findall(text)} - COMMON - {""}


def command_recall(pred, ref):
    """Share of the reference answer's commands and package names that the bot's answer also mentions."""
    want = command_words(ref)
    if not want:
        return np.nan
    return len(want & command_words(pred, code_only=False)) / len(want)


def rouge_l(preds, refs):
    from rouge_score import rouge_scorer
    sc = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    return np.array([sc.score(r, p)["rougeL"].fmeasure for p, r in zip(preds, refs)])


class BertScorer:
    """BERTScore F1 (Zhang et al. 2020): token embeddings from roberta-large layer 17, greedy matching of
    each token to its most similar token in the other text. Same settings as the bert-score package default."""
    def __init__(self, name="roberta-large", layer=17, device=None):
        import torch
        from transformers import AutoModel, AutoTokenizer
        self.device = device or get_device()
        self.tok = AutoTokenizer.from_pretrained(name)
        self.model = AutoModel.from_pretrained(name).to(self.device).eval()
        if self.device == "cuda":
            self.model = self.model.half()
        self.layer, self.torch = layer, torch

    def _emb(self, texts):
        torch = self.torch
        enc = self.tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=510).to(self.device)
        with torch.no_grad():
            h = self.model(**enc, output_hidden_states=True).hidden_states[self.layer].float()
        h = torch.nn.functional.normalize(h, dim=-1)
        m = enc["attention_mask"].clone()
        m[:, 0] = 0                                                      # drop <s>
        m[torch.arange(len(texts)), enc["attention_mask"].sum(1) - 1] = 0  # drop </s>
        return h, m.bool()

    def score(self, preds, refs, batch_size=32):
        out = []
        for s in range(0, len(preds), batch_size):
            hp, mp = self._emb([p or "." for p in preds[s:s + batch_size]])
            hr, mr = self._emb([r or "." for r in refs[s:s + batch_size]])
            sim = hp @ hr.transpose(1, 2)
            sim = sim.masked_fill(~mr[:, None, :], -2).masked_fill(~mp[:, :, None], -2)
            P = (sim.max(2).values * mp).sum(1) / mp.sum(1)
            R = (sim.max(1).values * mr).sum(1) / mr.sum(1)
            out += (2 * P * R / (P + R)).cpu().tolist()
        return np.array(out)


def answer_scores(preds, refs, dense=None, bert=None):
    """One row per answer: ROUGE-L, BERTScore F1, semantic similarity, command recall, length, refusal."""
    df = pd.DataFrame({"rougeL": rouge_l(preds, refs),
                       "command_recall": [command_recall(p, r) for p, r in zip(preds, refs)],
                       "words": [len(p.split()) for p in preds], "refused": [is_refusal(p) for p in preds]})
    if bert is not None:
        df["bertscore"] = bert.score(preds, refs)
    if dense is not None:
        df["semantic"] = (dense.encode(preds) * dense.encode(refs)).sum(1)
    return df


JUDGE_PROMPT = """You are checking answers given by an Ubuntu IT support bot.

User question:
{question}

Reference answer (accepted by the community on Ask Ubuntu):
{reference}

Answer A:
{a}

Answer B:
{b}

Which answer would solve the user's problem better? Judge correctness against the reference first, then whether the steps are clear and safe. Reply with a single letter: A, B, or T if they are equally good."""


def judge_pairs(model, tok, items, batch_size=4, max_chars=2500):
    """Pairwise LLM judge. items: dicts with question, reference, x, y. Each pair is judged twice with the
    order swapped (judges tend to favour the first answer); returns P(x better) from the A/B/T logits."""
    import torch
    device = next(model.parameters()).device
    letters = [tok.encode(l, add_special_tokens=False)[0] for l in ("A", "B", "T")]
    prompts, meta = [], []
    for n, it in enumerate(items):
        for swap in (False, True):
            a, b = (it["y"], it["x"]) if swap else (it["x"], it["y"])
            text = JUDGE_PROMPT.format(question=it["question"][:max_chars], reference=it["reference"][:max_chars],
                                       a=a[:max_chars], b=b[:max_chars])
            prompts.append(tok.apply_chat_template([{"role": "user", "content": text}], add_generation_prompt=True, tokenize=False))
            meta.append((n, swap))
    probs = np.zeros((len(prompts), 3))
    tok.padding_side = "left"
    for s in range(0, len(prompts), batch_size):
        enc = tok(prompts[s:s + batch_size], return_tensors="pt", padding=True, add_special_tokens=False).to(device)
        with torch.no_grad():
            logits = model(**enc, logits_to_keep=1).logits[:, -1, letters].float()
        probs[s:s + batch_size] = torch.softmax(logits, -1).cpu().numpy()
    res = np.zeros((len(items), 3))                     # x wins, y wins, tie, averaged over both orders
    for (n, swap), p in zip(meta, probs):
        x_win, y_win = (p[1], p[0]) if swap else (p[0], p[1])
        res[n] += np.array([x_win, y_win, p[2]]) / 2
    return pd.DataFrame(res, columns=["x_better", "y_better", "tie"])


def context_attribution(model, tok, question, context_rows, answer):
    """Leave-one-out explanation: how much less likely the bot's answer becomes when each library item is
    removed from the prompt. A big drop means the answer relied on that item."""
    full = encode_example(tok, build_messages(question, context_rows), answer, max_len=4096)
    base = answer_nll(model, tok, [full], 1)[0]
    rows = []
    for k in range(len(context_rows)):
        rest = context_rows.drop(context_rows.index[k])
        ex = encode_example(tok, build_messages(question, rest), answer, max_len=4096)
        rows.append({"item": k + 1, "title": context_rows.iloc[k].title,
                     "nll_increase": answer_nll(model, tok, [ex], 1)[0] - base})
    ex = encode_example(tok, build_messages(question, None), answer, max_len=4096)
    rows.append({"item": "none", "title": "(no library items at all)", "nll_increase": answer_nll(model, tok, [ex], 1)[0] - base})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- the chat bot
class SupportBot:
    """Chat bot = language model (+ LoRA) + library search. respond() returns the answer and the sources used."""
    def __init__(self, model, tok, library=None, search=None, k=3, min_similarity=0.55):
        self.model, self.tok, self.library, self.search, self.k, self.min_sim = model, tok, library, search, k, min_similarity

    def find(self, query):
        if self.library is None or self.search is None:
            return None, None
        idx, _ = self.search.search([query], self.k)
        sim = float(getattr(self.search, "last_dense_sim", [1.0])[0])
        return self.library.iloc[idx[0]], sim

    def prompt(self, message, history=None, use_library=True):
        query = message if not history else history[-1][0] + "\n" + message      # follow-ups keep the topic
        rows, sim = self.find(query) if use_library else (None, None)
        return chat_prompt(self.tok, build_messages(message, rows, history)), rows, sim

    def respond(self, message, history=None, use_library=True, max_new_tokens=450, stream=False, **gen):
        import threading
        from transformers import TextIteratorStreamer
        prompt, rows, sim = self.prompt(message, history, use_library)
        if not stream:
            return generate(self.model, self.tok, [prompt], max_new_tokens, 1, **gen)[0], rows, sim
        device = next(self.model.parameters()).device
        enc = self.tok([prompt], return_tensors="pt", add_special_tokens=False).to(device)
        streamer = TextIteratorStreamer(self.tok, skip_prompt=True, skip_special_tokens=True)
        stop_ids = [self.tok.convert_tokens_to_ids("<|im_end|>"), self.tok.convert_tokens_to_ids("<|endoftext|>")]
        kw = dict(max_new_tokens=max_new_tokens, streamer=streamer, eos_token_id=stop_ids, pad_token_id=self.tok.pad_token_id,
                  repetition_penalty=gen.get("repetition_penalty", 1.05), do_sample=False)
        threading.Thread(target=self.model.generate, kwargs={**enc, **kw}, daemon=True).start()
        return streamer, rows, sim


def build_search(cfg, lib, emb, device=None):
    """Rebuild the search chosen in the notebook from the model card: 'bm25', 'semantic' or 'hybrid'."""
    dense = DenseSearch(cfg["embedding_model"], device=device).fit(emb=emb)
    if cfg["type"] == "semantic":
        return dense
    b = cfg["bm25"]
    bm25 = BM25Search(b["preprocessing"], k1=b["k1"], b=b["b"], min_df=2).fit(prepped=lib.prepped.tolist())
    if cfg["type"] == "bm25":
        return bm25
    return HybridSearch(bm25, dense, dense_weight=cfg["semantic_weight"])


def save_library(lib, emb, folder):
    """Library items + int8-quantised embeddings (4x smaller than float32, same ranking in practice)."""
    from pathlib import Path
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    lib.to_parquet(folder / "library.parquet", compression="zstd")
    scale = np.abs(emb).max(1, keepdims=True) / 127
    np.save(folder / "embeddings_int8.npy", np.round(emb / scale).astype(np.int8))
    np.save(folder / "embeddings_scale.npy", scale.astype(np.float32))


def load_library(folder):
    from pathlib import Path
    folder = Path(folder)
    lib = pd.read_parquet(folder / "library.parquet")
    emb = np.load(folder / "embeddings_int8.npy").astype(np.float32) * np.load(folder / "embeddings_scale.npy")
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    return lib, emb.astype(np.float32)


# ---------------------------------------------------------------- downloads
def download(url, path, chunk=1 << 20):
    """Download a file with a progress line every 10%."""
    import urllib.request
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "it-support-bot"}), timeout=120) as r, open(tmp, "wb") as f:
        total, done, mark, t0 = int(r.headers.get("Content-Length", 0)), 0, 0.1, time.time()
        while True:
            b = r.read(chunk)
            if not b:
                break
            f.write(b)
            done += len(b)
            if total and done / total >= mark:
                print(f"  {done / 1e6:,.0f} / {total / 1e6:,.0f} MB ({done / 1e6 / (time.time() - t0):.1f} MB/s)", flush=True)
                mark += 0.1
    tmp.rename(path)
    return path


def md5(path, chunk=1 << 22):
    import hashlib
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def extract_7z(archive, names, out):
    """Extract a few files from a .7z archive with 7z (Linux/Colab) or bsdtar (macOS)."""
    import shutil
    import subprocess
    if shutil.which("7z") is None and shutil.which("apt-get"):
        subprocess.run("(apt-get -qq update && apt-get -qq install -y p7zip-full) > /dev/null", shell=True, check=False)
    if shutil.which("7z"):
        subprocess.run(["7z", "x", str(archive), f"-o{out}", "-y", *names], check=True, stdout=subprocess.DEVNULL)
    else:
        subprocess.run(["tar", "-xf", str(archive), "-C", str(out), *names], check=True)

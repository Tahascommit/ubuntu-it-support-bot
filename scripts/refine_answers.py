# Rewrite trusted Ask Ubuntu answers into clear help-desk replies with a teacher model (Qwen3-4B-Instruct-2507).
# The rewrite keeps each answer's fix and commands; itbot.refine_ok() drops rewrites that lost them. The result is the
# fine-tuning data of ML Model 3 (data/refined_answers.jsonl.gz). Needs a GPU with about 24 GB (the dataset in the
# repo was made on an RTX 6000 Ada: 4,000 answers in about an hour).
#
#   python scripts/refine_answers.py --pool raw/pool.parquet --n 4000 --out data/refined_answers.jsonl.gz
#
# --pool: the notebook's fine-tuning pool (columns question_id, title, question, answer, year), e.g. saved with
# pool.to_parquet("raw/pool.parquet") after section 6 of the notebook.
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import itbot  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--pool", required=True)
ap.add_argument("--n", type=int, default=4000)
ap.add_argument("--teacher", default="Qwen/Qwen3-4B-Instruct-2507")
ap.add_argument("--out", default="data/refined_answers.jsonl.gz")
ap.add_argument("--seed", type=int, default=101)
args = ap.parse_args()

pool = pd.read_parquet(args.pool)
# the same number of threads from every year, so the data is not dominated by the busy years
per_year = int(np.ceil(args.n / pool.year.nunique()))
sub = (pool.groupby("year", group_keys=False).apply(lambda g: g.sample(min(len(g), per_year), random_state=args.seed))
       .sample(frac=1, random_state=args.seed).head(args.n))
teacher, tok = itbot.load_llm(args.teacher)
prompts = [itbot.refine_prompt(tok, itbot.format_question(t, b), a) for t, b, a in zip(sub.title, sub.question, sub.answer)]
lengths = np.array([len(p) for p in prompts])
long_ = lengths >= np.quantile(lengths, 0.9)          # the longest prompts in small batches, the rest in big ones
refined = [None] * len(prompts)
for mask, bs in ((long_, 16), (~long_, 48)):
    idx = np.where(mask)[0]
    for i, r in zip(idx, itbot.generate(teacher, tok, [prompts[i] for i in idx], max_new_tokens=450, batch_size=bs, log=print)):
        refined[i] = r
keep = [itbot.refine_ok(r, a) for r, a in zip(refined, sub.answer)]
out = pd.DataFrame({"question_id": sub.question_id.values, "refined": refined})[keep]
out.to_json(args.out, orient="records", lines=True, compression="gzip")
print(f"kept {len(out):,} of {len(sub):,} rewrites -> {args.out}")

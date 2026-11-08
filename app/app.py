# IT support bot for Ubuntu and Linux: the Gradio chat app.
# On a laptop, from the project folder:   python app/app.py          (opens http://127.0.0.1:7860)
# In the Colab notebook:                  chat_app.build_demo(bot).launch(share=True)
import argparse
import json
import re
import sys
import threading
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import itbot  # noqa: E402

EXAMPLES = [
    "My Wi-Fi stopped working after a kernel update. How can I get it back?",
    "How do I find out which program is using port 8080?",
    "apt update fails with 'The following signatures couldn't be verified because the public key is not available'",
    "How can I make a script run automatically every time Ubuntu starts?",
    "My disk is almost full. How do I find out which folders take the most space?",
    "How do I set a static IP address on Ubuntu Server 24.04?",
    "What is a good recipe for banana bread?",
]
NO_SOURCES = "_Sources appear here after the first answer._"
OFF_TOPIC = "_The question is outside Ubuntu and Linux support, so no sources are shown._"


def load_bot(device=None):
    """Build the bot from the saved files: base model + LoRA adapter, library, and the tuned search."""
    card = json.load(open(ROOT / "models" / "model_card.json"))
    device = device or itbot.get_device()
    model, tok = itbot.load_llm(card["base_model"], device, adapter=str(ROOT / card["adapter"]))
    lib, emb = itbot.load_library(ROOT / "models" / "library")
    search = itbot.build_search(card["search"], lib, emb, device)
    return itbot.SupportBot(model, tok, lib, search, k=card["search"]["items_shown"]), card


def sources_md(rows, sim, min_sim):
    """The library items the answer was built on, as clickable links."""
    if rows is None or len(rows) == 0:
        return "_The library was not used for this answer._"
    lines = []
    if sim is not None and sim < min_sim:
        lines.append("**Not sure:** no closely matching solved thread was found, so please double-check this answer.\n")
    lines.append("**Sources the bot was shown:**\n")
    for n, r in enumerate(rows.itertuples(), 1):
        kind = f"Ask Ubuntu thread, {r.year}" if r.kind == "thread" else "official Ubuntu documentation"
        lines.append(f"{n}. [{r.title}]({r.url}) ({kind})")
    return "\n".join(lines)


def as_text(content):
    """A message's content as plain text. Gradio 6 returns earlier messages as a list of parts ({"type": "text", ...})."""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return str(content.get("text") or "")
    if isinstance(content, (list, tuple)):
        return "\n".join(as_text(part) for part in content).strip()
    return "" if content is None else str(content)


def stream_answer(bot, message, use_library=True, max_new_tokens=450):
    """Answer one question on its own, as the model was trained and tested: the earlier turns of the chat are shown
    on screen but not sent to the model. Streams the text; stops early if the model starts repeating itself."""
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

    prompt, rows, sim = bot.prompt(message, None, use_library)
    enc = bot.tok([prompt], return_tensors="pt", add_special_tokens=False).to(next(bot.model.parameters()).device)
    start = enc["input_ids"].shape[1]

    class RepeatStop(StoppingCriteria):
        """Stop when the last 12 tokens already appeared earlier in the answer (the start of a loop)."""
        fired = False

        def __call__(self, input_ids, scores, **kw):
            gen = input_ids[0, start:].tolist()
            n = 12
            if len(gen) >= 2 * n:
                tail = gen[-n:]
                if any(gen[i:i + n] == tail for i in range(len(gen) - 2 * n + 1)):
                    RepeatStop.fired = True
            return torch.tensor([RepeatStop.fired], device=input_ids.device)

    streamer = TextIteratorStreamer(bot.tok, skip_prompt=True, skip_special_tokens=True)
    stop_ids = [bot.tok.convert_tokens_to_ids("<|im_end|>"), bot.tok.convert_tokens_to_ids("<|endoftext|>")]
    kw = dict(max_new_tokens=max_new_tokens, streamer=streamer, eos_token_id=stop_ids, pad_token_id=bot.tok.pad_token_id,
              repetition_penalty=1.05, do_sample=False, stopping_criteria=StoppingCriteriaList([RepeatStop()]))
    threading.Thread(target=bot.model.generate, kwargs={**enc, **kw}, daemon=True).start()
    return streamer, rows, sim, RepeatStop


def trim_repeats(text):
    """Cut an answer where it starts repeating a sentence it already wrote."""
    seen = set()
    for m in re.finditer(r"[^.!?\n]+[.!?]?", text):
        sentence = m.group().strip()
        if len(sentence) > 25:
            if sentence in seen:
                return text[:m.start()].rstrip()
            seen.add(sentence)
    return text


def to_pairs(history):
    """Gradio message list -> [(user, assistant), ...]"""
    pairs, user = [], None
    for m in history:
        if m["role"] == "user":
            user = as_text(m["content"])
        elif m["role"] == "assistant" and user is not None:
            pairs.append((user, as_text(m["content"])))
            user = None
    return pairs


def build_demo(bot, card=None):
    def chat(message, history, use_library):
        if not message.strip():
            yield history, gr.update(), ""
            return
        streamer, rows, sim, stopper = stream_answer(bot, message, use_library)
        history = history + [{"role": "user", "content": message}, {"role": "assistant", "content": ""}]
        src = sources_md(rows, sim, bot.min_sim) if use_library else "_The library was switched off._"
        text = ""
        for piece in streamer:
            text += piece
            history[-1]["content"] = text
            yield history, src, ""
        if stopper.fired:                   # the model began to loop: keep the answer up to the first repeat
            history[-1]["content"] = text = trim_repeats(text)
            yield history, src, ""
        if itbot.is_refusal(text):          # a declined question: the closest threads are not real sources
            yield history, OFF_TOPIC, ""

    def compare(message):
        if not message.strip():
            return "", "", ""
        if hasattr(bot.model, "disable_adapter"):
            with bot.model.disable_adapter():
                base, _, _ = bot.respond(message, use_library=False)
        else:
            base, _, _ = bot.respond(message, use_library=False)
        tuned, rows, sim = bot.respond(message)
        return base, tuned, OFF_TOPIC if itbot.is_refusal(tuned) else sources_md(rows, sim, bot.min_sim)

    about = ("An IT support assistant for Ubuntu and Linux, built for an industry LLM capstone. "
             "It is the pre-trained **Qwen3-1.7B-Base fine-tuned with LoRA** on trusted Ask Ubuntu answers (rewritten into clear "
             "help-desk replies), and it looks things up in a library of "
             "solved threads and the official Ubuntu guides before it answers (retrieval-augmented generation).")
    if card:
        ts = card.get("test_scores", {})
        about += (f"\n\n- Library: {card.get('library_items', 0):,} solved threads and guide passages"
                  f"\n- Fine-tuned on {card['lora']['training_examples']:,} examples (LoRA rank {card['lora']['r']})"
                  f"\n- Test scores against the community's answers: BERTScore {ts.get('bertscore', 0):.3f}, "
                  f"semantic similarity {ts.get('semantic', 0):.3f}, command recall {ts.get('command_recall', 0):.3f}")
    about += ("\n\nAnswers can be wrong. Read commands before you run them, especially ones that start with `sudo` "
              "or delete, format or partition anything.")

    with gr.Blocks(title="Ubuntu IT support bot") as demo:
        gr.Markdown("# Ubuntu IT support bot\nAsk about installing software, drivers, networking, the terminal or system errors.")
        with gr.Tab("Chat"):
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(height=520, label="IT support bot", buttons=["copy"])
                    msg = gr.Textbox(placeholder="Describe your Ubuntu or Linux problem...", label="Your question", lines=2)
                    with gr.Row():
                        send = gr.Button("Send", variant="primary")
                        clear = gr.Button("New chat")
                    use_lib = gr.Checkbox(True, label="Look up solved threads and the Ubuntu guides first")
                with gr.Column(scale=2):
                    sources = gr.Markdown(NO_SOURCES)
            gr.Examples(EXAMPLES, inputs=msg, label="Try one of these")
            msg.submit(chat, [msg, chatbot, use_lib], [chatbot, sources, msg])
            send.click(chat, [msg, chatbot, use_lib], [chatbot, sources, msg])
            clear.click(lambda: ([], NO_SOURCES, ""), None, [chatbot, sources, msg])
        with gr.Tab("Compare with the base model"):
            gr.Markdown("The same question answered by **the pre-trained model before fine-tuning** (left) and by **the fine-tuned bot with its library** (right).")
            q = gr.Textbox(label="Question", lines=2)
            go = gr.Button("Ask both", variant="primary")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Before fine-tuning (Qwen3-1.7B-Base)")
                    base_out = gr.Markdown()
                with gr.Column():
                    gr.Markdown("### Fine-tuned bot + library")
                    tuned_out = gr.Markdown()
            src2 = gr.Markdown()
            gr.Examples(EXAMPLES[:4], inputs=q)
            go.click(compare, q, [base_out, tuned_out, src2])
            q.submit(compare, q, [base_out, tuned_out, src2])
        with gr.Tab("About"):
            gr.Markdown(about)
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Ubuntu IT support bot")
    ap.add_argument("--device", default=None, help="cuda, mps or cpu (default: cuda if available, else cpu)")
    ap.add_argument("--share", action="store_true", help="also create a public gradio.live link")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    print("loading the model and the library ...", flush=True)
    bot, card = load_bot(args.device)
    build_demo(bot, card).launch(server_name="127.0.0.1", server_port=args.port, share=args.share)

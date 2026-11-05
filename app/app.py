# IT support bot for Ubuntu and Linux: the Gradio chat app.
# On a laptop, from the project folder:   python app/app.py          (opens http://127.0.0.1:7860)
# In the Colab notebook:                  chat_app.build_demo(bot).launch(share=True)
import argparse
import json
import sys
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


def to_pairs(history):
    """Gradio message list -> [(user, assistant), ...]"""
    pairs, user = [], None
    for m in history:
        if m["role"] == "user":
            user = m["content"]
        elif m["role"] == "assistant" and user is not None:
            pairs.append((user, m["content"]))
            user = None
    return pairs


def build_demo(bot, card=None):
    def chat(message, history, use_library):
        if not message.strip():
            yield history, gr.update(), ""
            return
        streamer, rows, sim = bot.respond(message, to_pairs(history), use_library, stream=True)
        history = history + [{"role": "user", "content": message}, {"role": "assistant", "content": ""}]
        src = sources_md(rows, sim, bot.min_sim) if use_library else "_The library was switched off._"
        text = ""
        for piece in streamer:
            text += piece
            history[-1]["content"] = text
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

# Ubuntu IT Support Bot: an industry-specific LLM fine-tuned on Ask Ubuntu

An IT support assistant for Ubuntu and Linux, built for the industry LLM capstone (industry: **Technology and IT**). It is the pre-trained **Qwen3-1.7B-Base** from Hugging Face, fine-tuned with **LoRA** on questions that Ubuntu users really asked and the answers the community trusted, rewritten by a larger teacher model into clear help-desk replies. Before it answers, it looks up similar solved threads and the official Ubuntu guides (**retrieval-augmented generation**), shows its sources, and politely declines questions that are not about IT.

The notebook runs on a free Google Colab T4 (about 3 hours). The results below come from the same notebook on a rented NVIDIA RTX 6000 Ada (vast.ai, about 70 minutes).

Everything is in one notebook that follows the course ML submission template:
**[IT_Support_LLM_Bot.ipynb](IT_Support_LLM_Bot.ipynb)**

## Repository layout

```
├── IT_Support_LLM_Bot.ipynb        the whole project, with the outputs of the final run
├── itbot.py                        shared code: cleaning, search, prompts, LoRA training, evaluation, the bot
├── app/app.py                      Gradio chat app (chat with sources, side-by-side comparison with the base model)
├── scripts/
│   ├── fetch_stackexchange_api.py  recent Ask Ubuntu questions, off-topic questions, monthly counts (Stack Exchange API)
│   ├── scrape_ubuntu_docs.py       the official Ubuntu Server / Desktop guides and LTS release notes (web scraping)
│   └── refine_answers.py           rewrites trusted answers into help-desk replies with a teacher model (needs a GPU)
├── data/                           what the scripts collected, incl. the 3,329 refined answers (the 1 GB dump is downloaded by the notebook)
├── models/
│   ├── lora_adapter/               the fine-tuned LoRA adapter (the base model comes from Hugging Face)
│   ├── library/                    the bot's library: solved threads + guide passages, with embeddings
│   └── model_card.json             settings, search parameters and test scores
├── outputs/                        every chart of the notebook and the result tables
└── requirements.txt
```

## Data

| Source | Collected with | Size | Licence |
|---|---|---|---|
| Ask Ubuntu data dump (April 2024) | public dataset, [archive.org](https://archive.org/details/stackexchange), checked against its MD5 | 414,451 questions, 523,728 answers, 42,035 duplicate links | CC BY-SA |
| Ask Ubuntu after the dump (Apr 2024 - Nov 2026) | Stack Exchange API | 2,000 top questions with answers, monthly counts | CC BY-SA |
| Off-topic questions | Stack Exchange API, 12 non-IT sites | 1,200 titles | CC BY-SA |
| Refined answers | `scripts/refine_answers.py`: Qwen3-4B-Instruct rewrites trusted answers, keeping their commands | 3,329 kept of 4,000 | derived from CC BY-SA answers |
| Official Ubuntu guides | web scraping, 1 page per second as robots.txt asks | 2,959 passages from 347 pages (383 fetched) | Desktop guide CC BY-SA 4.0; the Server guide states no licence, so every passage keeps its source link |

Cleaning keeps one trusted answer per question: the accepted answer if the community did not vote it down, otherwise the best answer with at least two up-votes. HTML is converted to text with code blocks kept, and e-mails, user names in paths and prompts, and @mentions are removed.

## Results

300 test questions (July 2023 to March 2024) that the model never saw, scored against the community's trusted answers. Ranges are 95% bootstrap intervals.

| | Search bot (tuned) | Qwen3-1.7B-Base as is | **Fine-tuned** |
|---|---|---|---|
| Command recall (the right commands and packages) | 0.182 | 0.177 (0.149-0.211) | **0.279** (0.247-0.313) |
| Semantic similarity | 0.730 | 0.700 (0.685-0.714) | **0.776** (0.768-0.785) |
| BERTScore | 0.818 | 0.787 | **0.829** |
| ROUGE-L | 0.115 | 0.096 | **0.144** |
| Off-topic questions declined | - | 1.3% | **100%** |
| Words per answer | 126 | 190 (rarely ends) | **108** |

Two LLM judges from different model families compared the answers in both orders (150 test questions). The fine-tuned bot with its library was preferred:

| Judge | vs pre-trained + library | vs pre-trained as is | vs search bot |
|---|---|---|---|
| Phi-4 (14B) | **72.7%** | **67.3%** | **68.0%** |
| Qwen3-4B-Instruct | **65.3%** | **66.7%** | **66.7%** |

The same gains hold on 200 questions asked after the data dump (April 2024 to November 2026): command recall 0.145 to 0.223, semantic similarity 0.673 to 0.765. An answer takes about 5 seconds on the GPU.



## Running it

**The whole notebook:** open it in Colab (or Jupyter on any GPU machine), choose **Runtime > Change runtime type > T4 GPU** and **Run all**. It downloads the data, trains, evaluates, downloads a zip with the trained adapter and the figures, and starts the chat app with a public link. On a free T4 this takes about 3 hours and needs a session that does not run out of RAM; a GPU with 24 GB or more is faster and safer.

**Only the chat app, on a free Colab T4** (it uses the trained adapter and library from this repository, no training):

```python
!git clone https://github.com/Tahascommit/ubuntu-it-support-bot
%cd ubuntu-it-support-bot
!pip install -q transformers==5.20.2 peft==0.21.2 accelerate==1.15.0 sentence-transformers==6.1.0 gradio==6.30.0 markdownify==1.2.3
!pip uninstall -y -q torchao      # Colab's old torchao stops peft from loading
!python app/app.py --share         # prints a public gradio.live link
```

**The chat app on a laptop** (Python 3.12):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app/app.py
```

It opens at http://127.0.0.1:7860. The base model (about 4 GB) is downloaded from Hugging Face the first time.

## Notes and limits

- Answers can be wrong. Read commands before running them, especially ones with `sudo` or ones that delete, format or partition.
- The community answers are from 2010 to 2026; the bot shows the year of every solved thread it used, so old advice can be spotted.
- The bot is for Ubuntu and Linux support only.

## Credits

Questions and answers: Ask Ubuntu contributors, from the Stack Exchange data dump and API, licensed CC BY-SA. Guides: Canonical's Ubuntu documentation. Base model: Qwen3-1.7B-Base by the Qwen team (Apache 2.0). Teacher and first judge: Qwen3-4B-Instruct-2507 (Apache 2.0). Second judge: Microsoft Phi-4 (MIT). Embeddings: BAAI bge-small-en-v1.5 (MIT).

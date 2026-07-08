# NSTT — Nepali Speech-to-Text

Fine-tune [OpenAI Whisper-Small](https://huggingface.co/openai/whisper-small) on Nepali speech corpora for academic experimentation (ST7088CEM Artificial Neural Networks).

## Repository layout

```
NSTT/
├── data/          # Raw and processed datasets (gitignored; sync via Google Drive)
├── notebooks/     # Colab notebooks (setup, data prep, training, eval)
├── src/           # Reusable Python modules
├── models/        # Local checkpoints (gitignored; primary store on Drive)
├── reports/       # WER/CER tables, plots, error analysis
├── dashboard/     # Streamlit report viewer (local)
└── .duo/          # Duo framework project state and handoffs
```

See [`.duo/project-context.md`](.duo/project-context.md) for architecture and roadmap.

## Quick start — Google Colab

### 1. Open the setup notebook

**Option A — Upload from this repo**

1. Clone or download this repository.
2. Go to [Google Colab](https://colab.research.google.com/).
3. **File → Upload notebook** and select `notebooks/00_setup.ipynb`.

**Option B — Open from GitHub** (after pushing this repo)

1. In Colab: **File → Open notebook → GitHub**.
2. Enter your repository URL and open `notebooks/00_setup.ipynb`.

**Option C — Copy from Google Drive**

1. Upload the whole `NSTT/` folder to Google Drive.
2. In Colab: **File → Open notebook → Google Drive** and open `NSTT/notebooks/00_setup.ipynb`.

### 2. Enable GPU

1. **Runtime → Change runtime type**.
2. Set **Hardware accelerator** to **GPU** (T4 on free tier).
3. Click **Save**.

### 3. Run all cells

Use **Runtime → Run all** (or run cells top to bottom). The notebook will:

- Mount Google Drive (`/content/drive`)
- Print GPU name and CUDA availability
- Install dependencies from `requirements.txt`
- Import `transformers`, `datasets`, `torchaudio`, `librosa`, `evaluate`, `jiwer`, and `accelerate`

If `torchaudio` fails to install, check the Colab runtime’s `torch` version and install a matching `torchaudio` build (see comment in `requirements.txt`).

### 4. Point the project at Drive (recommended)

After setup, keep data and checkpoints on Drive so sessions survive disconnects:

```python
PROJECT_ROOT = "/content/drive/MyDrive/NSTT"  # adjust to your Drive path
```

Later notebooks (T-002+) will read/write under `data/` and `models/` relative to that root.

## Local development (optional)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

GPU training is intended for Colab; local use is mainly for editing `src/` and notebooks.

## Dependencies

Pinned in [`requirements.txt`](requirements.txt): `transformers`, `datasets`, `accelerate`, `evaluate`, `jiwer`, `librosa`, `torchaudio`, `soundfile`, `matplotlib`, `streamlit`, `pandas`.

## Running the dashboard locally

After evaluation and error analysis have produced files under `reports/`:

```bash
source .venv/bin/activate   # if using a venv
pip install -r requirements.txt
streamlit run dashboard/app.py
```

The dashboard reads `reports/wer_cer_results.csv`, `reports/error_samples.csv`, `reports/error_categories.csv` (and `.png` if present), plus training curves from a checkpoint's `trainer_state.json` (default: `models/whisper-small-ne-smoke/checkpoint-3/`). Use the sidebar to point at a different checkpoint after a full Colab training run.

The **Live Demo** tab lets you record speech with your browser microphone (`st.audio_input`) and transcribe it with the checkpoint selected in the sidebar. Grant microphone permission when prompted. Transcription runs locally on CPU; quality will be poor until you train on the full SLR54 corpus in Colab.

**Note:** Current bundled reports reflect the smoke-checkpoint sanity check unless you regenerate them from a fully trained model.

## Notebook pipeline (Colab)

| Notebook | Purpose |
|----------|---------|
| `notebooks/00_setup.ipynb` | Environment setup, GPU check, dependencies |
| `notebooks/01_data_prep.ipynb` | SLR54 download, preprocess, manifests |
| `notebooks/02_finetune_whisper.ipynb` | Whisper-Small fine-tuning |
| `notebooks/03_evaluate.ipynb` | WER/CER evaluation (in-domain + FLEURS) |
| `notebooks/04_error_analysis.ipynb` | Error analysis and report artifacts |

## License & attribution

Academic integrity: attribute any adapted public code with a comment and citation (see `start.md` §7).

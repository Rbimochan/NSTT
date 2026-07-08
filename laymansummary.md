# NSTT — What This Project Is, In Plain English

## The idea
Teach a computer to listen to spoken Nepali and write down what it hears — a
"speech-to-text" system, the same kind of technology behind voice assistants
and auto-captioning, but built specifically for Nepali. This is a university
project (ST7088CEM, Artificial Neural Networks, Softwarica College / Coventry
University), so the goal is both a working demo and a written report proving
the work and results.

## How it works, without the jargon
1. **Start with an existing "smart listener".** OpenAI already built a
   general-purpose speech-recognition model called Whisper. It understands
   many languages reasonably but isn't great at Nepali specifically.
2. **Show it lots of Nepali speech.** We use a public dataset of Nepali
   recordings with matching written transcripts (~157,000 short clips,
   roughly 165 hours) so the model can learn Nepali's sounds and script.
3. **Fine-tune, don't rebuild.** Instead of training a model from zero (which
   would need huge computing power), we take Whisper and retrain it just
   enough on Nepali data — cheaper and faster, using free Google Colab GPUs.
4. **Grade its homework.** After training, we test it on speech it has never
   heard before and measure how many words/characters it gets wrong (called
   WER/CER — Word/Character Error Rate). Lower is better.
5. **Explain the mistakes.** Beyond a single score, we look at *what kinds*
   of errors it makes (confusing similar-sounding letters, unfamiliar words,
   mixed English-Nepali speech, etc.) — this is the "error analysis" that
   makes the report more than just a number.
6. **Show it off.** A simple web dashboard lets anyone view the results, and
   even record their own voice and see it transcribed live.

## What's been built so far
- ✅ The full data pipeline: downloading, cleaning, and splitting the Nepali
  speech dataset.
- ✅ The training pipeline: code that fine-tunes Whisper on Colab's free GPU.
- ✅ The grading pipeline: automatic WER/CER scoring on both familiar and
  unfamiliar Nepali speech.
- ✅ The error-analysis pipeline: charts and tables explaining mistake
  patterns.
- ✅ A full dry-run ("smoke test") proving every piece connects correctly
  end-to-end, using a tiny 3-step training run just to check the plumbing —
  not real results.
- ✅ A dashboard (web app) to browse results and reports.
- ✅ A "live demo" tab where you can talk into your microphone and see it
  transcribed on the spot.

## What's left
The one remaining step is **the real training run**: letting the model
train properly (2,000 steps instead of 3) on Google Colab's GPU, using the
*full* dataset instead of a tiny sample. This takes real GPU time, so it's a
manual step the human runs, not something automated. Once that's done and
the real scores come in, the last task is to swap the placeholder ("smoke
test") numbers in the report for the real ones and write up a final results
summary.

## Bottom line
The scaffolding, pipeline, and tooling are 100% built and tested — 8 of 9
planned tasks are complete. What remains isn't more coding; it's running the
actual training job and reporting the real numbers it produces.

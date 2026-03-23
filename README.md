# cosmic-ai

Fine-tuned CodeBERT for automated COSMIC Functional Size Measurement.

COSMIC FSM (ISO/IEC 19761) measures software size by counting data movements in functional processes: entries, exits, reads, and writes. The problem with manual measurement is it takes hours and needs a certified expert. This project automates it by fine-tuning CodeBERT to predict E, X, R, W values per line of source code.

Validated against expert measurements from two peer-reviewed papers: 96% accuracy on C programs, 88% on Arduino/IoT code.

---

## How it works

Each line of source code is tokenized and passed through the fine-tuned CodeBERT encoder. The [CLS] token goes into a 4-output regression head, one output per COSMIC movement type. Total CFP = E + X + R + W.

Trained for 5 epochs on an annotated dataset of C, Arduino, Python, and Java code labeled with expert COSMIC counts.

---

## Performance

| Metric | Total CFP |
|--------|-----------|
| MSE    | 0.095     |
| MAE    | 0.138     |
| R²     | 0.887     |

Validation:
- Koulla et al. (2022): C program, expert 45 CFP, predicted 43 (96%)
- Soubra & Abran (2017): Arduino app, expert 17 CFP, predicted 15 (88%)

---

## Getting started
```bash
git clone https://github.com/AhmedEl-Gohary/cosmic-ai.git
cd cosmic-ai
pip install -r requirements.txt
cd src && python app.py
```

Open `http://localhost:5000`, upload a source file, get line-by-line CFP predictions.

---

## CLI
```bash
python calculate_cfp.py -f path/to/code.c
```

---

## Training on your own data

Prepare a CSV with columns `code, E, X, R, W, CFP`, then run:
```bash
python finetune.py
```

Best model saves to `codebert-cfp/best-model/`.

---

## Project structure
```
cosmic-ai/
├── Paper/
│   └── Ahmed_El_Gohary_Bachelor_Thesis.pdf
├── src/
│   ├── app.py               # Flask backend and REST API
│   ├── calculate_cfp.py     # CLI tool
│   ├── finetune.py          # Fine-tuning script
│   ├── cosmic_analyzer.py   # C and Arduino analyzer with CSV export
│   └── c_cfp.py / arduino_cfp.py
├── templates/
│   └── index.html
└── test/
    ├── code.c
    ├── code.ino
    └── cosmic_dataset.csv
```

---

## Stack

CodeBERT (microsoft/codebert-base), PyTorch, Hugging Face Transformers, Flask, pandas, scikit-learn

---

## Research

Bachelor's thesis at the German University in Cairo, supervised by Dr. Milad Ghantous and Dr. Hassan Soubra.

Full paper: [Ahmed_El_Gohary_Bachelor_Thesis.pdf](Paper/Ahmed_El_Gohary_Bachelor_Thesis.pdf)

---

## License

MIT

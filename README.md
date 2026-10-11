# Hindi & Tamil Extractive Question Answering (XLM-RoBERTa)

Fine-tuning a multilingual transformer (`xlm-roberta-base`) to answer questions in **Hindi** and **Tamil** by locating the answer span inside a given passage. Built on the Kaggle [chaii - Hindi and Tamil Question Answering](https://www.kaggle.com/competitions/chaii-hindi-and-tamil-question-answering) dataset.

## Overview

Extractive QA does not generate new text. Given a **context** and a **question**, the model predicts the **start** and **end** token of the answer within the context.

**Example**

| | |
|---|---|
| Context | The Taj Mahal was built by Shah Jahan in Agra. |
| Question | Who built the Taj Mahal? |
| Predicted span | **Shah Jahan** |

## Workflow

1. **Load data**: read `train.csv` and `test.csv` (columns: `context`, `question`, `answer_text`, `answer_start`).
2. **Tokenize**: encode context and question together (max length 512) with offset mappings.
3. **Align labels**: convert character-level answer positions into token-level `start_positions` / `end_positions`.
4. **Split**: 80% train / 20% validation (`random_state=42`).
5. **Fine-tune**: `AutoModelForQuestionAnswering` with Hugging Face `Trainer`.
6. **Sanity check**: run the model on a validation example and compare predicted vs. true start/end positions.

## Training Configuration

| Parameter | Value |
|---|---|
| Base model | `FacebookAI/xlm-roberta-base` |
| Learning rate | 2e-5 |
| Batch size | 8 (train and eval) |
| Epochs | 2 |
| Weight decay | 0.01 |
| Evaluation | every 500 steps, best model loaded at end |

## Requirements

- Python 3.8+
- PyTorch
- transformers
- datasets
- scikit-learn
- pandas, numpy, tqdm
- A GPU is recommended (the notebook was written for Kaggle)

```bash
pip install torch transformers datasets scikit-learn pandas numpy tqdm
```

## Usage

1. Download the chaii dataset from Kaggle (or run the notebook on Kaggle, where the paths below already work):
   ```
   /kaggle/input/chaii-hindi-and-tamil-question-answering/train.csv
   /kaggle/input/chaii-hindi-and-tamil-question-answering/test.csv
   ```
2. Update the file paths in the notebook if running locally.
3. Run all cells in `learning-extractive-qa-with-chaii-dataset.ipynb`.

## Project Structure

```
.
├── learning-extractive-qa-with-chaii-dataset.ipynb   # Main notebook
└── README.md
```

## Limitations

- Contexts longer than 512 tokens are truncated rather than split into overlapping windows.
- Label alignment needs exact token-boundary matches, so some examples may get `-1` labels.
- No evaluation metric (e.g. word-level Jaccard) is implemented.
- Does not convert predicted positions into answer text or generate test-set predictions/submission.

## Possible Improvements

- Use a sliding window (`stride` / `return_overflowing_tokens`) for long contexts.
- Add post-processing to decode predicted spans into answer text.
- Implement the Jaccard score for evaluation.
- Generate predictions on `test.csv` for submission.
- Try larger models (e.g. `xlm-roberta-large`, MuRIL, or InfoXLM).

## Acknowledgements

- Dataset: [chaii competition](https://www.kaggle.com/competitions/chaii-hindi-and-tamil-question-answering) by Google Research India
- Model: [XLM-RoBERTa](https://huggingface.co/FacebookAI/xlm-roberta-base)
- Libraries: Hugging Face Transformers and Datasets

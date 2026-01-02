# Professor-Review-Rating-Predictor

BERT-Based Professor Rating Prediction

Fine-tuned a pretrained transformer model to predict 1–5 star professor ratings from student-written reviews. Framed as an ordinal NLP classification task and evaluated using accuracy and mean absolute error (MAE).

Summary

Built an end-to-end NLP pipeline to map free-text course reviews to star ratings

Addressed class imbalance and ordinal label structure

Achieved strong performance with low average prediction error

Dataset

Source: PlanetTerp professor reviews

Size: ~900 cleaned reviews

Labels: 1–5 star ratings

Preprocessing

Removed empty and short reviews

Filtered invalid ratings

Balanced classes via upsampling

Stratified train / validation / test split

Model

Architecture: BERT-based transformer

Pretrained model: nlptown/bert-base-multilingual-uncased-sentiment

Task: 5-class sequence classification

Tokenizer: Hugging Face AutoTokenizer

The model was selected because it is pretrained on sentiment-style data and aligns naturally with a 5-star rating scale.

Training

Loss: Weighted cross-entropy

Optimizer: AdamW

Learning rate: 2e-5

Weight decay: 0.015

Label smoothing: 0.1

Epochs: 3

Early stopping: Enabled

Training implemented using Hugging Face’s Trainer API with a custom weighted loss function.

Evaluation

Metrics reported:

Accuracy: Exact star match

MAE (Mean Absolute Error): Average star distance

Results

Accuracy: ~84%

MAE: ~0.18–0.22 stars

An MAE of ~0.22 means predictions are, on average, within about a quarter of a star of the true rating.

Analysis

Most errors are within ±1 star

Very few large-magnitude misclassifications

MAE provides a more informative signal than accuracy for ordinal labels

Tech Stack

Python

PyTorch

Hugging Face Transformers

scikit-learn

Pandas / NumPy

Key Takeaways

Transformer models can effectively model ordinal sentiment tasks

MAE is critical for evaluating ordered classification problems

Class balancing and weighted loss materially improve performance

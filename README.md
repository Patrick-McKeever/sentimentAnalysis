# Untitled Sentiment Analysis Project

A small sentiment analysis project trained on the IMDB review dataset using BERT.

## Data

Based on the well-known IMDB review dataset. Although the files were too large to upload to github (exceed 25mb upper cap), they are publicly available in multiple locations after a cursory google.

## Results

The "base model" (logistic regression) achieves the following accuracy scores:

```
              precision    recall  f1-score   support

         neg       0.82      0.84      0.83       246
         pos       0.84      0.83      0.83       254

    accuracy                           0.83       500
   macro avg       0.83      0.83      0.83       500
weighted avg       0.83      0.83      0.83       500
```

The latter model, using pre-trained BERT network (base model only used BERT for tokenization), produces the following results:

```
              precision    recall  f1-score   support

       False       0.91      0.90      0.91       246
        True       0.91      0.91      0.91       254

    accuracy                           0.91       500
   macro avg       0.91      0.91      0.91       500
weighted avg       0.91      0.91      0.91       500
```

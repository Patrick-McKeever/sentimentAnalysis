# Untitled Sentiment Analysis Project

A small sentiment analysis project trained on the IMDB review dataset using BERT.

## Data

Based on the well-known IMDB review dataset, available in the `data` directory.

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

The latter model, using pre-trained BERT network (base model only used BERT for tokenization), produces the following results given a training sample of n = 2000 and 5 epochs worth of training:

```
              precision    recall  f1-score   support

       False       0.91      0.90      0.91       246
        True       0.91      0.91      0.91       254

    accuracy                           0.91       500
   macro avg       0.91      0.91      0.91       500
weighted avg       0.91      0.91      0.91       500
```

## Models

The aforementioned models are available in pickle form. `logReg.sav` refers to the logistic regression model, while `finalModel.sav` refers to the BERT model.

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

The latter model, using pre-trained BERT network (base model only used BERT for tokenization), remains a work in progress. Code should compile, but Debian's lack of drivers for my GPU means I have to use CPU, prompting memory overflows. At the moment, I am unable to verify effectiveness (code never progresses past epoch loop on line 131) or assess accuracy for the latter half of the script.

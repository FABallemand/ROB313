# Practical Work: Fire Detection

# Unsupervised and Self-supervised Learning

## Instructions
Instructions for the practical work are detailed in `instructions.md`.

## Descriptions
Classical approaches for image classification relies on deep neural networks trained on large datasets of images and labels, a technique called Supervised Learning (SL). Eventhough these methods have proven their worth for many image classification tasks, their supervised nature prevent them from being used on tasks where there is few labelled training datas. Image labelisation is often time consuming and costly. For instance, simple tasks like object recognition require a lot of training samples to account for the variety of objects. Such data extensive methods sometimes requires experts to labelize data. It should also be noted that in both cases labelisation can be incorrect and negatively impact the training of the network. This is without mentioning the ethical issues hidden behind low income labelisation labour.

In this work, we try to reduce the use of labelised data by exploring unsupervised learning and Self-Supervised Learning (SSL) techniques that can leverage hidden knowledge contained in images without labels. The goal of this study is to assess the performance gain of SSL on downstream tasks that have large amount of unlabeled data available but few labeled data. We apply this technique to wildfire detection through satellite images as analysing satellite images is a domain where SL would require experts to annotate large databases of images to the best of their ability. A process that end up being costly, time consuming and unreliable. First, we introduce the dataset and do a quick overview of self-supervised techniques. Then, we experiment four different methods based on unsupervised and self-supervised learning, namely Gaussian Mixture Models (GMM), Auto-Encoders (AE), Variational Auto-Encoder (VAE) and the SimCLR SSL framework

## Repository Organisation
```
practical_work/
    ├── Project-Computer-Vision/ # VGG-16 Auto-Encoder and GMM
    ├── SimCLR/                  # SimCLR
    ├── instructions.md          # Project instructions
    ├── README.md                # README
    └── report.pdf               # Project report
```

## Authors
- ALLEMAND Fabien (Télécom SudParis)
- LAMURRI Pierre (ENSTA)
- PETITCOLIN Corentin (ENSTA)
- RIGAUD Charles (ENSTA)
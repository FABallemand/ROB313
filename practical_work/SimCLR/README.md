# Practical Work: Fire Detection

# SimCLR

## Descriptions
In this work, we try to reduce the use of labelised data by exploring Self-Supervised Learning (SSL) techniques that can leverage hidden knowledge contained in images without labels. The goal of this study is to assess the performance gain of SSL on downstream tasks that have large amount of unlabeled data available but few labeled data. We apply this technique to wildfire detection through satellite images as analysing satellite images is a domain where SL would require experts to annotate large databases of images to the best of their ability. A process that end up being costly, time consuming and unreliable. Our work is based on the SimCLR SSL framework.

## Related Links
- [Wildfire Prediction Dataset](https://www.kaggle.com/datasets/abdelghaniaaba/wildfire-prediction-dataset/data)
- [Self-supervised learning](https://en.wikipedia.org/wiki/Self-supervised_learning)
- [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)
- [NT-Xent (Normalized Temperature-Scaled Cross-Entropy) Loss Explained and Implemented in PyTorch](https://towardsdatascience.com/nt-xent-normalized-temperature-scaled-cross-entropy-loss-explained-and-implemented-in-pytorch-cc081f69848/)
- [gradcam_plus_plus-pytorch](https://github.com/vickyliin/gradcam_plus_plus-pytorch)
- [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam/)

## Repository Organisation
```
SimCLR/
    ├── report/              # Project report
    ├── test_res/            # Testing results
    ├── train_res/           # Training results (model weights)
    ├── dataset.py           # Python code for custom datasets
    ├── losses.py            # Python code for custom loss functions
    ├── models.py            # Python code for custom models
    ├── practical_work.ipynb # Test notebook
    ├── README.md            # README
    ├── test.py              # Python code for testing models
    ├── test.sh              # Slurm script for testing models on Télécom Paris GPU cluster
    ├── train_sl.py          # Python code for training models with supervised learning
    ├── train_sl.sh          # Slurm script for training models with supervised learning on Télécom Paris GPU cluster
    ├── train_ssl.py         # Python code for training models with self-supervised learning
    ├── train_ssl.sh         # Slurm script for training models with self-supervised learning on Télécom Paris GPU cluster
    └── transformations.py   # Python code for custom loss transformations
```

## Experiments Settings

### ResNet
```json
- {'job_id': '285136', 'model': 'ResNet', 'epochs': 100, 'batch_size': 128, 'learning_rate': 0.0001, 'lmbda': 0.5, 'temperature': 0.1, 'device': 'cuda', 'save_path': 'train_res/285136'}
- {'job_id': '285444', 'model': 'ResNet', 'encoder_id': 285136, 'encoder_freezing': True, 'epochs': 100, 'batch_size': 128, 'learning_rate': 0.0001, 'device': 'cuda', 'save_path': 'train_res/285444'}
- {'job_id': '285460', 'model': 'ResNet', 'encoder_id': None, 'encoder_freezing': True, 'epochs': 100, 'batch_size': 128, 'learning_rate': 0.0001, 'device': 'cuda', 'save_path': 'train_res/285460'}
- {'job_id': '287271', 'max_valid_2': 0.5, 'model': 'ResNet', 'encoder_id': 285136, 'encoder_freezing': True, 'epochs': 50, 'batch_size': 128, 'learning_rate': 0.0001, 'device': 'cuda', 'save_path': 'train_res/287271'}
- {'job_id': '287649', 'max_valid_2': 0.5, 'model': 'ResNet', 'encoder_id': None, 'encoder_freezing': True, 'epochs': 50, 'batch_size': 128, 'learning_rate': 0.0001, 'device': 'cuda', 'save_path': 'train_res/287649'}
- {'job_id': '287990', 'max_valid_2': 0.25, 'model': 'ResNet', 'encoder_id': 285136, 'encoder_freezing': True, 'epochs': 50, 'batch_size': 128, 'learning_rate': 0.0001, 'device': 'cuda', 'save_path': 'train_res/287990'}
- {'job_id': '287992', 'max_valid_2': 0.25, 'model': 'ResNet', 'encoder_id': None, 'encoder_freezing': True, 'epochs': 50, 'batch_size': 128, 'learning_rate': 0.0001, 'device': 'cuda', 'save_path': 'train_res/287992'}
- {'job_id': '288475', 'max_valid_2': 0.1, 'model': 'ResNet', 'encoder_id': 285136, 'encoder_freezing': True, 'epochs': 50, 'batch_size': 128, 'learning_rate': 0.0001, 'device': 'cuda', 'save_path': 'train_res/288475'}
- {'job_id': '288476', 'max_valid_2': 0.1, 'model': 'ResNet', 'encoder_id': None, 'encoder_freezing': True, 'epochs': 50, 'batch_size': 128, 'learning_rate': 0.0001, 'device': 'cuda', 'save_path': 'train_res/288476'}
- {'job_id': '288502', 'max_valid_2': 0.05, 'model': 'ResNet', 'encoder_id': 285136, 'encoder_freezing': True, 'epochs': 50, 'batch_size': 128, 'learning_rate': 0.0001, 'device': 'cuda', 'save_path': 'train_res/288502'}
- {'job_id': '288503', 'max_valid_2': 0.05, 'model': 'ResNet', 'encoder_id': None, 'encoder_freezing': True, 'epochs': 50, 'batch_size': 128, 'learning_rate': 0.0001, 'device': 'cuda', 'save_path': 'train_res/288503'}

- {'job_id': '287371', 'max_valid_2': 0.5, 'model': 'ResNet', 'encoder_id': None, 'encoder_freezing': True, 'epochs': 50, 'batch_size': 128, 'learning_rate': 0.0001, 'device': 'cuda', 'save_path': 'train_res/287371'}
```

### UNet
```json
- {'job_id': '285138', 'model': 'UNet', 'epochs': 100, 'batch_size': 16, 'learning_rate': 0.0001, 'lmbda': 0.5, 'temperature': 0.1, 'device': 'cuda', 'save_path': 'train_res/285138'}
- {'job_id': '286611', 'model': 'UNet', 'encoder_id': 285138, 'encoder_freezing': True, 'epochs': 100, 'batch_size': 16, 'learning_rate': 0.0001, 'device': 'cuda', 'save_path': 'train_res/286611'}
- {'job_id': '286618', 'model': 'UNet', 'encoder_id': None, 'encoder_freezing': True, 'epochs': 100, 'batch_size': 16, 'learning_rate': 0.0001, 'device': 'cuda', 'save_path': 'train_res/286618'}
- {'job_id': '286619', 'model': 'UNet', 'encoder_id': None, 'encoder_freezing': False, 'epochs': 100, 'batch_size': 16, 'learning_rate': 0.0001, 'device': 'cuda', 'save_path': 'train_res/286619'}
- {'job_id': '288494', 'max_valid_2': 0.1, 'model': 'UNet', 'encoder_id': 285138, 'encoder_freezing': True, 'epochs': 50, 'batch_size': 16, 'learning_rate': 0.0001, 'device': 'cuda', 'save_path': 'train_res/288494'}
- {'job_id': '288497', 'max_valid_2': 0.1, 'model': 'UNet', 'encoder_id': None, 'encoder_freezing': False, 'epochs': 50, 'batch_size': 16, 'learning_rate': 0.0001, 'device': 'cuda', 'save_path': 'train_res/288497'}
```

## Author
- Fabien ALLEMAND
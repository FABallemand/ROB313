# Practical Work: Fire Detection

## Instructions
Instructions for the practical work are detailed in `instructions.md`.

## Descriptions

## Related Links
- [Wildfire Prediction Dataset](https://www.kaggle.com/datasets/abdelghaniaaba/wildfire-prediction-dataset/data)
- [Self-supervised learning](https://en.wikipedia.org/wiki/Self-supervised_learning)
- [A Simple Framework for Contrastive Learning of Visual Representations](https://arxiv.org/abs/2002.05709)
- [NT-Xent (Normalized Temperature-Scaled Cross-Entropy) Loss Explained and Implemented in PyTorch](https://towardsdatascience.com/nt-xent-normalized-temperature-scaled-cross-entropy-loss-explained-and-implemented-in-pytorch-cc081f69848/)

## Notes

### ResNet
- {'job_id': '285136', 'model': 'ResNet', 'epochs': 100, 'batch_size': 128, 'learning_rate': 0.0001, 'lmbda': 0.5, 'temperature': 0.1, 'device': 'cuda', 'save_path': 'train_res/285136'}
- {'job_id': '285444', 'model': 'ResNet', 'encoder_id': 285136, 'encoder_freezing': True, 'epochs': 100, 'batch_size': 128, 'learning_rate': 0.0001, 'device': 'cuda', 'save_path': 'train_res/285444'}
- {'job_id': '285460', 'model': 'ResNet', 'encoder_id': None, 'encoder_freezing': True, 'epochs': 100, 'batch_size': 128, 'learning_rate': 0.0001, 'device': 'cuda', 'save_path': 'train_res/285460'}

### UNet
- {'job_id': '285138', 'model': 'UNet', 'epochs': 100, 'batch_size': 16, 'learning_rate': 0.0001, 'lmbda': 0.5, 'temperature': 0.1, 'device': 'cuda', 'save_path': 'train_res/285138'}
- {'job_id': '286611', 'model': 'UNet', 'encoder_id': 285138, 'encoder_freezing': True, 'epochs': 100, 'batch_size': 16, 'learning_rate': 0.0001, 'device': 'cuda', 'save_path': 'train_res/286611'}
- {'job_id': '286618', 'model': 'UNet', 'encoder_id': None, 'encoder_freezing': True, 'epochs': 100, 'batch_size': 16, 'learning_rate': 0.0001, 'device': 'cuda', 'save_path': 'train_res/286618'}
- {'job_id': '286619', 'model': 'UNet', 'encoder_id': None, 'encoder_freezing': False, 'epochs': 100, 'batch_size': 16, 'learning_rate': 0.0001, 'device': 'cuda', 'save_path': 'train_res/286619'}

## Author
- Fabien ALLEMAND
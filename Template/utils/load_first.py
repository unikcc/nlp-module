
from loaders.textcnn_loader import MyDataLoader as TextCNNLoader 

from models.textCNN import TextCNN

from trainer.textcnn_trainer import textCNNTrainer

def get_model_loader(model_name):
    # model, loader trainer
    coll = {
        'textcnn': [TextCNN, TextCNNLoader, textCNNTrainer],
    }
    return coll[model_name]
    # models = {'textcnn': TextCNN, 'mtl': MTLBertClassification, 'fasttext': FastTextModel}
    # loaders = {'textcnn': TextCNNLoader, 'bert': BertLoader, 'mtl': MTLLoader, 'fasttext': FastTextLoader}
    # trainers = {'textcnn': textCNNTrainer, 'bert': STLBertTrainer, 'mtl': MTLBertTrainer, 'fasttext': FastTextTrainer}
    # return models[model_name], loaders[model_name], trainers[model_name]
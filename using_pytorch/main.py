import pandas as pd 
import os  
import torch 
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn
from torch.optim import Adam

import optuna

import argparse
import ignite
from ignite.engine import create_supervised_evaluator, create_supervised_trainer, Events
from ignite.metrics import Accuracy

import dataset 
import model_torch  

train_data_path = "/home/hasan/Data Set/covid19/COVID-19 Radiography Database/train"
valid_data_path = "/home/hasan/Data Set/covid19/COVID-19 Radiography Database/test"
cls_name = ['COVID','NORMAL','Viral Pneumonia']


def path_label(data_path, class_name=cls_name):
    feature_label = {'path': [], 'label': []}
    
    for cls_n in class_name:
        path = os.path.join(data_path, cls_n)
        cls_l = class_name.index(cls_n) 
        for img in os.listdir(path):
            img = os.path.join(path, img)

            feature_label['path'].append(img)
            feature_label['label'].append(cls_l)
            
    
    return pd.DataFrame(feature_label)

train_data = path_label(train_data_path, class_name=cls_name)
valid_data = path_label(valid_data_path, class_name=cls_name)

train_data = train_data.sample(frac=1).reset_index(drop=True)
valid_data = valid_data.sample(frac=1).reset_index(drop=True)


def get_data_loaders():
    train_set = dataset.custom_dataset(
                    image = train_data.path.values,
                    label = train_data.label.values,
                    train_data_aug = True,
                    )
    valid_set = dataset.custom_dataset(
                    image = valid_data.path.values,
                    label = valid_data.label.values,
                    train_data_aug = False    
                    )  
    
    
    train_loader = DataLoader(
                    train_set,
                    batch_size = 32, 
                    shuffle = True,
                    num_workers = 8
                    )
    
    valid_loader = DataLoader(
                    valid_set,
                    batch_size = 32,
                    shuffle = False,
                    num_workers = 8
                    )
    
    return train_loader, valid_loader


####################################################################
# Objective Function
####################################################################
def objective(trial):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model_torch.Net() 
    model.to(device)

    optimizer = Adam(model.parameters())
    loss_fn = nn.MSELoss()

    trainer = create_supervised_trainer(model, optimizer, loss_fn, device=device)
    evaluator = create_supervised_evaluator(model, metrics={"accuracy": Accuracy()}, device=device)

    # Register a pruning handler to the evaluator.
    pruning_handler = optuna.integration.PyTorchIgnitePruningHandler(trial, "accuracy", trainer)
    evaluator.add_event_handler(Events.COMPLETED, pruning_handler)

    # Loading dataset 
    train_loader, val_loader = get_data_loaders()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_results(engine):
        evaluator.run(val_loader)
        validation_acc = evaluator.state.metrics["accuracy"]
        print("Epoch: {} Validation accuracy: {:.2f}".format(engine.state.epoch, validation_acc))

    trainer.run(train_loader, max_epochs=10)

    evaluator.run(val_loader)
    return evaluator.state.metrics["accuracy"]



parser = argparse.ArgumentParser(description="PyTorch Ignite example.")
parser.add_argument(
        "--pruning",
        "-p",
        action="store_true",
        help="Activate the pruning feature. `MedianPruner` stops unpromising "
        "trials at the early stages of training.",
    )
args = parser.parse_args()
pruner = optuna.pruners.MedianPruner() if args.pruning else optuna.pruners.NopPruner()

study = optuna.create_study(pruner=pruner)
study.optimize(objective, n_trials=5, timeout=600)



print("Number of finished trials: ", len(study.trials))

print("Best trial:")
trial = study.best_trial

print("  Value: ", trial.value)

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))

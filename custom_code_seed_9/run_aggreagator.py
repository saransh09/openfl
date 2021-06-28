import torch
import torch.nn as nn
import torch.optim as optim
from openfl.interface.interactive_api.experiment import TaskInterface, DataInterface, ModelInterface, FLExperiment
from copy import deepcopy
from model_ import UNet, soft_dice_loss, soft_dice_coef
from dataloader_ import FedDataset, KvasirDataset
import numpy as np
import argparse

parser = argparse.ArgumentParser()
## The seed for which the dataset split is required
parser.add_argument('-seed', '--seed', metavar='seed', required=True, help='The seed for the shuffling of dataset')
## The number of epochs for which you want to run the training
parser.add_argument('-epochs', '--epochs', metavar='epochs', default=10)
args = parser.parse_args()

SEED = int(args.seed)
EPOCHS = int(args.epochs)

"""
UNet model definition
"""
# Initialize the model
model_unet = UNet()
# Initialize the optimizer that will be used in the model
optimizer_adam = optim.Adam(model_unet.parameters(), lr=1e-4)

# Let's register the tasks
framework_adapter = 'openfl.plugins.frameworks_adapters.pytorch_adapter.FrameworkAdapterPlugin'
MI = ModelInterface(model=model_unet, optimizer=optimizer_adam, framework_plugin=framework_adapter)

# Save the initial model state
initial_model = deepcopy(model_unet)

# Make the dataset
fed_dataset = FedDataset(KvasirDataset, seed=SEED, train_bs=8, valid_bs=8)


TI = TaskInterface()
import tqdm
# Task interface currently supports only standalone functions.

@TI.add_kwargs(**{'some_parameter': 42})
@TI.register_fl_task(model='unet_model', data_loader='train_loader', \
                     device='device', optimizer='optimizer')
def train(unet_model, train_loader, optimizer, device, loss_fn=soft_dice_loss, some_parameter=None):
    if not torch.cuda.is_available():
        device = 'cpu'

    train_loader = tqdm.tqdm(train_loader, desc="train")

    unet_model.train()
    unet_model.to(device)

    losses = []

    for data, target in train_loader:
        data, target = torch.tensor(data).to(device), torch.tensor(
            target).to(device, dtype=torch.float32)
        optimizer.zero_grad()
        output = unet_model(data)
        loss = loss_fn(output=output, target=target)
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().cpu().numpy())

    return {'train_loss': np.mean(losses),}


@TI.register_fl_task(model='unet_model', data_loader='val_loader', device='device')     
def validate(unet_model, val_loader, device):
    unet_model.eval()
    unet_model.to(device)

    val_loader = tqdm.tqdm(val_loader, desc="validate")

    val_score = 0
    total_samples = 0

    with torch.no_grad():
        for data, target in val_loader:
            samples = target.shape[0]
            total_samples += samples
            data, target = torch.tensor(data).to(device), \
                torch.tensor(target).to(device, dtype=torch.int64)
            output = unet_model(data)
            val = soft_dice_coef(output, target)
            val_score += val.sum().cpu().numpy()

    return {'dice_coef': val_score / total_samples,}



# Create a federation
from openfl.interface.interactive_api.federation import Federation
federation = Federation(central_node_fqdn='localhost', disable_tls=True)
col_data_paths = {'one': '1,2', 'two': '2,2'}
federation.register_collaborators(col_data_paths=col_data_paths)

# create an experimnet in federation
fl_experiment = FLExperiment(federation=federation,)
fl_experiment.prepare_workspace_distribution(model_provider=MI, task_keeper=TI, data_loader=fed_dataset, rounds_to_train=EPOCHS , opt_treatment='CONTINUE_GLOBAL')
# This command starts the aggregator server
fl_experiment.start_experiment(model_provider=MI)

import argparse
import json
import math
import os
import shutil
import urllib.request
from importlib import import_module
from logging import DEBUG, FileHandler, Formatter, StreamHandler, getLogger
from math import floor

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchsso
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from torchsso.optim import SecondOrderOptimizer, VIOptimizer
from torchsso.utils import Logger
from torchvision import datasets, transforms

from models import Net
import optimizers

torch.manual_seed(1234)


def validate(model, device, val_loader, optimizer, mode: str = "Eval"):
    model.eval()
    val_mse = 0
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            """
            if isinstance(optimizer, VIOptimizer):
                output = optimizer.prediction(x_batch)
            else:
                output = model(x_batch)
            """
            output = model(x_batch)

            val_mse += F.mse_loss(output, y_batch, reduction="sum").item()
    val_mse = val_mse / len(val_loader.dataset)
    print("\n{} Average MSE: {}".format(mode, val_mse))

    return val_mse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", type=str,
                        default="data/3droad.mat", help="dataset file")
    parser.add_argument('--epochs', type=int, default=30,
                        help='number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=1024,
                        help='input batch size for training')
    parser.add_argument('--val_batch_size', type=int, default=1024,
                        help='input batch size for valing')
    # Training Settings
    parser.add_argument('--arch_file', type=str, default=None,
                        help='name of file which defines the architecture')
    parser.add_argument('--arch_name', type=str, default='LeNet5',
                        help='name of the architecture')
    parser.add_argument('--arch_args', type=json.loads, default=None,
                        help='[JSON] arguments for the architecture')
    parser.add_argument('--optim_name', type=str, default=VIOptimizer.__name__,
                        help='name of the optimizer')
    parser.add_argument('--optim_args', type=json.loads, default=None,
                        help='[JSON] arguments for the optimizer')
    parser.add_argument('--curv_args', type=json.loads, default=dict(),
                        help='[JSON] arguments for the curvature')
    parser.add_argument('--fisher_args', type=json.loads, default=dict(),
                        help='[JSON] arguments for the fisher')
    parser.add_argument('--scheduler_name', type=str, default=None,
                        help='name of the learning rate scheduler')
    parser.add_argument('--scheduler_args', type=json.loads, default=None,
                        help='[JSON] arguments for the scheduler')
    # Options
    parser.add_argument('--no_cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed')
    parser.add_argument('--log_interval', type=int, default=20,
                        help='how many batches to wait before logging training status')
    parser.add_argument('--checkpoint_interval', type=int, default=50,
                        help='how many epochs to wait before logging training status')
    parser.add_argument('--out', type=str, default='results/regression/',
                        help='dir to save output files')
    parser.add_argument('--config', default=None,
                        help='config file path')
    parser.add_argument("--log_name", default=None, required=True, type=str,
                        help="log name")
    args = parser.parse_args()
    dict_args = vars(args)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    if args.config is not None:
        with open(args.config) as f:
            config = json.load(f)

        dict_args.update(config)

    if not os.path.isfile(args.datapath):
        print('Downloading \'3droad\' UCI dataset...')
        urllib.request.urlretrieve(
            'https://www.dropbox.com/s/f6ow1i59oqx05pl/3droad.mat?dl=1', args.datapath)
    data = loadmat(args.datapath)['data']
    X = data[:, :-1]
    X = X - X.min(0)[0]
    X = 2 * (X / X.max(0)[0]) - 1
    y = data[:, -1]

    train_val_n = int(floor(0.8 * len(X)))
    X_train_val = X[:train_val_n]
    y_train_val = y[:train_val_n]
    X_test = X[train_val_n:, :]
    y_test = y[train_val_n:]

    X_train, X_val, y_train, y_val = train_test_split(X_train_val,
                                                      y_train_val,
                                                      test_size=0.2)
    dtype = torch.float32
    X_train = torch.tensor(X_train, dtype=dtype, requires_grad=False)
    y_train = torch.tensor(y_train, dtype=dtype, requires_grad=False)
    X_val = torch.tensor(X_val, dtype=dtype, requires_grad=False)
    y_val = torch.tensor(y_val, dtype=dtype, requires_grad=False)
    X_test = torch.tensor(X[train_val_n:, :], dtype=dtype, requires_grad=False)
    y_test = torch.tensor(y[train_val_n:], dtype=dtype, requires_grad=False)

    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True)
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(
        val_dataset, batch_size=args.val_batch_size, shuffle=True)
    test_dataset = TensorDataset(X_test, y_test)
    test_loader = DataLoader(
        test_dataset, batch_size=args.val_batch_size, shuffle=True)

    data_dim = X_train.size(-1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = Net(data_dim).to(device).float()

    optim_kwargs = {} if args.optim_args is None else args.optim_args

    if args.optim_name == SecondOrderOptimizer.__name__:
        optimizer = SecondOrderOptimizer(
            model, **config["optim_args"], curv_kwargs=config["curv_args"])
    elif args.optim_name == VIOptimizer.__name__:
        optimizer = VIOptimizer(model, dataset_size=len(train_loader.dataset), seed=args.seed,
                                **config["optim_args"], curv_kwargs=config["curv_args"])
    else:
        modules = import_module("optimizers")
        optim_class = getattr(modules, args.optim_name)
        optimizer = optim_class(model.parameters())

    if args.scheduler_name is None:
        scheduler = None
    else:
        scheduler_class = getattr(
            torchsso.optim.lr_scheduler, args.scheduler_name, None)
        if scheduler_class is None:
            scheduler_class = getattr(
                torch.optim.lr_scheduler, args.scheduler_name)
        scheduler_kwargs = config["scheduler_args"]
        scheduler = scheduler_class(optimizer, **scheduler_kwargs)

    log_file_name = "log_" + args.log_name
    logger = Logger(args.out, log_file_name)
    logger.start()
    # train
    epochs = 120
    model.train()
    print("=========== Start ===========")
    for i in range(args.epochs):
        losses = 0
        for minibatch_i, (x_batch, y_batch) in enumerate(train_loader):

            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            """
            def closure():
                optimizer.zero_grad()
                output = model(x_batch)
                loss = F.mse_loss(output, y_batch, reduction="sum").float()
                loss.backward()

                return loss, output
            """
            optimizer.zero_grad()
            output = model(x_batch)
            loss = F.mse_loss(output, y_batch, reduction="sum").float()
            loss.backward()
            optimizer.step()
            losses += loss.item()
            if (minibatch_i + 1) % args.log_interval == 0:
                print("Train Epoch: {} batch idx: {} elapsed time:  {:.1f}s MSE: {}".format(
                    i+1, minibatch_i+1, logger.elapsed_time, loss.item() / args.batch_size))
        losses = losses / len(train_loader.dataset)

        val_mse = validate(model, device, val_loader, optimizer, "Eval")

        iteration = (i + 1) * len(train_loader)
        log = {"epoch": i+1,
               "iteration": iteration,
               "mse": losses,
               "val_mse": val_mse,
               "lr": optimizer.param_groups[0]["lr"],
               "momentum": optimizer.param_groups[0].get("momentum", 0)}
        logger.write(log)

        if i % args.checkpoint_interval == 0 or i + 1 == args.epochs:
            path = os.path.join(
                args.out, "epoch{}_{}.ckpt".format(i+1, args.log_name))
            data = {"model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": i + 1}
            torch.save(data, path)

    print("=========== Test ===========")
    test_mse = validate(model, device, test_loader, optimizer, "Test")
    log = {"test_mse": test_mse}
    logger.write(log)


if __name__ == "__main__":
    main()

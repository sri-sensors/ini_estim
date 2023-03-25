import sys
import argparse
import pathlib
from tqdm.auto import tqdm, trange
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import ini_estim.ml as ml


class Encoder(nn.Module):
    def __init__(self, ndim=32):
        super().__init__()
        self.ndim = ndim
        self.enc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, self.ndim)
        )
    def forward(self, x):
        return self.enc(x)
    

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train an encoder for MNIST dataset by maximizing mutual information."
    )
    parser.add_argument(
        "-e", "--epochs", metavar="epochs", type=int, default=10,
        help="Number of training epochs. Default is 10."
    )
    parser.add_argument(
        "-l", "--loss", metavar="loss", type=str, 
        choices=["mi", "jsd"], default="jsd",
        help="Choice of loss function. Options are \"mi\" or \"jsd\" (default)."
    )
    parser.add_argument(
        "output_folder", metavar="output_folder", type=str,
        default="results", help="Destination directory for results. Default: ./results"
    )
    parser.add_argument(
        "-df", "--data_folder", metavar="data_folder", type=str,
        default="", help="MNIST data directory. If unspecified, the data will "
        "be downloaded to a sub-directory \"data\" in the results directory."
    )
    parser.add_argument(
        "-d", "--dimensions", metavar="encoder_dimensions", type=int,
        default=32, help="Number of dimensions at the output of the encoder. Default is 32."
    )
    parser.add_argument(
        "-sf", "--save_frequency", metavar="save_frequency", type=int,
        default=5, help="How often (# of epochs) to save intermediate results. Default is 5."
    )
    return parser.parse_args()


def load_data(data_folder, batch_size=64):
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
    ])
    train_set = torchvision.datasets.MNIST(
        root=data_folder, train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, drop_last=True
        )
    return train_set, train_loader, batch_size


def save_progress(args, encoder, epochs, loss_history, output_folder):
    output_name = "encoder{}_{}epochs_{}.pth".format(encoder.ndim, epochs, args.loss)
    outpath = pathlib.Path(output_folder, output_name)
    torch.save(encoder.state_dict(), outpath)
    output_loss_path = output_folder / "encoder{}_{:g}epochs_{}_losshistory.csv".format(encoder.ndim, epochs, args.loss)
    np.savetxt(output_loss_path, loss_history, delimiter=",")

def main():
    args = parse_args()
    print("Training MNIST for {} epochs using {} loss function.".format(
        args.epochs, args.loss
    ))
    output_folder = pathlib.Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    if not len(args.data_folder):
        mnist_folder = pathlib.Path(args.output_folder) / "mnist_data"
    else:
        mnist_folder = pathlib.Path(args.data_folder)
    train_set, train_loader, batch_size = load_data(mnist_folder)
    
    encoder = Encoder(args.dimensions)
    mine_model = ml.mine.BasicNet(28*28, encoder.ndim)
    opt_mi = optim.Adam(mine_model.parameters())
    opt_enc = optim.Adam(encoder.parameters())

    if args.loss == "mi":
        loss_fun = ml.loss.mine_loss
    else:
        loss_fun = ml.loss.jsd_loss
    data_numel = 28*28
    epochs = args.epochs
    its_per_epoch = (len(train_set))//batch_size
    loss_history = []
    last_epoch = -1
    for epoch in trange(epochs):    
        keyint = False
        running_loss = 0.0
        for i, data in tqdm(enumerate(train_loader), leave=False, total=its_per_epoch):
            try:
                inputs, labels = data
                opt_mi.zero_grad()
                opt_enc.zero_grad()
                out = encoder(inputs)
                N = inputs.shape[0]
                inputs_rand = inputs[np.random.choice(N, N)]
                Tp = mine_model(inputs.view(-1, data_numel), out.view(-1, encoder.ndim))
                Tq = mine_model(inputs_rand.view(-1, data_numel), out.view(-1, encoder.ndim))
                loss = loss_fun(Tp, Tq)
                loss.backward()
                opt_mi.step()
                opt_enc.step()
                running_loss += loss.item()
            except KeyboardInterrupt:
                print("Stopping training!")
                keyint = True               
                last_epoch += i/its_per_epoch
                break
        if keyint:
            break

        last_epoch = epoch
        loss_history.append(running_loss/its_per_epoch)
        if (epoch + 1) % args.save_frequency == 0:
            print("\nSaving progress...")
            save_progress(args, encoder, epoch + 1, loss_history, output_folder)
    if keyint:
        epochs_completed = last_epoch + 1
    else:
        epochs_completed = epochs
    print("Training complete. Writing results to: ", output_folder)
    save_progress(args, encoder, epochs_completed, loss_history, output_folder)


if __name__ == "__main__":
    main()

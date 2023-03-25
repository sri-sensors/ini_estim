import argparse
import pathlib
import json
import ini_estim.datasets as ini_datasets
from ini_estim.ml.training.direct_encoding import train_mi_encoder, get_encoder
from ini_estim.ml.loss import InfoMaxLoss
import ini_estim.ml.discriminator_models as dm
from torch.utils.data import DataLoader



def float_ranged(arg, minval=0.0, maxval=1.0):
    """ Float with min/max type argument for argparse """
    try:
        f = float(arg)
    except ValueError:    
        raise argparse.ArgumentTypeError("Must be a floating point number")
    if f < minval or f > maxval:
        raise argparse.ArgumentTypeError(
            "Argument must be >= {:g} and <= {:g}".format(minval, maxval)
            )
    return f


def parse_args(args=None, prog=None):
    parser = argparse.ArgumentParser(
        prog=prog, description="Train an encoder by maximizing mutual information"
    )
    parser.add_argument(
        "-e", "--epochs", type=int, default=50,
        help="Number of training epochs. Default is 50."
    )
    parser.add_argument(
        "-b", "--batch_size", type=int, default=32,
        help="The batch size for each iteration within an epoch. Default is 32."
    )
    parser.add_argument(
        "-l", "--loss", type=str, choices=["mine", "jsd"], default="jsd",
        help="Choice of loss function. Options are \"mine\" or the default, "
             "\"jsd\". "
    )
    parser.add_argument(
        "-o", "--output_folder", type=str, default="./results",
        help="Destination directory for results. Default is ./results"
    )
    parser.add_argument(
        "-i", "--input_folder", type=str, default="./data",
        help="Directory where input data is stored. Default is ./data"
    )
    parser.add_argument(
        "-d", "--dimensions", type=int, default=1,
        help="Number of dimensions (features) in final output. Note that for "
             "sequences, the output will be seq_len x dimensions."
    )
    parser.add_argument(
        "-ds", "--dataset", type=str, choices=ini_datasets.datasets.keys(), 
        default="uci_har", help="The dataset to train on. The default is uci_har"
    )
    parser.add_argument(
        "-si", "--save_interval", type=int, default=5,
        help="How often (# of epochs) to save the model and results (default: 5)."
    )

    # Sub-parsers for different encoders
    subparsers = parser.add_subparsers(
        title="model", description="The network model to use for the encoder."
        "Each model has its own set of sub-options.",
        dest="model")
    subparsers.required = True
    subparsers.metavar = "model"

    # ESN
    pesn = subparsers.add_parser(
        "esn", help="Echo state network with linear output layer.")
    pesn.add_argument(
        "-hu", "--hidden_units", type=int, default=256, 
        help="The number of units in the reservoir. The default is 256."
    )
    pesn.add_argument(
        "--density", type=float_ranged, default=0.25,
        help="The density of the hidden layer network. Default is 0.25"
    )
    pesn.add_argument(
        "--spectral_radius", type=float, default=0.9,
        help="The spectral radius of the hidden layer matrix. Default is 0.9"
    )
    pesn.add_argument(
        "--leak_rate", type=float_ranged, default=1.0,
        help="The leak rate of the hidden state. The default value is 1.0, "
             "meaning that there is no integration of preceding states."
    )

    # MLP
    pmlp = subparsers.add_parser(
        "mlp", help="Multi-layer perceptron with 1 hidden layer.")
    pmlp.add_argument(
        "-hu", "--hidden_units", type=int,
        help="The number of nodes in hidden layer. ")

    return parser.parse_args(args=args)


def save_args(args):
    if not args.get("output_folder"):
        return
    
    save_path = pathlib.Path(args["output_folder"], "config.json")
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w+') as f:
        json.dump(args, f, indent=2)


def main(args=None, prog=None):
    args = parse_args(args, prog)
    train_set, val_set, test_set, num_features_in, num_samples, variablelen = \
        ini_datasets.get_dataset(args.dataset, args.input_folder)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    validation_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, drop_last=False
    )
    num_features_out = args.dimensions
    Nin = num_features_in*num_samples
    Nout = num_features_out*num_samples

    argsdict = vars(args)
    argsdict['in_features'] = num_features_in
    save_args(argsdict)

    lossfun = InfoMaxLoss(dm.MLPDiscriminator(Nin, Nout), loss_type=args.loss)
    encoder = get_encoder(
        args.model, num_features_in, num_features_out, vars(args))

    train_mi_encoder(encoder, lossfun, train_loader, validation_loader, 
        test_loader, num_epochs=args.epochs, save_dir=args.output_folder,
        validation_interval=args.save_interval)
    

if __name__ == "__main__":
    main()

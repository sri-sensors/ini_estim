import argparse
import json
import pathlib
import warnings
import torch
from distutils.util import strtobool
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from ini_estim.datasets import datasets, get_dataset
from ini_estim.ml.models import ESN, MLP, TimeSeriesCNN, TimeSeriesMultiCNN
from ini_estim.ml.training import (
                                    DirectInfoMaxTrainer, 
                                    CPCTrainer, 
                                    SeriesPredictiveCodingTrainer,
                                    SeriesPredictiveCodingTrainer2
                                    )
from ini_estim.config import get_config
from ini_estim.ml.utilities import padder
from ini_estim.scripts.utils import get_path, float_ranged



def targs(help, type_, default, **kwargs):
    return dict(help=help, type=type_, default=default, **kwargs)

trainers = {
    "direct": (DirectInfoMaxTrainer, "Direct MI maximization between input and output."),
    "cpc": (CPCTrainer, "Contrastive Predictive Coding - maximize MI with future encodings."),
    "spc": (SeriesPredictiveCodingTrainer, 
            "Series Predictive Coding - maximize MI with future inputs."),
    "spc2": (SeriesPredictiveCodingTrainer2, 
            "Series Predictive Coding (alternate) - maximize MI with future inputs.")
}
trainer_args = {
    "direct": {
        "model_lr": targs("Model learning rate", float, 1e-3),
        "loss_lr": targs("Loss learning rate", float, 1e-3)
    },
    "cpc": {
        "model_lr": targs("Model learning rate", float, 1e-3),
        "loss_lr": targs("Loss learning rate", float, 1e-3),
        "num_pred": targs("Number of prediction steps", int, 10),
        "use_jsd": targs(
            "Flag to use Jensen-Shannon Divergence instead of InfoNCE measure", 
            strtobool, 
            False
            ),
        "context_net": targs("Context network type", str, "none",
            choices=["cnn", "none"]
            )
    },
    "spc": {
        "model_lr": targs("Model learning rate", float, 1e-3),
        "loss_lr": targs("Loss learning rate", float, 1e-3),
        "num_pred": targs("Number of prediction steps", int, 3),
    },
    "spc2": {
        "model_lr": targs("Model learning rate", float, 1e-3),
        "loss_lr": targs("Loss learning rate", float, 1e-3),
        "num_pred": targs("Number of prediction steps", int, 3),
        "discriminator": targs("Loss Discriminator", str, "mlp", 
            choices=["mlp", "bilinear", "bilinear2", "bilinear3"]
            ),
        "include0": targs("Include current sample in MI", strtobool, False)
    }
}


def get_valid_models():
    out = {
        "esn": ESN,
        "mlp": MLP,
        "cnn": TimeSeriesCNN,
        "mcnn": TimeSeriesMultiCNN
    }
    return out



def parse_args(args=None, prog=None):
    inicfg = get_config()
    default_datadir = inicfg.get("dataset_directory")
    if default_datadir is None:
        default_datadir = "./data"
    default_savefolder = "./results"
    cfgparser = argparse.ArgumentParser(
        prog=prog, description="Config", allow_abbrev=False, add_help=False
    )
    cfgparser.add_argument(
        "--cfg", type=str, default=None,
        help="Optional configuration file with all arguments in JSON format."
    )
    cfgparser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Optional checkpoint file to initialize training. Overrides config file path."
    )
    cfgparser.add_argument(
        "--force", action="store_true", default=False, 
        help="Flag to force saving even if existing contents will be overwritten.")

    cfgargs, unknown = cfgparser.parse_known_args(args)
    if cfgargs.checkpoint is not None:
        checkpoint_dir = pathlib.Path(cfgargs.checkpoint).parent
        newcfgfile = checkpoint_dir / "config.json"
        if newcfgfile.exists():
            print("Setting configuration file to: ", str(newcfgfile))
            cfgargs.cfg = newcfgfile
        default_savefolder = str(checkpoint_dir)

    if cfgargs.cfg is not None:
        with open(cfgargs.cfg) as f:
            default_cfg = json.load(f)
        if "checkpoint" in default_cfg:
            del default_cfg["checkpoint"]
        default_savefolder = str(pathlib.Path(cfgargs.cfg).parent)
        default_cfg["save_folder"] = default_savefolder
    else:
        default_cfg = {}
    parser = argparse.ArgumentParser(
        prog=prog, description="Train an encoder", allow_abbrev=False, parents=[cfgparser]
    )
    parser.add_argument(
        "--epochs", type=int, default=250,
        help="Number of training epochs. Default is 250."
    )
    parser.add_argument(
        "-b", "--batch_size", type=int, default=32,
        help="The batch size for each iteration within an epoch. Default is 32."
    )
    parser.add_argument(
        "-s", "--save_folder", type=get_path, default=default_savefolder,
        help="Destination directory for results. Default is {}".format(default_savefolder)
    )
    parser.add_argument(
        "-d", "--data_folder", type=str, default=default_datadir,
        help="Directory where input data is stored. Default is {}".format(default_datadir)
    )
    parser.add_argument(
        "-data", "--dataset", choices=datasets.keys(),
        default="uci_har", help="The target dataset, uci_har by default."
    )
    parser.add_argument(
        "--noise", type=float, default=0.0,
        help="The amount of relative noise to add to the dataset."
    )
    parser.add_argument(
        "-f", "--features", type=int, default=1, help="Number of output features in the encoder"
    )
    models = get_valid_models()
    model_help = "The encoder to train."
    for k, v in models.items():
        model_help += "\n\t{}: {}".format(k, v.description())
    parser.add_argument(
        "-enc", "--encoder", choices=get_valid_models().keys(), default="esn",
        help=model_help
    )
    parser.add_argument(
        "-hn", "--hidden_nodes", type=int, default=None, 
        help="The number of hidden nodes. Valid option for ESN, MLP, or CNN."
    )
    parser.add_argument(
        "-ks", "--kernel_size", type=int, default=3, 
        help="The kernel size for CNN encoder."
    )
    parser.add_argument(
        "-group", "--group_convolutions", action="store_true",
        help="Flag for using grouped convolutions with CNN. If not set, each "
             "input channel has its own set of kernels. Otherwise, "
             "convolutions will combine channels."
    )
    parser.add_argument(
        "-fp", "--force_positive", action="store_true",
        help="Force output to be positive."
    )
    parser.add_argument(
        "-nl", "--num_layers", type=int, default=3,
        help="The number of layers for a multi-CNN encoder"
    )
    parser.add_argument(
        "--density", type=float_ranged, default=0.25,
        help="Density for ESN hidden layer, by default 0.25. Ignored for MLP."
    )
    parser.add_argument(
        "--spectral_radius", type=float, default=0.9,
        help="Spectral radius of ESN hidden layer, by default 0.9. Ignored for MLP."
    )
    parser.add_argument(
        "--leak_rate", type=float_ranged, default=1.0,
        help="The leak rate of the ESN hidden layer, by default 1.0 (no leaky "
            "integration). Ignored for MLP."
    )
    parser.add_argument(
        "--disable_gpu", "-no_gpu", action="store_true",
        help="Flag for disabling GPU usage."
    )
    subparsers = parser.add_subparsers(
        title="method", dest="method", metavar="method",
        description="The optimization method"
    )
    for k, v in trainer_args.items():
        p = subparsers.add_parser(k, help=trainers[k][1])
        for name, params in v.items():
            p.add_argument("--" + name, **params)
    parser.set_defaults(**default_cfg)
    args = parser.parse_args(unknown)
    
    return args, cfgargs.checkpoint, cfgargs.force


def save_args(args, force_save=False):
    outfile = pathlib.Path(args.save_folder) / "config.json"
    outfile.parent.mkdir(parents=True, exist_ok=True)
    check_existence = not force_save
    if check_existence and outfile.exists():
        k = input(
            "Warning: a configuration file already exists in this folder. "
            "Contents will be overwritten. Enter \"y\" to proceed, anything else "
            "to quit: "
        )
        if k.lower() != "y":
            exit(-1)
    
    outdict = vars(args).copy()
    if "cfg" in outdict:
        del outdict['cfg']
    if "save_folder" in outdict:
        del outdict["save_folder"]
    
    with open(outfile, 'w+') as f:
        json.dump(outdict, f, indent=2)


def get_model(args, data_features):
    in_features = data_features
    out_features = args.features
    hidden_size = args.hidden_nodes
    out_act = "relu" if args.force_positive else None
    if args.encoder == "mlp":
        if args.hidden_nodes is None:
            args.hidden_nodes = (in_features + out_features + 1) // 2
            save_args(args, False)
        mlp_size = [data_features, args.hidden_nodes, args.features]
        model = MLP(mlp_size, flatten=False, out_activation=out_act)
    elif args.encoder == "esn":
        if args.hidden_nodes is None:
            args.hidden_nodes = 128
            save_args(args, False)
        model = ESN(
            data_features, args.features, args.hidden_nodes, 
            density=args.density, spectral_radius=args.spectral_radius, 
            leak_rate=args.leak_rate, out_activation=out_act
        )
    elif args.encoder == "cnn":
        if args.group_convolutions:
            groups = 1
        else:
            groups = data_features
        # make sure hidden nodes is multiple of groups
        hidden_nodes = groups * ((args.hidden_nodes + groups - 1) // groups)
        model = TimeSeriesCNN(
            data_features, args.features, args.kernel_size, hidden_nodes,
            groups=groups, out_activation=out_act
        )
    elif args.encoder == "mcnn":
        if args.group_convolutions:
            groups = 1
        else:
            groups = in_features
        # make sure hidden nodes is multiple of groups
        hidden_nodes = groups * ((args.hidden_nodes + groups - 1) // groups)
        model = TimeSeriesMultiCNN(
            in_features, out_features, args.kernel_size, hidden_nodes,
            groups=groups, num_layers=args.num_layers, out_activation=out_act
        )
    else:
        raise ValueError("Unsupported model \"{}\"".format(args.encoder))

    return model


def main(args=None, prog=None):
    args, checkpoint, force_save = parse_args(prog=prog, args=args)
    save_args(args, force_save)
    print("Loading {} dataset...".format(args.dataset))
    dataset = get_dataset(args.dataset, args.data_folder, noise=args.noise)
    model = get_model(args, data_features=dataset.num_features)
    TrainerClass = trainers[args.method][0]
    
    argsdict = vars(args)
    targs = {k: argsdict[k] for k in trainer_args[args.method].keys()}
    trainer = TrainerClass(model, save_dir=args.save_folder, disable_cuda=args.disable_gpu, **targs)
    if checkpoint is not None:
        print("Loading checkpoint: {}".format(checkpoint))
        trainer.load_checkpoint(checkpoint)
        print("Checkpoint loaded with {} epochs of training.".format(trainer.next_epoch))
    
    collate_fun = padder if dataset.variable_length else None
    loader_args = dict(batch_size=args.batch_size, collate_fn=collate_fun)
    trainer.train_loader = DataLoader(dataset.train_set, **loader_args)
    trainer.test_loader = DataLoader(dataset.test_set, **loader_args)
    trainer.validation_loader = DataLoader(dataset.val_set, **loader_args)

    logdir = pathlib.Path(args.save_folder) / "logs"
    writer = SummaryWriter(logdir, flush_secs=30)

    def callback(epoch):
        epoch = trainer.next_epoch - 1
        writer.add_scalar("train", trainer.train_loss[-1], epoch)
        if len(trainer.validation_loss):
            vepoch, loss = trainer.validation_loss[-1]
            if vepoch == epoch:
                writer.add_scalar("validation", loss, epoch)
        if len(trainer.test_loss):
            tepoch, loss = trainer.test_loss[-1]
            if tepoch == epoch:
                writer.add_scalar("test", loss, epoch)

    trainer.post_epoch_callback = callback
    print("Starting training using {}... TensorBoard logs located at: {}".format(
            trainer.device, logdir)
         )
    try:
        trainer.train(args.epochs)
    except KeyboardInterrupt:
        print("Training was interrupted. Shutting down.")
    
    writer.close()
    print("Training finished. Results are located at: {}".format(args.save_folder))


if __name__ == "__main__":
    main()
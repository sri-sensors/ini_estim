import argparse
import json
import pathlib
import warnings
import torch

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from ini_estim.datasets import datasets, get_dataset
from ini_estim.ml.models import ESN, MLP, TimeSeriesCNN, TimeSeriesMultiCNN, Linear, Diagonal, model_from_checkpoint
from ini_estim.config import get_config
from ini_estim.ml.utilities import padder
from ini_estim.scripts.utils import get_path, float_ranged
from ini_estim.ml.training.array_encoder import ElectrodeArrayDIMTrainer
from ini_estim.electrodes import electrode_arrays


def get_valid_models():
    out = {
        "esn": ESN,
        "mlp": MLP,
        "cnn": TimeSeriesCNN,
        "mcnn": TimeSeriesMultiCNN,
        "linear": Linear,
        "diagonal": Diagonal
    }
    return out


def get_config_parser(args=None, prog=None):
    inicfg = get_config()
    default_datadir = inicfg.get('dataset_directory')
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
    return cfgparser, default_datadir, default_savefolder
    

def parse_args(args=None, prog=None):
    cfgparser, default_datadir, default_savefolder = get_config_parser(args, prog)
    cfg, args = cfgparser.parse_known_args(args)
    if cfg.checkpoint is not None:
        checkpoint_dir = pathlib.Path(cfg.checkpoint).parent
        newcfgfile = checkpoint_dir / "config.json"
        if newcfgfile.exists():
            print("Setting configuration file to: ", str(newcfgfile))
            cfg.cfg = newcfgfile
        default_savefolder = str(checkpoint_dir)

    if cfg.cfg is not None:
        with open(cfg.cfg) as f:
            default_cfg = json.load(f)
        if "checkpoint" in default_cfg:
            del default_cfg["checkpoint"]
        default_savefolder = str(pathlib.Path(cfg.cfg).parent)
        default_cfg["save_folder"] = default_savefolder
    else:
        default_cfg = {}

    parser = argparse.ArgumentParser(
        prog=prog, description="Train data encoder for electrode array", 
        allow_abbrev=False, parents=[cfgparser]
    )
    parser.add_argument(
        "--epochs", type=int, default=500,
        help="Number of training epochs. Default is 500."
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
        "-data", "--dataset", choices=datasets.keys(), required=True,
        help="The target dataset, uci_har by default."
    )
    parser.add_argument(
        "--noise", type=float, default=0.0,
        help="The amount of relative noise to add to the dataset."
    )
    parser.add_argument(
        "-pre", "--pre_encoder", type=str, default=None,
        help="Path to pre-encoder checkpoint"
    )
    ea_options = list(electrode_arrays.keys())
    parser.add_argument(
        "-ea", "--electrode_array", type=str, choices=ea_options,
        default="generic_cuff", help="The electrode array type."
    )

    parser.add_argument(
        "-np", "--num_sampling_points", type=int, default=500, 
        help="Number of sampling points in the electrode array, by default 500"
    )
    models = get_valid_models()
    model_help = "The encoder to train"
    for k, v in models.items():
        model_help += "\n\t{}: {}".format(k, v.description())
    parser.add_argument(
        "-enc", "--encoder", choices=models.keys(), default="esn",
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
        "-is", "--input_scale", type=float, default=1.0,
        help="The input scale factor to the electrode array. By default 1.0."
    )
    parser.add_argument(
        "-fp", "--force_positive_current", action="store_true",
        help="Force input current to electrode array to be positive."
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
    parser.add_argument(
        "--model_lr", type=float, help="model learning rate", default=2e-4
    )
    parser.add_argument(
        "--loss_lr", type=float, help="loss learning rate", default=2e-4
    )
    parser.set_defaults(**default_cfg)
    args = parser.parse_args(args)
    return args, cfg.checkpoint, cfg.force
    

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


def main(args=None, prog=None):
    args, checkpoint, force_save = parse_args(args, prog)
    save_args(args, force_save)
    
    print("Loading {} dataset...".format(args.dataset))
    dataset = get_dataset(args.dataset, args.data_folder, noise=args.noise)
    if args.pre_encoder:
        print("Loading pre-encoder")
        pre_encoder = model_from_checkpoint(args.pre_encoder)
        print(pre_encoder)
        in_features = pre_encoder.out_features
    else:
        in_features = dataset.num_features
        pre_encoder = None
    eacls = electrode_arrays[args.electrode_array]
    num_encoder_features = eacls.MAX_ELECTRODES
    model = get_model(args, in_features, num_encoder_features)
    trainer = ElectrodeArrayDIMTrainer(
        model, args.electrode_array, args.num_sampling_points, 
        pre_encoder_model=pre_encoder,
        input_scale=args.input_scale, save_dir=args.save_folder, 
        disable_cuda=args.disable_gpu, model_lr=args.model_lr,
        loss_lr=args.loss_lr, force_positive_current=args.force_positive_current
    )
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



def get_model(args, in_features, out_features):
    hidden_size = args.hidden_nodes
    out_act = "relu" if args.force_positive_current else None
    if args.encoder == "mlp":
        if args.hidden_nodes is None:
            args.hidden_nodes = (in_features + out_features + 1) // 2
            save_args(args, False)
        mlp_size = [in_features, args.hidden_nodes, out_features]
        model = MLP(mlp_size, flatten=False, out_activation=out_act)
    elif args.encoder == "esn":
        if args.hidden_nodes is None:
            args.hidden_nodes = 128
            save_args(args, False)
        model = ESN(
            in_features, out_features, args.hidden_nodes, 
            density=args.density, spectral_radius=args.spectral_radius, 
            leak_rate=args.leak_rate, out_activation=out_act
        )
    elif args.encoder == "cnn":
        if args.group_convolutions:
            groups = 1
        else:
            groups = in_features
        # make sure hidden nodes is multiple of groups
        hidden_nodes = groups * ((args.hidden_nodes + groups - 1) // groups)
        model = TimeSeriesCNN(
            in_features, out_features, args.kernel_size, hidden_nodes,
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
    elif args.encoder == "linear":
        model = Linear(in_features, out_features, out_activation=out_act)
    elif args.encoder == "diagonal":
        if in_features != out_features:
            raise ValueError("Diagonal can only be used when in_features = out_features")
        model = Diagonal(in_features, out_activation=out_act)
    else:
        raise ValueError("Unsupported model \"{}\"".format(args.encoder))

    return model


if __name__ == "__main__":
    main()

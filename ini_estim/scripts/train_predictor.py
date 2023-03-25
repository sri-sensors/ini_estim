import argparse
import pathlib
import shutil
import warnings
import json
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from ini_estim.ml.models import model_from_checkpoint, TimeSeriesCNN
from ini_estim.ml.training import PredictorTrainer
from ini_estim.scripts.utils import get_path
from ini_estim.config import get_config
from ini_estim.datasets import datasets, get_dataset
from ini_estim.ml.utilities import padder


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
        prog=prog, description="Trains a CNN to predict a future sample of "
        " an input sequence based on an encoding.",  allow_abbrev=False, 
        parents=[cfgparser]
    )
    parser.add_argument(
        "-encoder", "--encoder_checkpoint", type=get_path, default=None,
        help="The data encoder to load. Leave unspecified for no encoder."
        )
    parser.add_argument(
        "-la", "--lookahead", type=int, default=1, 
        help="The number of time steps ahead to predict, by default 1."
    )
    parser.add_argument(
        "-hn", "--hidden_nodes", type=int, default=16,
        help="The number of hidden nodes in the decoder, by default 16."
    )
    parser.add_argument(
        "-k", "--kernel_size", type=int, default=3,
        help="The kernel size of the convolutional weights, by default 3."
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
        "--epochs", type=int, default=250,
        help="Number of training epochs. Default is 250."
    )
    parser.add_argument(
        "-b", "--batch_size", type=int, default=32,
        help="The batch size for each iteration within an epoch. Default is 32."
    )
    parser.add_argument(
        "-lr", "--learning_rate", type=float, default=1e-3,
        help="The learning rate."
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
        "--disable_gpu", "-no_gpu", action="store_true",
        help="Flag for disabling GPU usage. "
    )
    parser.set_defaults(**default_cfg)
    return parser.parse_args(args), cfgargs.checkpoint, cfgargs.force


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
    with open(outfile, 'w+') as f:
        json.dump(outdict, f, indent=2)


def main(args=None, prog=None):
    args, checkpoint, force_save = parse_args(prog=prog, args=args)
    save_args(args, force_save)

    dataset = get_dataset(args.dataset, args.data_folder, noise=args.noise)

    encoder = model_from_checkpoint(args.encoder_checkpoint)
    if encoder is None:
        decoder_in = dataset.num_features
    else:
        decoder_in = encoder.out_features
    decoder = TimeSeriesCNN(
        decoder_in, dataset.num_features, args.kernel_size, 
        args.hidden_nodes
        )
    trainer = PredictorTrainer(
        decoder, encoder, args.lookahead, args.learning_rate, 
        save_dir=args.save_folder, disable_cuda=args.disable_gpu
        )
    if checkpoint is not None:
        trainer.load_checkpoint(checkpoint)
    
    collate_fun = padder if dataset.variable_length else None
    loader_args = dict(batch_size=args.batch_size, collate_fn=collate_fun)
    trainer.train_loader = DataLoader(dataset.train_set, **loader_args)
    trainer.test_loader = DataLoader(dataset.test_set, **loader_args)
    trainer.validation_loader = DataLoader(dataset.val_set, **loader_args)
    logdir = pathlib.Path(args.save_folder) / "logs"
    writer = SummaryWriter(logdir, flush_secs=15)

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
    print("Starting training... TensorBoard logs located at: ", str(logdir))
    try:
        trainer.train(args.epochs)
    except KeyboardInterrupt:
        print("Training was interrupted. Shutting down.")
    
    writer.close()
    print("Training finished. Results are located at: {}".format(args.save_folder))


if __name__ == "__main__":
    main()


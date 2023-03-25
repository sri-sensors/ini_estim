import argparse
import pathlib
import json
from ini_estim.scripts.utils import get_path, save_args
from torch.utils.tensorboard import SummaryWriter
from ini_estim.config import get_config
from ini_estim.datasets.base import XYCSVDataset
from ini_estim.ml.training.mine import MINEDataTrainer
from torch.utils.data import DataLoader


def parse_args(args=None, prog=None):
    inicfg = get_config()
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
        prog=prog, description="Estimate MI of data using MINE", #parent=[cfgparser]
    )
    parser.add_argument(
        "--epochs", type=int, default=500,
        help="Number of training epochs. Default is 500."
    )
    parser.add_argument(
        "-b", "--batch_size", type=int, default=4096,
        help="The batch size for each iteration within an epoch. Default is 4096."
    )
    parser.add_argument(
        "-lr", "--learning_rate", type=float, help="The learning rate", default=2e-3
    )
    parser.add_argument(
        "-s", "--save_folder", type=get_path, default=default_savefolder,
        help="Destination directory for results. Default is {}".format(default_savefolder)
    )
    parser.add_argument(
        "-xd", "--xdelimiter", type=str, default=",", help="Delimiter for x data"
    )
    parser.add_argument(
        "-yd", "--ydelimiter", type=str, default=",", help="Delimiter for y data"
    )
    parser.add_argument(
        "--normalize", action="store_true", default=False, help="Flag to normalize data"
    )
    parser.add_argument(
        "--randomize", action="store_true", default=False, help="Flag to randomize order"
    )
    parser.add_argument(
        "xcsvfile", type=str, help="path to x data (csv)"
    )
    parser.add_argument(
        "ycsvfile", type=str, help="path to y data (csv)"
    )
    parser.set_defaults(**default_cfg)
    args = parser.parse_args(unknown)
    return args, cfgargs.checkpoint, cfgargs.force


def main(args=None, prog=None):
    args, checkpoint, force_save = parse_args(args, prog)
    save_args(args, force_save)
    print("Loading data...")
    dataset = XYCSVDataset(
        args.xcsvfile, args.ycsvfile, args.xdelimiter, args.ydelimiter, args.normalize, args.randomize
    )
    trainer = MINEDataTrainer(
        dataset.xfeatures, dataset.yfeatures, args.learning_rate, args.save_folder
    )
    if checkpoint is not None:
        print("Loading checkpoint: {}".format(checkpoint))
        trainer.load_checkpoint(checkpoint)
        print("Checkpoint loaded with {} epochs of training.".format(trainer.next_epoch))
    trainer.train_loader = DataLoader(dataset, batch_size=args.batch_size, drop_last=False)
    logdir = pathlib.Path(args.save_folder)
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

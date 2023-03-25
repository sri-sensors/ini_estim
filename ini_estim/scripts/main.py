import argparse
import sys
from collections import namedtuple
import pathlib
from ini_estim.scripts.config import main as config_main
from ini_estim.scripts.export_model import main as export_main
from ini_estim.scripts.train_preencoder import main as train_encoder
from ini_estim.scripts.train_predictor import main as train_predictor
from ini_estim.scripts.deprecated.train_autoencoder import main as train_autoencoder_old
from ini_estim.scripts.deprecated.train_encoder import main as train_encoder_old
from ini_estim.scripts.deprecated.train_decoder import main as train_decoder_old
from ini_estim.scripts.train_ea_encoder import main as train_ea_encoder
from ini_estim.scripts.estimate_encoder_mi import main as estimate_encoder_mi
from ini_estim.scripts.estimate_data_mi import main as estimate_data_mi
Command = namedtuple("Command", ["help", "prog"])


commands = {
    "config": Command("Get/set the current INI-ESTIM configuration.", config_main),
    "export": Command("Export a model from a training checkpoint", export_main),
    "train_ea_encoder": Command("Train signal encoder for an electrode array using MI based opimtimization", train_ea_encoder),
    "train_encoder": Command("Train signal encoder using mutual information based optimization.", train_encoder),
    "train_predictor": Command(
        "Trains a CNN to predict a future sample of an input sequence based on an encoding.", train_predictor),
    "estimate_encoder_mi": Command(
        "Estimate mutual information between encoding and input data using MINE.", estimate_encoder_mi
    ),
    "estimate_data_mi": Command("Estimate MI for CSV data using MINE", estimate_data_mi),
    "train_encoder_old": Command("Train signal encoder using mutual information based optimization. (Deprecated)", train_encoder_old),
    "train_decoder_old": Command("Train decoder for signal encoded with train_signal_encoder old. (Deprecated)", train_decoder_old)
}


def parse_args(args=None, prog=None):
    parser = argparse.ArgumentParser(
        prog=prog, description="Run INI-ESTIM", add_help=False
    )
    subparsers = parser.add_subparsers(
        title="command", dest="command", 
        description="The INI-ESTIM command to run.",
        )
    
    for name, command in commands.items():
        subparsers.add_parser(name, help=command.help, add_help=False)
    
    args, command_args = parser.parse_known_args(args)
    if args.command is None:
        parser.print_help()
        exit(0)
    return args, command_args


def main(args=None, prog=None):
    if prog is None:
        prog = pathlib.Path(sys.argv[0]).name
    args, command_args = parse_args(args, prog)
    command = commands[args.command].prog
    prog = "{} {}".format(prog, args.command)
    command(args=command_args, prog=prog)


if __name__ == "__main__":
    main(prog=sys.argv[0])


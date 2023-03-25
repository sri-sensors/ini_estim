import argparse
import json
import pathlib
from ini_estim.datasets import get_dataset
from ini_estim.ml.training.direct_encoding import get_encoder
from ini_estim.ml.decoder import MLPDecoder, train_decoder
import torch
from torch.utils.data import DataLoader
import numpy as np

def parse_args(args=None, prog=None):
    parser = argparse.ArgumentParser(
        prog=prog, description="Train a decoder for a specific encoder.")
    parser.add_argument(
        "-cp", "--checkpoint", type=str, required=True, 
        help="Path to checkpoint file loadable with torch.load containing encoder state dictionary."
        )
    parser.add_argument(
        "-cfg", "--config", type=str, required=True,
        help="Path to JSON config file containing encoder and dataset specifications."
    )
    parser.add_argument(
        "-o", "--output_folder", type=str, default="./results",
        help="Destination directory for results. Default is ./results"
    )
    parser.add_argument(
        "-i", "--input_folder", type=str, default="./data",
        help="Directory where the input data are stored. Default is ./data"
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
        "-hf", "--hidden_unit_factor", type=float, default=5,
        help="Factor for determining number hidden units in decoder."
    )
    parser.add_argument(
        "-si", "--save_interval", type=int, default=5,
        help="How often (# of epochs) to save the model and results (default: 5)."
    )
    return parser.parse_args(args=args)


def save_args(args, cfg):
    if not args.get("output_folder"):
        return
    
    save_path = pathlib.Path(args["output_folder"])
    save_path.mkdir(parents=True, exist_ok=True)
    with open(save_path / "decoder_config.json", 'w+') as f:
        json.dump(args, f, indent=2)
    with open(save_path / "encoder_config.json", "w+") as f:
        json.dump(cfg, f, indent=2)


def main(args=None, prog=None):
    args = parse_args(args=args, prog=prog)
    with open(args.config, 'r') as f:
        cfg = json.load(f)
    
    out_features = cfg.get("dimensions")
    encoder_name = cfg.get("model")

    train_set, val_set, test_set, in_features, num_samples, variablelen = \
        get_dataset(cfg["dataset"],  args.input_folder)
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    validation_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, drop_last=False
    )
    encoder = get_encoder(encoder_name, in_features, out_features, cfg)
    checkpoint = torch.load(args.checkpoint)
    vargs = vars(args)
    vargs["num_samples"] = num_samples
    vargs["in_features"] = in_features
    save_args(vargs, cfg)
    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    encoder.eval()

    decoder_in = out_features*num_samples
    decoder_out = in_features*num_samples
    decoder = MLPDecoder(
        decoder_in, [num_samples, in_features], args.hidden_unit_factor
    )
    loss_history, validation_loss_history, test_loss_history = train_decoder(
        decoder, encoder, train_loader, validation_loader, test_loader,
        args.epochs, args.output_folder, args.save_interval
    )
    print("Finished. Final test loss: {}".format(test_loss_history[-1]))
    nloss = len(test_loss_history)
    tsteps = args.save_interval
    
    idx = np.argmin(test_loss_history)
    print("Lowest test loss: {:g} at {} epochs".format(
        test_loss_history[idx], tsteps*(idx+1)
    ))
    idx = np.argmin(validation_loss_history)
    print("Test loss at validation minimum: {:g} at {} epochs".format(
        test_loss_history[idx], tsteps*(idx+1)
    ))    

if __name__ == "__main__":
    main()
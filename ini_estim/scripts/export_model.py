import torch
import argparse


def parse_args(args=None, prog=None):
    parser = argparse.ArgumentParser(
        description="Export model from training checkpoint",
        prog=prog
    )
    parser.add_argument("training_checkpoint", help="Path to training checkpoint")
    parser.add_argument("output_path", help="Output filename, usually ending in .pt")
    parser.add_argument("--dryrun", help="Dry run only, no actual exporting.", action="store_true")
    return parser.parse_args(args=args)


def main(args=None, prog=None):
    args = parse_args(args, prog)
    
    print("Saving model from {} to {}".format(args.training_checkpoint, args.output_path))
    if not args.dryrun:
        checkpoint = torch.load(
            args.training_checkpoint, map_location=torch.device('cpu')
            )
        torch.save(checkpoint['model'], args.output_path)
    print("Finished.")
    

if __name__ == "__main__":
    main()

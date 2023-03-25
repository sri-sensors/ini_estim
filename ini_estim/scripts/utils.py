import argparse
import pathlib
import json


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


def get_path(path):
    if path is None:
        return None
    path = pathlib.Path(path).absolute()
    return str(path)



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

import torch
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import argparse


def plot(checkpoint_file):
    filepath = pathlib.Path(checkpoint_file)
    data = torch.load(filepath)
    test_loss = data.get("test_loss", None)
    test_loss_history = data.get("test_loss_history", [])
    if len(test_loss_history):
        test_loss = test_loss_history[-1]
    print("Final test loss: {:g}".format(test_loss))
    lh = data["loss_history"]
    plt.figure()
    plt.plot(np.arange(1, len(lh)+1), lh, label="training")
    if len(data["validation_loss_history"]):
        vl = data["validation_loss_history"]
        vlsteps = (len(lh) + len(vl) - 1)//len(vl)
        plt.plot(
            np.arange(vlsteps, vlsteps*len(vl) + 1, vlsteps), vl, 
            label="validation"
        )
    else:
        vl = None
    if len(test_loss_history):
        nloss = len(test_loss_history)
        tsteps = (nloss + len(lh) - 1)//nloss
        
        idx = np.argmin(test_loss_history)
        print("Lowest test loss: {:g} at {} epochs".format(
            test_loss_history[idx], tsteps*(idx+1)
        ))
        if vl:
            idx = np.argmin(vl)
            print("Test loss at validation minimum: {:g} at {} epochs".format(
                test_loss_history[idx], tsteps*(idx+1)
            ))

        plt.plot(
            np.arange(tsteps, tsteps*nloss + 1, tsteps), test_loss_history,
            label="test"
        )
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("{}/{}".format(filepath.parent.name, filepath.stem))
    plt.legend()
    plt.grid()

    plt.show()


def main():
    parser = argparse.ArgumentParser(usage="Plot loss data from a checkpoint file")
    parser.add_argument(
        "checkpoint_file", type=str, 
        help="path to checkpoint file."
    )
    args = parser.parse_args()
    plot(args.checkpoint_file)


if __name__ == "__main__":
    main()

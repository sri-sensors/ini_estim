"""Reads in .CSV files that contain multiple-repetition sensor recordings from
the DEKA LUKE arm, and reformats to txt file or generate plots."""
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from pathlib import Path


def reshape_data(data, col_names, start_pt, trial_length, n_repeats):
    sensors = []
    positions = []
    status = []
    sensor_data = []
    position_data = []
    status_data = []
    for i, c in enumerate(col_names):
        # print('{}: {}'.format(i, c))
        if 'force_sensor' in c and 'status' not in c:
            sensors.append(c)
            if len(sensor_data) == 0:
                sensor_data = data[c].to_numpy()
            else:
                sensor_data = np.column_stack(
                    (sensor_data, data[c].to_numpy())
                )
        if 'position' in c:
            positions.append(c)
            if len(position_data) == 0:
                position_data = data[c].to_numpy()
            else:
                position_data = np.column_stack(
                    (position_data, data[c].to_numpy())
                )
        if 'status' in c:
            status.append(c)
            if len(status_data) == 0:
                status_data = data[c].to_numpy()
            else:
                status_data = np.column_stack(
                    (status_data, data[c].to_numpy())
                )
    sensor_data = np.reshape(
        sensor_data[start_pt:(start_pt + trial_length * n_repeats), :],
        (n_repeats, trial_length, sensor_data.shape[1])
    )

    position_data = np.reshape(
        position_data[start_pt:(start_pt + trial_length * n_repeats), :],
        (n_repeats, trial_length, position_data.shape[1])
    )

    return sensors, sensor_data, positions, position_data


def save_luke_data_to_txtfile(par_dir, f, pos, pos_data, sensors, sensor_data):
    object_labels = {'black': 1,  # BLACK_MANDRILL_BALL_HARD
                     'blue': 2,  # BLUE_MADRILL_BALL_MEDIUM
                     'red': 3,  # RED_MANDRILL_BALL_SOFT
                     'plushy': 4,  # PLUSH_TOY_SOFT
                     'large': 5,  # LARGE_SIZE_CHIRP_BALL
                     'medium': 6,  # MEDIUM_SIZE_CHIRP_BALL
                     'small': 7,  # SMALL_SIZE_CHIRP_BALL
                     }
    data_loc = Path(par_dir).parent / 'formatted' / 'sensor_data'
    for idx, s in enumerate(pos):
        with open(data_loc / Path(s + '.txt'), 'a') as ff:
            np.savetxt(ff, pos_data[:, :, idx])
    for idx, s in enumerate(sensors):
        with open(data_loc / 'object_info.txt', 'a') as ff:
            label = np.tile(int(object_labels[f.split('_')[0]]),
                            sensor_data.shape[1])
            np.savetxt(ff, label, fmt='%0.0f')
        with open(data_loc / Path(s + '.txt'), 'a') as ff:
            np.savetxt(ff, sensor_data[:, :, idx])


def load_deka_luke_csv(parent_dir, n_repeats, trial_length, str_filter=None,
                       save_to_txt=False, summary_plots=False):
    """Loads LUKE data from CSV and reshapes it to a numpy array sized
    n_repeats x trial_length (in samples) x n_sensors.
    Note that for data on 2020-05-27,
    n_repeats = 50, trial_length = 401, and n_sensors is 13 or 6 for
    'cutanteous' or 'position' sensors, respectively.

    :param parent_dir the directory (str) where .csv files live.
    :param n_repeats: number of times a particular task was repeated in file
    :param trial_length: number of samples (period) for requested task
    :param str_filter: a string to filter the .csv file names for analysis. for
                       example, 'grasp6' loads and formats/plots only files with
                       'grasp6' in the title.
    :param save_to_txt: Boolean flag to save data to txt file or not
    :param summary_plots: Boolean flag to plot a summary of the data

    :returns
    """

    start_pts = {
        'small_chirp_ball_grasp6_2020-05-27-13-41-42.csv': 495,
        'large_chirp_ball_grasp5_2020-05-27-10-21-35.csv': 1806
    }

    for f in os.listdir(parent_dir):
        if f.endswith('.csv'):

            # To only look at Grasp6
            if str_filter is not None:
                if str_filter not in f:
                    continue

            data = pd.read_csv(os.path.join(parent_dir, f), sep=' ',
                               skiprows=1)
            col_names = data.columns.values.tolist()

            time = data['timestamp'].to_numpy()
            time = time - time[0]  # Convert to time relative to start of file.

            # Find the point to start reshaping data using a particular CAN
            # frame that records the command sequence
            my_data = data[col_names[43]].to_numpy()

            if f in start_pts.keys():
                start_pt = start_pts[f]
            else:
                start_pt = np.where(my_data > 0)[0][0] - 85

            # # Can plot can_data to verify that trials are properly separated.
            # can_data = np.reshape(
            #     my_data[start_pt:(start_pt + trial_length * n_repeats)],
            #     (n_repeats, trial_length)
            # ).T

            # Reshape the sensor and position data
            sensors, sensor_data, positions, pos_data = \
                reshape_data(data, col_names, start_pt, trial_length, n_repeats)

            if save_to_txt:
                save_luke_data_to_txtfile(parent_dir, f, positions, pos_data,
                                          sensors, sensor_data)

            if summary_plots:
                fig, ax = plt.subplots(3, 2, figsize=(16, 8))
                for i in range(6):
                    ax[np.unravel_index(i, (3, 2))].plot(
                        time[:pos_data.shape[1]], pos_data[:, :, i].T)
                    ax[np.unravel_index(i, (3, 2))].set_title(positions[i])
                ax[2, 0].set_xlabel('time (s)')
                ax[2, 1].set_xlabel('time (s)')
                plt.suptitle(f)
                plt.tight_layout()
                plt.show()
                # plt.savefig(os.path.join(root,
                #                          'figures',
                #                          '{}_positions'.format(f[:-4]))
                #             )
                # plt.close(fig)

                sensor_to_plot = [0, 1, 5, 6, 11, 12]
                fig, ax = plt.subplots(3, 2, figsize=(16, 8))
                for i, s in enumerate(sensor_to_plot):
                    ax[np.unravel_index(i, (3, 2))].plot(
                        time[:sensor_data.shape[1]], sensor_data[:, :, s].T)
                    ax[np.unravel_index(i, (3, 2))].set_title(sensors[s])
                ax[2, 0].set_xlabel('time (s)')
                ax[2, 1].set_xlabel('time (s)')
                plt.suptitle(f)
                plt.tight_layout()
                plt.show()
                # plt.savefig(os.path.join(root,
                #                          'figures',
                #                          '{}_sensors'.format(f[:-4])))
                # plt.close(fig)


if __name__ == "__main__":
    # Load and format data from the 2020-05-27 collect
    # Parameters specific to the 2020-05-27 collect:
    n_reps = 50
    trial_len = 401

    data_dir = r'C:\Users\e30302\SRI International\AIE-INI - Documents\D' \
             r'ata\deka_data\20200527_data'

    load_deka_luke_csv(parent_dir=data_dir,
                       n_repeats=n_reps,
                       trial_length=trial_len,
                       str_filter='grasp6',
                       save_to_txt=False,
                       summary_plots=True)

"""Entry point to the program"""
import logging

import ini_estim.sensors.generic

logging.basicConfig(level=logging.INFO)
from ini_estim import encoders, nerves, sensors, simulators, stimulus

if __name__ == "__main__":
    # logging.getLogger().setLevel(logging.INFO)
    # Instantiate objects required for a simulation

    stimulator = stimulus.NeuralStimulator(
        stimulator_config=stimulus.ROUND_CUFF
    )
    encoder = encoders.Encoder(
        kind='linear',
        stimulator_params=stimulator.get_hardware_params()
    )
    my_nerve = nerves.Nerve(
        num_neurons=17,
        spontaneous_rate=1
    )

    # Create simulator and add sensors.
    simulator = simulators.SimulatorOld(encoder, stimulator, my_nerve)
    for e in simulator.electrodes:
        simulator.add_sensor(
            ini_estim.sensors.generic.PressureSensor(0.0, 1.0, 500.0, 0.1),
            e,
            1.0
        )
    simulator.run(duration=5)

    # score = simulator.finish()

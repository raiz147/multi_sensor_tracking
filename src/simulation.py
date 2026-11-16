
from dataclasses import dataclass

import numpy as np


@dataclass
class Sensor:
    name: str
    position: np.ndarray
    measurement_std: float



def simulate_multiple_targets(num_steps=100, dt=1.0):
    """
    Simulate two objects whose trajectories intersect
    at an angle around time step 40.

    State format: [x, y, vx, vy]
    """
    states = np.zeros((num_steps, 2, 4), dtype=float)

    # Target 1: approaches the intersection from the lower left.
    states[0, 0] = [-32.0, -16.0, 0.8, 0.4]

    # Target 2: approaches from the upper right.
    states[0, 1] = [32.0, 24.0, -0.8, -0.6]

    for step in range(1, num_steps):
        for target in range(2):
            previous = states[step - 1, target]

            states[step, target, 0] = (
                previous[0] + previous[2] * dt
            )

            states[step, target, 1] = (
                previous[1] + previous[3] * dt
            )

            states[step, target, 2] = previous[2]
            states[step, target, 3] = previous[3]

    return states




def generate_measurements(
    true_states,
    sensors,
    rng,
    detection_probability=0.90
):
    """
    Generate noisy sensor measurements.

    Each object has a probability of being detected
    by each sensor at each time step.
    """

    num_steps = true_states.shape[0]

    measurements = {
        sensor.name: []
        for sensor in sensors
    }

    for sensor in sensors:

        for step in range(num_steps):

            step_measurements = []

            for target_state in true_states[step]:

                # Randomly determine whether the
                # sensor detects this object.
                detected = (
                    rng.random() < detection_probability
                )

                if not detected:
                    continue

                noise = rng.normal(
                    loc=0.0,
                    scale=sensor.measurement_std,
                    size=2
                )

                measurement = (
                    target_state[:2] + noise
                )

                step_measurements.append(measurement)

            rng.shuffle(step_measurements)

            measurements[sensor.name].append(
                step_measurements
            )

    return measurements


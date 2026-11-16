
import numpy as np

from src.simulation import (
    Sensor,
    simulate_multiple_targets,
    generate_measurements
)

from src.tracking import (
    MultiObjectTracker,
    cluster_measurements
)

from src.plotting import plot_tracking_results


def main():

    rng = np.random.default_rng(seed=7)

    num_steps = 100
    dt = 1.0

    # --------------------------------------------------
    # 1. Configure sensors
    # --------------------------------------------------

    sensors = [
        Sensor(
            name="Sensor A",
            position=np.array([-10.0, -5.0]),
            measurement_std=1.2
        ),
        Sensor(
            name="Sensor B",
            position=np.array([25.0, 5.0]),
            measurement_std=2.0
        ),
        Sensor(
            name="Sensor C",
            position=np.array([10.0, 30.0]),
            measurement_std=0.7
        )
    ]

    # --------------------------------------------------
    # 2. Configure detection probability
    # --------------------------------------------------

    detection_probability = 1

    # --------------------------------------------------
    # 3. Simulate target movement
    # --------------------------------------------------

    true_states = simulate_multiple_targets(
        num_steps=num_steps,
        dt=dt
    )

    # --------------------------------------------------
    # 4. Generate sensor measurements
    # --------------------------------------------------

    measurements = generate_measurements(
        true_states,
        sensors,
        rng,
        detection_probability=detection_probability
    )

    # --------------------------------------------------
    # 5. Initialize multi-object tracker
    # --------------------------------------------------

    tracker = MultiObjectTracker(
        process_acceleration_std=0.2,
        gate_distance=10.0,
        max_missed=5,
        dt=dt
    )

    # --------------------------------------------------
    # 6. Collect measurements over initialization window
    # --------------------------------------------------

    initialization_window = 5

    initial_measurements = []

    for sensor in sensors:

        for step in range(initialization_window):

            initial_measurements.extend(
                measurements[sensor.name][step]
            )

    print("\nINITIALIZATION CHECK")
    print("--------------------")
    print(
        "Detection probability:",
        detection_probability
    )
    print(
        "Initialization window:",
        initialization_window
    )
    print(
        "Total collected measurements:",
        len(initial_measurements)
    )

    # --------------------------------------------------
    # 7. Cluster initial measurements
    # --------------------------------------------------

    initial_clusters = cluster_measurements(
        initial_measurements,
        cluster_distance=12.0
    )

    print(
        "Detected clusters:",
        len(initial_clusters)
    )

    # Initialize tracks.
    tracker.initialize(initial_clusters)

    print(
        "Initialized tracks:",
        len(tracker.tracks)
    )

    # --------------------------------------------------
    # 8. Map tracks to true targets for evaluation
    # --------------------------------------------------

    target_for_track = {}

    for track in tracker.tracks:

        distances = [
            np.linalg.norm(
                track.position -
                true_states[0, target_index, :2]
            )
            for target_index in range(true_states.shape[1])
        ]

        target_for_track[track.track_id] = int(
            np.argmin(distances)
        )

    # --------------------------------------------------
    # 9. Store initial estimates
    # --------------------------------------------------

    estimates = {
        track.track_id: [track.position.copy()]
        for track in tracker.tracks
    }

    # --------------------------------------------------
    # 10. Process remaining time steps
    # --------------------------------------------------

    for step in range(initialization_window, num_steps):

        measurements_this_step = {
            sensor.name: measurements[sensor.name][step]
            for sensor in sensors
        }

        tracker.update(
            measurements_this_step,
            sensors
        )

        # Store estimates for active tracks.
        for track in tracker.tracks:

            if track.track_id not in estimates:
                estimates[track.track_id] = []

            estimates[track.track_id].append(
                track.position.copy()
            )

    # --------------------------------------------------
    # 11. Calculate tracking performance
    # --------------------------------------------------

    print("\nTRACKING RESULTS")
    print("----------------")

    if not estimates:
        print("No tracking results were generated.")

    for track_id, positions in estimates.items():

        if track_id not in target_for_track:
            print(
                f"Track {track_id}: "
                "No evaluation mapping"
            )
            continue

        target_index = target_for_track[track_id]

        estimated_positions = np.array(positions)

        # Align estimates with the time steps they represent.
        true_positions = true_states[
            initialization_window - 1:
            initialization_window - 1 + len(estimated_positions),
            target_index,
            :2
        ]

        # Ensure matching array lengths.
        comparison_length = min(
            len(estimated_positions),
            len(true_positions)
        )

        estimated_positions = estimated_positions[
            :comparison_length
        ]

        true_positions = true_positions[
            :comparison_length
        ]

        errors = (
            estimated_positions - true_positions
        )

        position_rmse = np.sqrt(
            np.mean(np.sum(errors**2, axis=1))
        )

        print(
            f"Track {track_id} | "
            f"Target {target_index + 1} | "
            f"Position RMSE: {position_rmse:.3f}"
        )

    # --------------------------------------------------
    # 12. Plot tracking results
    # --------------------------------------------------

    estimates_for_plot = {
        track_id: np.array(positions)
        for track_id, positions in estimates.items()
        if len(positions) > 0
    }

    plot_tracking_results(
        true_states,
        estimates_for_plot
    )


if __name__ == "__main__":
    main()

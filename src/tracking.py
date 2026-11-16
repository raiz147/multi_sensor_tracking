
from dataclasses import dataclass

import numpy as np

from src.kalman_filter import KalmanFilter2D


@dataclass
class Track:
    track_id: int
    kalman_filter: KalmanFilter2D
    hits: int = 1
    missed: int = 0

    @property
    def position(self):
        return self.kalman_filter.state[:2]

    @property
    def state(self):
        return self.kalman_filter.state


def associate_measurements(
    tracks,
    measurements,
    gate_distance=10.0
):
    """
    Greedy nearest-neighbor association with distance gating.

    Returns:
        matches: list of (track_index, measurement_index)
    """
    candidates = []

    for track_index, track in enumerate(tracks):
        for measurement_index, measurement in enumerate(measurements):

            distance = np.linalg.norm(
                track.position - measurement
            )

            if distance <= gate_distance:
                candidates.append(
                    (distance, track_index, measurement_index)
                )

    # Process shortest candidate distances first.
    candidates.sort(key=lambda item: item[0])

    matches = []
    used_tracks = set()
    used_measurements = set()

    for distance, track_index, measurement_index in candidates:

        if track_index in used_tracks:
            continue

        if measurement_index in used_measurements:
            continue

        matches.append(
            (track_index, measurement_index)
        )

        used_tracks.add(track_index)
        used_measurements.add(measurement_index)

    return matches


def cluster_measurements(measurements, cluster_distance=6.0):
    """
    Group nearby measurements using a simple
    distance-based clustering algorithm.

    Returns one averaged position per cluster.
    """

    clusters = []

    for measurement in measurements:

        assigned = False

        for cluster in clusters:

            center = np.mean(cluster, axis=0)

            distance = np.linalg.norm(
                measurement - center
            )

            if distance <= cluster_distance:
                cluster.append(measurement)
                assigned = True
                break

        if not assigned:
            clusters.append([measurement])

    cluster_centers = []

    for cluster in clusters:
        center = np.mean(cluster, axis=0)
        cluster_centers.append(center)

    return cluster_centers


class MultiObjectTracker:

    def __init__(
        self,
        process_acceleration_std=0.2,
        gate_distance=10.0,
        max_missed=5,
        dt=1.0
    ):
        self.tracks = []
        self.next_track_id = 1

        self.process_acceleration_std = (
            process_acceleration_std
        )

        self.gate_distance = gate_distance
        self.max_missed = max_missed
        self.dt = dt

    def initialize(self, initial_measurements):
        """
        Initialize one track per measurement from the first sensor.

        Initial velocity is unknown, so it starts at zero.
        """
        for measurement in initial_measurements:

            initial_state = [
                measurement[0],
                measurement[1],
                0.0,
                0.0
            ]

            kalman_filter = KalmanFilter2D(
                initial_state=initial_state,
                dt=self.dt,
                process_acceleration_std=(
                    self.process_acceleration_std
                )
            )

            track = Track(
                track_id=self.next_track_id,
                kalman_filter=kalman_filter
            )

            self.tracks.append(track)
            self.next_track_id += 1

    def predict(self):
        for track in self.tracks:
            track.kalman_filter.predict()

    def process_sensor(
        self,
        measurements,
        measurement_std
    ):
        """
        Associate one sensor's measurements with existing tracks,
        then update matched Kalman filters.
        """
        matches = associate_measurements(
            self.tracks,
            measurements,
            self.gate_distance
        )

        assigned_track_indices = set()

        for track_index, measurement_index in matches:

            track = self.tracks[track_index]
            measurement = measurements[measurement_index]

            track.kalman_filter.update(
                measurement,
                measurement_std
            )

            assigned_track_indices.add(track_index)

        return assigned_track_indices

    def update(self, measurements_by_sensor, sensors):

        # Step 1: Predict every existing track.
        self.predict()

        # Step 2: Track which objects receive at
        # least one associated measurement.
        assigned_this_step = set()

        # Step 3: Process each sensor.
        for sensor in sensors:

            measurements = measurements_by_sensor[sensor.name]

            assigned = self.process_sensor(
                measurements,
                sensor.measurement_std
            )

            assigned_this_step.update(assigned)

        # Step 4: Update track lifecycle.
        for index, track in enumerate(self.tracks):

            if index in assigned_this_step:

                track.hits += 1
                track.missed = 0

            else:

                track.missed += 1

        # Step 5: Remove tracks missing for
        # too many consecutive time steps.
        self.tracks = [
            track
            for track in self.tracks
            if track.missed <= self.max_missed
        ]


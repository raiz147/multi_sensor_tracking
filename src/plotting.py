
import matplotlib.pyplot as plt


def plot_tracking_results(true_states, estimates):
    plt.figure(figsize=(11, 7))

    # Actual object trajectories.
    for target_index in range(true_states.shape[1]):
        plt.plot(
            true_states[:, target_index, 0],
            true_states[:, target_index, 1],
            linestyle="--",
            label=f"True Target {target_index + 1}"
        )

    # Estimated track trajectories.
    for track_id, positions in estimates.items():
        plt.plot(
            positions[:, 0],
            positions[:, 1],
            linewidth=2,
            label=f"Estimated Track {track_id}"
        )

        plt.scatter(
            positions[0, 0],
            positions[0, 1],
            marker="o"
        )

    plt.title("Multi-Object Tracking and Data Association")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    plt.tight_layout()
    plt.show()

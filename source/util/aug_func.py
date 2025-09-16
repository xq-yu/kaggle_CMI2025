import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R


def stretch_sequence(sequence: np.ndarray, stretch_factor=1) -> np.ndarray:
    """
    Stretch/compress the sequence in time.

    Args:
        sequence (np.ndarray [T, N]): The input sequence.
        stretch_factor (float, optional): Stretch factor. Defaults to 1.

    Returns:
        np.ndarray [T2, N]: The stretched/compressed sequence.
    """
    seq_len, num_features = sequence.shape

    # Create new time sampling points
    original_time = np.linspace(0, 1, seq_len)
    new_length = max(1, int(seq_len / stretch_factor))
    new_time = np.linspace(0, 1, new_length)

    if new_length == seq_len:
        return sequence

    # Interpolate each feature
    stretched_sequence = np.zeros((new_length, num_features))
    for i in range(num_features):
        interpolator = interp1d(
            original_time,
            sequence[:, i],
            kind="linear",
            bounds_error=False,
            fill_value="extrapolate",
        )
        stretched_sequence[:, i] = interpolator(new_time)

    return stretched_sequence
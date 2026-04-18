"""Deep-learning lead completion module.

Uses a 1-D / 2-D convolutional autoencoder to extend partial ECG leads
(2.5 s or 5 s recordings) to the full 10-second duration.  Includes
normalization helpers, model loading, and the main completion entry point.
"""

from __future__ import annotations

import torch.nn as nn
import torch
import numpy as np
from scipy import signal
from scipy.interpolate import interp1d

# --- Signal parameters ---
NUM_LEADS = 12
SIGNAL_LENGTH = 5000  # 10 seconds at 500 Hz
RESAMPLED_LENGTH = 512  # Model input length after resampling
SAMPLING_FREQ = 500  # Hz
LOW_CUTOFF_HZ = 0.05
HIGH_CUTOFF_HZ = 150.0
RESAMPLED_FREQ = 512  # Hz


class Convolution1D_layer(nn.Module):
    """1-D convolution applied independently to each of the 12 leads."""

    def __init__(self, in_f, out_f, device):
        super(Convolution1D_layer, self).__init__()
        self.f = out_f
        self.device = device
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=in_f, out_channels=out_f, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(num_features=out_f),
            nn.LeakyReLU(0.02),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        b = len(x)
        new_x = torch.tensor(np.zeros((b, self.f, NUM_LEADS, int(x.shape[-1] / 2))).astype("float32")).to(self.device)
        for i in range(NUM_LEADS):
            new_x[:, :, i, :] = self.conv(x[:, :, i, :])
        return new_x


class Deconvolution1D_layer(nn.Module):
    """1-D transposed convolution (upsampling) applied per lead."""

    def __init__(self, in_f, out_f, device):
        super(Deconvolution1D_layer, self).__init__()
        self.device = device
        self.f = out_f
        self.deconv = nn.Sequential(
            nn.ConvTranspose1d(in_channels=in_f, out_channels=out_f, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(num_features=out_f),
            nn.LeakyReLU(0.02),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        b = len(x)
        new_x = torch.tensor(np.zeros((b, self.f, NUM_LEADS, int(x.shape[-1] * 2))).astype("float32")).to(self.device)
        for i in range(NUM_LEADS):
            new_x[:, :, i, :] = self.deconv(x[:, :, i, :])
        return new_x


class Convolution2D_layer(nn.Module):
    """2-D convolution across leads and time simultaneously."""

    def __init__(self, in_f, out_f):
        super(Convolution2D_layer, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_f, out_channels=out_f, kernel_size=(13, 4), stride=(1, 2), padding=(6, 1)),
            nn.BatchNorm2d(num_features=out_f),
            nn.LeakyReLU(0.02),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        new_x = self.conv(x)
        return new_x


class Deconvolution2D_layer(nn.Module):
    """2-D transposed convolution (upsampling) across leads and time."""

    def __init__(self, in_f, out_f):
        super(Deconvolution2D_layer, self).__init__()
        self.f = out_f
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=in_f, out_channels=out_f, kernel_size=(13, 4), stride=(1, 2), padding=(6, 1)
            ),
            nn.BatchNorm2d(num_features=out_f),
            nn.LeakyReLU(0.02),
            # nn.Dropout(0.2)
        )

    def forward(self, x):
        new_x = self.deconv(x)
        return new_x


class Autoencoder_net(nn.Module):
    """Dual-path (1-D + 2-D) convolutional autoencoder for ECG completion.

    The encoder applies parallel 1-D (per-lead) and 2-D (cross-lead)
    convolutions, concatenates their feature maps, and the decoder
    mirrors this structure with skip connections from the encoder.
    """

    def __init__(self, device):
        super(Autoencoder_net, self).__init__()
        self.first_conv2D = Convolution2D_layer(1, 16)
        self.first_conv1D = Convolution1D_layer(1, 16, device)

        self.second_conv2D = Convolution2D_layer(16, 32)
        self.second_conv1D = Convolution1D_layer(16, 32, device)

        self.third_conv2D = Convolution2D_layer(32, 64)
        self.third_conv1D = Convolution1D_layer(32, 64, device)

        self.fourth_conv2D = Convolution2D_layer(64, 128)
        self.fourth_conv1D = Convolution1D_layer(64, 128, device)

        self.first_deconv1D = Deconvolution1D_layer(256, 128, device)
        self.first_deconv2D = Deconvolution2D_layer(256, 128)

        self.second_deconv1D = Deconvolution1D_layer(256, 64, device)
        self.second_deconv2D = Deconvolution2D_layer(256, 64)

        self.third_deconv1D = Deconvolution1D_layer(128, 32, device)
        self.third_deconv2D = Deconvolution2D_layer(128, 32)

        self.fourth_deconv1D = Deconvolution1D_layer(64, 1, device)
        self.fourth_deconv2D = Deconvolution2D_layer(64, 1)

        self.final_conv = nn.Sequential(
            nn.ConvTranspose2d(in_channels=1, out_channels=1, kernel_size=(13, 3), stride=(1, 1), padding=(6, 1)),
            nn.Tanh(),
        )

        self.transition_block = nn.Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=256, kernel_size=(13, 3), stride=(1, 1), padding=(6, 1)),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.02),
        )

    def forward(self, x):
        conv2D_1 = self.first_conv2D(x)
        conv1D_1 = self.first_conv1D(x)
        conv_1 = torch.concat((conv1D_1, conv2D_1), axis=1)
        # print("Conv1: ",conv1D_1.shape)
        # print("Conv1: ",conv2D_1.shape)

        conv2D_2 = self.second_conv2D(conv2D_1)
        conv1D_2 = self.second_conv1D(conv1D_1)
        # print("Conv2: ",conv1D_2.shape)
        # print("Conv2: ",conv2D_2.shape)
        conv_2 = torch.concat((conv1D_2, conv2D_2), axis=1)
        # print("Conv2: ",conv_2.shape)

        conv2D_3 = self.third_conv2D(conv2D_2)
        conv1D_3 = self.third_conv1D(conv1D_2)
        # print("Conv3: ",conv1D_3.shape)
        # print("Conv3: ",conv2D_3.shape)
        conv_3 = torch.concat((conv1D_3, conv2D_3), axis=1)
        # print("Conv3: ",conv_3.shape)

        conv2D_4 = self.fourth_conv2D(conv2D_3)
        conv1D_4 = self.fourth_conv1D(conv1D_3)
        # print("Conv4: ",conv1D_4.shape)
        # print("Conv4: ",conv2D_4.shape)
        conv_4 = torch.concat((conv1D_4, conv2D_4), axis=1)
        # print("Conv4: ",conv_4.shape)

        transition = self.transition_block(conv_4)
        # print("Transition: ", transition.shape)

        deconv2D_1 = self.first_deconv2D(transition)
        # print("Deconv 1: ",deconv2D_1.shape)
        deconv_1 = torch.concat((deconv2D_1, conv_3), axis=1)
        # print("Deconv 1 Concat: ",deconv_1.shape)

        deconv2D_2 = self.second_deconv2D(deconv_1)
        # print("Deconv 2: ",deconv2D_2.shape)
        deconv_2 = torch.concat((deconv2D_2, conv_2), axis=1)
        # print("Deconv 2 Concat: ",deconv_2.shape)

        deconv2D_3 = self.third_deconv2D(deconv_2)
        # print("Deconv 3: ",deconv2D_3.shape)
        deconv_3 = torch.concat((deconv2D_3, conv_1), axis=1)
        # print("Deconv 3 Concat: ",deconv_3.shape)

        deconv2D_4 = self.fourth_deconv2D(deconv_3)
        # print("Deconv 4: ",deconv2D_4.shape)

        out = self.final_conv(deconv2D_4)
        out = torch.squeeze(out, 1)
        return out


def linear_interpolation(signal: np.ndarray) -> np.ndarray:
    """Resample a signal to ``SIGNAL_LENGTH`` (5000) samples via linear interpolation.

    Parameters
    ----------
    signal : numpy.ndarray
        Input signal of arbitrary length.

    Returns
    -------
    numpy.ndarray
        Resampled signal with ``SIGNAL_LENGTH`` samples.
    """
    original_length = len(signal)
    new_length = SIGNAL_LENGTH

    # Create the original x-axis values
    x_original = np.linspace(0, original_length - 1, original_length)

    # Create the new x-axis values
    x_new = np.linspace(0, original_length - 1, new_length)

    # Perform linear interpolation
    f = interp1d(x_original, signal, kind="linear")
    interpolated_signal = f(x_new)

    return interpolated_signal


def denormalization(signal: np.ndarray, original_min: float, original_max: float) -> np.ndarray:
    """Reverse min-max normalization from [-1, 1] back to original scale.

    Parameters
    ----------
    signal : numpy.ndarray
        Normalized signal in the range [-1, 1].
    original_min : float
        Original minimum value used during normalization.
    original_max : float
        Original maximum value used during normalization.

    Returns
    -------
    numpy.ndarray
        Signal restored to its original amplitude range.
    """
    denormalized_signal = (signal + 1) * (original_max - original_min) / 2 + original_min
    return denormalized_signal


def normalization(Z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bandpass-filter, resample and normalize a multi-lead signal matrix.

    Applies a 4th-order Butterworth bandpass filter, resamples to
    ``RESAMPLED_LENGTH`` (512) samples, and normalizes to [-1, 1].

    Parameters
    ----------
    Z : numpy.ndarray
        Signal matrix of shape ``(num_samples, num_leads)``.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(normalized_matrix, scale_matrix)`` where *normalized_matrix*
        has shape ``(RESAMPLED_LENGTH, num_leads)`` and *scale_matrix*
        has shape ``(num_leads, 2)`` with ``[min, max]`` per lead.
    """
    new_Z = np.zeros((RESAMPLED_LENGTH, len(Z)))
    scale_Z = np.zeros((len(Z), 2))
    for i in range(len(Z)):
        mini = Z[:, i].min()
        maxi = Z[:, i].max()
        nyquist = 0.5 * SAMPLING_FREQ
        low_cutoff = LOW_CUTOFF_HZ / nyquist
        high_cutoff = HIGH_CUTOFF_HZ / nyquist
        new_sampling_frequency = RESAMPLED_FREQ
        original_sampling_frequency = SIGNAL_LENGTH
        b, a = signal.butter(4, [low_cutoff, high_cutoff], btype="band")
        filtered_signal = signal.lfilter(b, a, Z[i])
        resampled_signal, mini, maxi = normalization2(
            signal.resample(
                filtered_signal, int(len(filtered_signal) * (new_sampling_frequency / original_sampling_frequency))
            )
        )

        if np.all(np.isnan(resampled_signal)):
            resampled_signal = np.random.normal(0, 1, (RESAMPLED_LENGTH))
        new_Z[:, i] = resampled_signal
        scale_Z[i, :] = [mini, maxi]
    return (new_Z, scale_Z)


def normalization2(Z: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Min-max normalize a signal to the range [-1, 1].

    Parameters
    ----------
    Z : numpy.ndarray
        Input signal.

    Returns
    -------
    tuple[numpy.ndarray, float, float]
        ``(normalized_signal, min_value, max_value)``.
    """
    mini = Z.min()
    maxi = Z.max()
    return (-1 + ((Z - mini) * (2)) / (maxi - mini), mini, maxi)


def replace_random(array: np.ndarray, True_data: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Prepare an ECG matrix as model input with partial masking.

    Maps each lead's known time segment into a random-initialized
    tensor of shape ``(1, 12, 512)``. When ``True_data`` is ``True``,
    the full signal is used without masking.

    Parameters
    ----------
    array : numpy.ndarray
        Multi-lead signal matrix of shape ``(num_leads, num_samples)``.
    True_data : bool, optional
        If ``True``, fill the entire tensor with real data (no masking).

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(input_tensor, scale_matrix)`` ready for model inference.
    """
    if len(array) == 13:
        dic_split = {
            0: (0, 128),
            1: (0, 512),
            2: (0, 128),
            3: (128, 256),
            4: (128, 256),
            5: (128, 256),
            6: (256, 384),
            7: (256, 384),
            8: (256, 384),
            9: (384, 512),
            10: (384, 512),
            11: (384, 512),
        }
    else:
        dic_split = {
            0: (0, 256),
            1: (0, 256),
            2: (0, 256),
            3: (0, 256),
            4: (0, 256),
            5: (0, 256),
            6: (256, 512),
            7: (256, 512),
            8: (256, 512),
            9: (256, 512),
            10: (256, 512),
            11: (256, 512),
        }

    final_matrix = np.random.random((1, NUM_LEADS, RESAMPLED_LENGTH))
    array, scale = normalization(array)
    if not True_data:
        for i in range(NUM_LEADS):
            final_matrix[0, i, dic_split[i][0] : dic_split[i][1]] = array[dic_split[i][0] : dic_split[i][1], i]
    else:
        for i in range(NUM_LEADS):
            final_matrix[0, i, :] = array[:, i]
    return (final_matrix, scale)


def load_model(path: str, device: str) -> Autoencoder_net:
    """Load a pre-trained autoencoder from a ``.pth`` weights file.

    Parameters
    ----------
    path : str
        Path to the model weights file.
    device : str
        PyTorch device string (e.g. ``"cpu"`` or ``"cuda"``).

    Returns
    -------
    Autoencoder_net
        Model in evaluation mode.
    """
    model = Autoencoder_net(device)
    model.load_state_dict(torch.load(path, map_location=torch.device(device), weights_only=True))
    model.eval()
    return model


def completion_(ecg: dict[str, np.ndarray], path_model: str, device: str) -> dict[str, np.ndarray]:
    """Complete partial ECG leads to full 10-second recordings.

    Normalizes and resamples the input leads, runs them through the
    autoencoder, then denormalizes and resamples back to 5000 samples.

    Parameters
    ----------
    ecg : dict[str, numpy.ndarray]
        Extracted leads keyed by name.
    path_model : str
        Path to the ``.pth`` model weights.
    device : str
        PyTorch device string.

    Returns
    -------
    dict[str, numpy.ndarray]
        Completed leads with 5000 samples each.
    """
    model = load_model(path_model, device)
    if "IIc" in ecg.keys():
        dic_sorted = ["I", "IIc", "III", "AVL", "AVR", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    else:
        dic_sorted = ["I", "II", "III", "AVL", "AVR", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
    matrix_to_complete = np.zeros((NUM_LEADS, SIGNAL_LENGTH))
    for k in dic_sorted:
        matrix_to_complete[dic_sorted.index(k), :] = np.nan_to_num(ecg[k])
    matrix_to_complete = np.array(matrix_to_complete)
    ecg_to_complete, ecg_scale = replace_random(matrix_to_complete, True_data=False)
    inp = torch.tensor(np.expand_dims(ecg_to_complete, 1).astype("float32")).to(device)
    comp = model(inp).detach().numpy()
    ecg_complete = np.zeros((NUM_LEADS, SIGNAL_LENGTH))
    for l in range(NUM_LEADS):
        ecg_complete[l, :] = denormalization(linear_interpolation(comp[0, l, :]), ecg_scale[l, 0], ecg_scale[l, 1])

    ecg = {}
    for k in dic_sorted:
        if k == "IIc":
            ecg["II"] = ecg_complete[dic_sorted.index(k), :]
        else:
            ecg[k] = ecg_complete[dic_sorted.index(k), :]
    return ecg

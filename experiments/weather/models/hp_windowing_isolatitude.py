import healpix
import chealpix as chp
import torch


def get_isolatitude_windows_hp(nside):
    polar_idx = list(range(0, nside))
    current_idx = 0
    north_idxs = []
    north_eq_idxs = []
    south_eq_idxs = []
    south_idxs = []
    n_pixels = healpix.nside2npix(nside)
    for window_idx in polar_idx:
        north_idxs.append([current_idx + i for i in range(4 * (window_idx + 1))])
        current_idx += 4 * (window_idx + 1)

    for window_idx in range(nside):
        north_eq_idxs.append([current_idx + i for i in range(4 * nside)])
        current_idx += 4 * nside

    for window_idx in range(nside - 1):
        south_eq_idxs.append([current_idx + i for i in range(4 * nside)])
        current_idx += 4 * nside

    # nside 2, 0 -> 0

    # nside 3, 0 -> 1
    # nside 3, 1 -> 0
    for window in reversed(north_idxs):
        south_idxs.append([n_pixels - 1 - idx for idx in window])

    return north_idxs + north_eq_idxs + south_eq_idxs + south_idxs


def to_interspersed_windows(nside, max_size, window):
    n_pixels_in_window = len(window)
    n_sub_windows = n_pixels_in_window // max_size + 1
    nest_idxs = chp.ring2nest(nside, window)
    sub_windows = []
    for sub_idx in range(n_sub_windows):
        sub_windows.append(nest_idxs[sub_idx::n_sub_windows].tolist())
    return sub_windows


def flattened_interspersed(nside, max_window_size, windows):
    interspersed = [
        to_interspersed_windows(nside, max_window_size, window) for window in windows
    ]
    return [window for subwins in interspersed for window in subwins]


def pad_windows(max_window_size, windows):
    padded_windows = []
    current_padded = []
    for idx, window in enumerate(windows):
        fits_in_window = len(current_padded) + len(window) <= max_window_size
        if fits_in_window:
            current_padded.extend(window)
        if not fits_in_window:
            current_padded.extend(
                [current_padded[-1]] * (max_window_size - len(current_padded))
            )
            padded_windows.append(current_padded)
            current_padded = list(window)
        if idx == len(windows) - 1 and len(current_padded) > 0:
            current_padded.extend(
                [current_padded[-1]] * (max_window_size - len(current_padded))
            )
            padded_windows.append(current_padded)
    return padded_windows


def test_pad_windows(nside):
    max_window_size = 16
    hp_windows = get_isolatitude_windows_hp(nside)
    interspersed = flattened_interspersed(nside, max_window_size, hp_windows)
    padded_windows = pad_windows(max_window_size, interspersed)

    data = torch.rand((2, 3, healpix.nside2npix(nside), 48))
    data_pre = data.clone()
    indices = torch.tensor(padded_windows)

    # Extract windows
    windowed = data[:, :, indices, :]

    # Use windows to reconstruct original tensor
    new = torch.zeros(data.shape)
    new[:, :, indices, :] = windowed
    assert (new - data_pre).sum() == 0.0


def window_reverse(windows, window_size, D, N, device):
    window_size_d, window_size_hp = window_size
    nside = healpix.npix2nside(N)
    C = windows.shape[-1]

    hp_windows = get_isolatitude_windows_hp(nside)
    interspersed = flattened_interspersed(nside, window_size_hp, hp_windows)
    padded_windows = pad_windows(window_size_hp, interspersed)

    indices = torch.tensor(padded_windows, device=device)

    Nw, W = indices.shape

    B = int(windows.shape[0] / (D * N // (window_size_hp * window_size_d)))
    x = windows.view(B, D // window_size_d, Nw, window_size_d, W, -1)
    x = x.permute(0, 1, 3, 2, 4, 5)

    # B, Nd, Wd, Nw, W, C
    # 0   1   2   3  4  5
    x = x.contiguous().view(B, D, Nw, W, C)

    new = torch.zeros((B, D, N, C), device=device)
    new[:, :, indices, :] = x

    return new


def window_partition(x: torch.Tensor, window_size, device):
    window_size_d, window_size_hp = window_size

    nside = healpix.npix2nside(x.shape[2])
    hp_windows = get_isolatitude_windows_hp(nside)
    interspersed = flattened_interspersed(nside, window_size_hp, hp_windows)
    padded_windows = pad_windows(window_size_hp, interspersed)

    indices = torch.tensor(padded_windows, device=device)
    windowed = x[:, :, indices, :]

    B, D, Nw, W, C = windowed.shape
    x = windowed.view(B, D // window_size_d, window_size_d, Nw, W, C)

    # B, Nd, Wd, Nw, W, C
    # 0   1   2   3  4  5

    x = x.permute(0, 1, 3, 2, 4, 5)
    windows = x.contiguous().view(-1, window_size_d * window_size_hp, C)
    return windows

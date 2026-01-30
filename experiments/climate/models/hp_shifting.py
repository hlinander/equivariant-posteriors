import math

import torch
import numpy as np

# import healpix as hp
import chealpix as chp

# from lib.render_duck import insert_artifact
import lib.render_duck as rd


def get_attn_mask_from_mask(mask, window_size, window_partition):
    """Translates mask of shape (N) with different int values to attention mask of shape (nW,
    window_size, window_size) with values in {0,-100} suitable for attention module"""

    mask = mask[None, :, :, None]
    mask_windows = window_partition(mask, window_size)  # nW, window_size, 1
    mask_windows = mask_windows.squeeze()  # nW, window_size

    # JG: The following gives a tensor of shape (nW, window_size, window_size) it computes
    # inside each window (0th index) for each pixel pair if they lie in the same subwindow
    # (=0) or not (!=0)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    # JG: pixel pairs with non-matching subwindows are set to -100, matching pairs to 0 inside
    # the attention module, this gets added to the argument of softmax, so for -100 the
    # attention output is 0
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
        attn_mask == 0, float(0.0)
    )
    return attn_mask


class NoShift:
    def get_mask(self, window_partition):
        return None

    def shift(self, x):
        return x

    def shift_back(self, x):
        return x


class NestRollShift:
    def __init__(self, shift_size, input_resolution, window_size):
        self.shift_size = shift_size
        self.input_resolution = input_resolution
        self.window_size = window_size
        self.window_size_d, self.window_size_hp = window_size

    def get_mask(self, window_partition):
        # calculate attention mask for SW-MSA
        D, N = self.input_resolution
        img_mask = torch.zeros(D, N)
        # JG: These slices select points inside full windows and partly split windows,
        # cf. Figure 4 in the SWIN paper. For healpy, there are just three cases
        hp_slices = (
            slice(0, -self.window_size_hp),
            slice(-self.window_size_hp, -self.window_size_hp // 2),
            slice(-self.window_size_hp // 2, None),
            # slice(-self.window_size_hp//2, -1),
        )
        d_slices = (
            slice(0, -2),
            slice(-2, None),
            slice(-2, None),
        )
        # JG: In this loop, the three different subwindows A, B, ... from Figure 4 are given
        # different numbers 0,1,3
        cnt = 0
        for d_slice, hp_slice in zip(d_slices, hp_slices):
            img_mask[d_slice, hp_slice] = cnt
            cnt += 1

        attn_mask = get_attn_mask_from_mask(
            img_mask, self.window_size, window_partition
        )
        return attn_mask

    def shift(self, x):
        # breakpoint()
        return torch.roll(
            x, shifts=[-self.window_size_d // 2, -self.window_size_hp // 2], dims=[1, 2]
        )

    def shift_back(self, x):
        # return torch.roll(x, shifts=self.shift_size, dims=2)
        return torch.roll(
            x, shifts=[self.window_size_d // 2, self.window_size_hp // 2], dims=[1, 2]
        )


class NestGridShift:
    def __init__(self, nside, base_pix, window_size):
        assert (
            base_pix == 8
        ), "NestGridShift is currently only implemented for 8 base pixels"

        self.nside = nside
        self.ws = window_size
        self.base_pix = base_pix

        self.npix = self.base_pix * self.nside**2
        self.n_windows = self.npix // self.ws
        self.base_pix_len = (self.npix // self.base_pix) // self.ws
        self.hws = self.ws // 2  # half window size
        self.qws = self.hws // 2  # quarter window size

        self.shift_idcs = self._get_shifted_idcs_dir1()
        self.shift_idcs = self.shift_idcs[self._get_shifted_idcs_dir2()]
        self._validate_shift_result()

        self.back_shift_idcs = self._get_inverse_index_map(self.shift_idcs)

    def _validate_shift_result(self):
        # check whether shifting is permutation of pixels
        diff = torch.max(torch.abs(self.shift_idcs.sort()[0] - torch.arange(self.npix)))
        assert (
            diff == 0
        ), f"shift validation failed for nside={self.nside}, window_size={self.ws}"

    def _log4(self, x):
        return int((math.log(x) / math.log(4)))

    def _get_scale(self, idx):
        """Return the scale of pixel position idx where idx should be the first pixel of a window

        The scale describes how far up the nested structure we have to go to jump over the previous
        block
        """
        assert idx % self.ws == 0
        w_idx = idx // self.ws
        scale = self.base_pix_len
        while w_idx % scale != 0:
            scale = scale // 4
        return self._log4(scale)

    def _get_offset_dir1(self, idx):
        """Returns offset to end of window
        Args:
        idx (int): beginning of window
        """

        # offsets between base pixels to join across base pixel borders
        assert idx % self.ws == 0

        BASE_PIX_OFFSETS = {0: 2, 1: 2, 2: 2, 3: 6, 4: 3, 5: 3, 6: 3, 7: 3}

        while True:
            scale = self._get_scale(idx)
            idx -= self.ws * 4**scale
            if scale >= self._get_scale(idx):
                break

        offset = sum([self.ws * (4**power) for power in range(0, scale + 1)])

        # if we are at a base pixel boundary, special treatment is needed
        if scale == self._log4(self.base_pix_len):
            # take jump over base pixel back
            idx += self.ws * 4**scale
            offset -= self.base_pix_len * self.ws

            base_pix = idx // (self.base_pix_len * self.ws)  # current base pixel
            # use offset from dict:
            offset += BASE_PIX_OFFSETS[base_pix] * self.base_pix_len * self.ws

        return offset

    def _test_get_offset_dir1(self):
        if self.base_pix_len > 32:
            assert self._get_offset_dir1(2 * self.ws) // self.ws == 1
            assert self._get_offset_dir1(3 * self.ws) // self.ws == 1
            assert self._get_offset_dir1(6 * self.ws) // self.ws == 1
            assert self._get_offset_dir1(7 * self.ws) // self.ws == 1
            assert self._get_offset_dir1(8 * self.ws) // self.ws == 5
            assert self._get_offset_dir1(9 * self.ws) // self.ws == 5
            assert self._get_offset_dir1(10 * self.ws) // self.ws == 1
            assert self._get_offset_dir1(11 * self.ws) // self.ws == 1
            assert self._get_offset_dir1(12 * self.ws) // self.ws == 5
            assert self._get_offset_dir1(32 * self.ws) // self.ws == 21
        assert self._get_offset_dir1(0) // (self.ws * self.base_pix_len) == 2

    def _get_shifted_idcs_dir1(self):
        """Returns indices of pixels shifted consistently by half a window size in direction 1"""

        ws = self.ws
        hws = self.hws

        result = torch.zeros(self.npix, dtype=torch.int64)

        for idx in range(self.n_windows):
            first = idx * ws
            os = self._get_offset_dir1(first)

            # first half-window
            result[first : first + hws] = torch.arange(first - os - hws, first - os)

            # second half-window
            result[first + hws : first + ws] = torch.arange(first, first + hws)

        result = result % self.npix

        return result

    def _test_shifted_idcs_dir1(self):
        shift_idcs = self._get_shifted_idcs_dir1()
        # check whether shifting is permutation of pixels
        assert torch.max(torch.abs(shift_idcs.sort()[0] - torch.arange(self.npix))) == 0

    def _get_offset_dir2(self, idx):
        """returns offset to the end of a window
        Args:
        idx (int): beginning of window
        """
        assert idx % self.ws == 0

        BASE_PIX_OFFSETS = {0: 3, 1: 3, 2: 3, 3: 3, 4: 3, 5: 3, 6: 3, 7: 3}

        scale = self._get_scale(idx)

        while (idx % (self.ws * 4 ** (scale + 1))) // (self.ws * 4**scale) == 2:
            idx -= 2 * self.ws * 4**scale
            scale = self._get_scale(idx)

        offset = sum([2 * self.ws * 4**power for power in range(0, scale)])

        # if we are at a base pixel boundary, special treatment is needed
        if scale == self._log4(self.base_pix_len):
            base_pix = idx // (self.base_pix_len * self.ws)  # current base pixel
            # use offset from dict:
            offset += BASE_PIX_OFFSETS[base_pix] * self.base_pix_len * self.ws

        return offset

    def _test_get_offset_dir2(self):
        if self.base_pix_len > 44:
            assert self.get_offset_dir2(4 * self.ws) == 2 * self.ws
            assert self.get_offset_dir2(12 * self.ws) == 2 * self.ws
            assert self.get_offset_dir2(16 * self.ws) == 10 * self.ws
            assert self.get_offset_dir2(20 * self.ws) == 2 * self.ws
            assert self.get_offset_dir2(24 * self.ws) == 10 * self.ws
            assert self.get_offset_dir2(28 * self.ws) == 2 * self.ws
            assert self.get_offset_dir2(36 * self.ws) == 2 * self.ws
            assert self.get_offset_dir2(44 * self.ws) == 2 * self.ws

    def _get_shifted_idcs_dir2(self):
        """Returns indices of pixels shifted consistently by half a window size in direction 2"""

        ws = self.ws
        hws = self.hws
        qws = self.qws

        result = torch.zeros(self.npix, dtype=torch.int64)
        for idx in range(self.n_windows):
            first = idx * ws
            os = self._get_offset_dir2(first)

            # first quarter window
            result[first : first + qws] = torch.arange(
                first - os - hws - qws, first - os - hws
            )

            # second quarter window
            result[first + qws : first + hws] = torch.arange(first, first + qws)

            # third quarter window
            result[first + hws : first + hws + qws] = torch.arange(
                first - os - qws, first - os
            )

            # fourth quarter window
            result[first + hws + qws : first + ws] = torch.arange(
                first + hws, first + hws + qws
            )

        result = result % self.npix

        return result

    def _test_shifted_idcs_dir2(self):
        shift_idcs = self._get_shifted_idcs_dir2()
        # check whether shifting is permutation of pixels
        assert torch.max(torch.abs(shift_idcs.sort()[0] - torch.arange(self.npix))) == 0

    def _get_inverse_index_map(self, idcs):
        return torch.sort(idcs)[1]

    def get_mask(self, get_attn_mask=True):
        # base pixels which are masked in first row and last column
        MASKED_BASE_PIX = [4, 5, 6, 7]

        # base pixels which are masked in first half window because of carry-over
        LEFT_CARRY_OVER_BASE_PIX = [0, 1, 2, 3]

        ws = self.ws
        hws = self.hws
        qws = self.qws

        mask = torch.zeros(self.npix)

        def right_mask_subset(first, size, mask_value):
            if size == ws:
                mask[first : first + qws] = mask_value
                mask[first + hws : first + hws + qws] = mask_value
            else:
                right_mask_subset(first, size // 4, mask_value)
                right_mask_subset(first + 2 * size // 4, size // 4, mask_value)

        def left_mask_subset(first, size, mask_value):
            if size == ws:
                mask[first : first + hws] = mask_value
            else:
                left_mask_subset(first, size // 4, mask_value)
                left_mask_subset(first + size // 4, size // 4, mask_value)

        for b, co in zip(MASKED_BASE_PIX, LEFT_CARRY_OVER_BASE_PIX):
            left_mask_subset(b * self.base_pix_len * ws, self.base_pix_len * ws, b + 1)
            right_mask_subset(
                b * self.base_pix_len * ws,
                self.base_pix_len * ws,
                b + 1 + len(MASKED_BASE_PIX),
            )
            first_co = co * self.base_pix_len * ws
            mask[first_co : first_co + qws] = b + 1

        if get_attn_mask:
            return get_attn_mask_from_mask(mask, ws)
        else:
            return mask

    def shift(self, x):
        return x[:, self.shift_idcs, ...].contiguous()

    def shift_back(self, x):
        return x[:, self.back_shift_idcs, ...].contiguous()


class RingShift:
    def __init__(self, nside, base_pix, window_size, shift_size, input_resolution):
        self.nside = nside
        self.base_pix = base_pix
        self.npix = base_pix * self.nside**2
        self.ws = window_size
        self.shift_size = shift_size
        self.window_size_d, self.window_size_hp = window_size
        self.input_resolution = input_resolution
        self.shift_size_d = self.window_size_d // 2
        self.shift_size_hp = shift_size  # self.window_size_hp // 2

        self.shift_idcs, self.mask = self._get_shifted_idcs_and_mask()
        self._validate_shift_result()
        self.back_shift_idcs = self._get_inverse_index_map(self.shift_idcs)

    def _get_shifted_idcs_and_mask(self):
        """shift hp image by converting to ring ordering and shifting there, then converting back to
        nest ordering

        """
        D, N = self.input_resolution

        ring_idcs = np.arange(12 * self.nside**2)
        # [0, 1, 2, 3, ...]
        shifted_ring_idcs = np.roll(ring_idcs, -self.shift_size_hp)
        # [2, 3, ..., 0, 1]
        shifted_ring_idcs_in_nest = chp.ring2nest(self.nside, shifted_ring_idcs)
        # [2_n, 3_n, ...,0_n, 1_n]

        # so far, this would return the image in ring indices, convert back to nested:
        nest_idcs = np.arange(self.npix)
        # chp.nest2ring()
        # nest_idcs_in_ring = hp.nest2ring(self.nside, nest_idcs)
        nest_idcs_in_ring = chp.nest2ring(self.nside, nest_idcs)
        result = shifted_ring_idcs_in_nest[nest_idcs_in_ring]

        mask = np.zeros((D, N))
        # mask[-self.shift_size :] = 1

        # H, W = self.input_resolution
        # img_mask = torch.zeros((1, D, N, 1))  # 1 H W 1
        d_slices = (
            # slice(0, -self.window_size_d),
            slice(-self.window_size_d, -self.shift_size_d),
            slice(-self.shift_size_d, None),
        )
        n_slices = (
            slice(0, -self.window_size_hp),
            slice(-self.window_size_hp, -self.shift_size_hp),
            slice(-self.shift_size_hp, None),
        )
        cnt = 0
        # Every depth needs a mask for the HP ring shift
        for d in range(0, D, self.window_size_d):
            for n in n_slices:
                # Every depth in the same window gets the same mask id
                for d_idx in range(self.window_size_d):
                    mask[d + d_idx, n] = cnt
                # mask[d, n] = cnt
                cnt += 1

        multiplier = cnt
        # The final
        for d_idx, d in enumerate(d_slices):
            mask[d, :] += (d_idx + 1) * multiplier
            # print("{} : {}", d, multiplier)
            # for n in n_slices:
            # mask[d, n] = cnt
            # cnt += 1
        # ----------------------
        # NOTE: Added following lines to get to work D=1 CASE
        # Double check everything works as intended
        if D > 1:
            assert mask[-1, 0] != mask[-2, 0]
            assert mask[-1, -self.shift_size_hp - 1] != mask[-1, -1]
        # ----------------------
        for d_idx in range(D):
            mask[d_idx, :] = mask[d_idx, nest_idcs_in_ring]

        # if rd.LAST_MODEL_ID is not None:
        #     print("Trying to insert artifact")
        #     index_check = np.array(shifted_ring_idcs, dtype=np.float32)
        #     index_check = index_check[nest_idcs_in_ring]
        #     np.save("/tmp/index_debug.npy", index_check)
        #     rd.ensure_duck(rd.LAST_RUN_CONFIG)
        #     rd.insert_model_with_model_id(rd.LAST_RUN_CONFIG, rd.LAST_MODEL_ID)
        #     rd.insert_artifact(
        #         rd.LAST_MODEL_ID, "index_debug.npy", "/tmp/index_debug.npy"
        #     )
        #     rd.sync(rd.LAST_RUN_CONFIG)
        # exit(0)
        #     mask_f32 = np.array(mask, dtype=np.float32)
        #     np.save("/tmp/mask_debug.npy", mask_f32)
        #     rd.ensure_duck(rd.LAST_RUN_CONFIG)
        #     rd.insert_model_with_model_id(rd.LAST_RUN_CONFIG, rd.LAST_MODEL_ID)
        #     rd.insert_artifact(
        #         rd.LAST_MODEL_ID, "mask_debug_f32_fixed_nest.npy", "/tmp/mask_debug.npy"
        #     )
        #     rd.sync(rd.LAST_RUN_CONFIG)
        #     print("Saved debug artifact")

        # exit(0)
        # mask_windows = window_partition(
        #     img_mask, self.window_size
        # )  # nW, window_size, window_size, 1
        # mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
        # attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        # attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
        #     attn_mask == 0, float(0.0)
        # )

        # mask = get_attn_mask_from_mask(mask, self.window_size)
        mask = torch.tensor(mask, dtype=torch.int64)
        result = torch.tensor(result, dtype=torch.int64)

        return result, mask

    def _validate_shift_result(self):
        # check whether shifting is permutation of pixels
        diff = torch.max(torch.abs(self.shift_idcs.sort()[0] - torch.arange(self.npix)))
        assert (
            diff == 0
        ), f"shift validation failed for nside={self.nside}, window_size={self.ws}"

    def _get_inverse_index_map(self, idcs):
        return torch.sort(idcs)[1]

    def get_mask(self, window_partition, get_attn_mask=True):
        mask = self.mask
        if get_attn_mask:
            return get_attn_mask_from_mask(mask, self.ws, window_partition)
        else:
            return mask

    def shift(self, x):
        return torch.roll(
            x[:, :, self.shift_idcs, ...].contiguous(),
            shifts=[-self.window_size_d // 2],
            dims=[1],
        )

    def shift_back(self, x):
        return torch.roll(
            x[:, :, self.back_shift_idcs, ...].contiguous(),
            shifts=[self.window_size_d // 2],
            dims=[1],
        )

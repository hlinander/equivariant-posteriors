import numpy as np
import torch
from dataclasses import dataclass
from pathlib import Path
from lib.dataspec import DataSpec

import experiments.x86extractor.extract as extract
from lib.data_utils import create_sample_legacy
from lib.serialization import serialize_human


@dataclass
class DataElfConfig:
    path: str
    as_seq: bool

    def serialize_human(self):
        return serialize_human(self.__dict__)


class DataElf(torch.utils.data.Dataset):
    def __init__(self, data_config: DataElfConfig):
        self.config = data_config
        root_path = Path(data_config.path)
        files = root_path.rglob("*")
        segments = []
        types = []
        for file in files:
            if file.is_file():
                segs = extract.process_elf_file(file)
                segs += extract.process_pe_file(file)
                for data, T in segs:
                    off = 0
                    while off < len(data):
                        part = data[off : off + 0x1000]
                        part = (
                            np.frombuffer(part, dtype=np.uint8).astype(np.float32)
                            / 255.0
                        )
                        # self.segments += [(part, T)]
                        segments += [part]
                        types += [0 if T == "CODE" else 1]
                        off += 0x1000
        segments = [seg for seg in segments if len(seg) == extract.PAGE_SIZE]
        self.segments = np.stack(segments)
        self.types = np.array(types, dtype=np.long)

    @staticmethod
    def data_spec(config: DataElfConfig):
        if config.as_seq:
            input_shape = [0x1000 // 16, 16]
        else:
            input_shape = [0x1000]
        return DataSpec(
            input_shape=torch.Size(input_shape),
            output_shape=torch.Size([2]),
            target_shape=torch.Size([2]),
        )

    def __getitem__(self, idx):
        data = self.segments[idx]
        type = self.types[idx]
        # fdata = np.float32([it / 0xFF for it in data])
        # fdata = np.array(data, dtype=np.float32) / 255.0
        if self.config.as_seq:
            data = data.reshape(-1, 16)
        return create_sample_legacy(data, type, idx)

    def __len__(self):
        return len(self.segments)

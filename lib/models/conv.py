import math
import torch
from dataclasses import dataclass
from lib.dataspec import DataSpec


@dataclass(frozen=True)
class ConvConfig:
    def serialize_human(self):
        return self.__dict__


class Conv(torch.nn.Module):
    def __init__(self, config: ConvConfig, data_spec: DataSpec):
        super().__init__()
        self.config = config
        self.c1 = torch.nn.Conv2d(3, 96, kernel_size=3)
        self.c2 = torch.nn.Conv2d(96, 96, kernel_size=3)
        self.c3 = torch.nn.Conv2d(96, 96, kernel_size=3, stride=2)
        self.c4 = torch.nn.Conv2d(96, 192, kernel_size=3)
        self.c5 = torch.nn.Conv2d(192, 192, kernel_size=3)
        self.c6 = torch.nn.Conv2d(192, 192, kernel_size=3, stride=2)
        self.c7 = torch.nn.Conv2d(192, data_spec.output_shape[-1], kernel_size=1)

        torch.nn.init.normal_(self.c1.weight, 0.0, std=math.sqrt(1.0 / (3 * 3 * 3)))
        torch.nn.init.normal_(self.c1.bias, 0.0, std=math.sqrt(1e-7))

        torch.nn.init.normal_(self.c2.weight, 0.0, std=math.sqrt(1.0 / (96 * 3 * 3)))
        torch.nn.init.normal_(self.c2.bias, 0.0, std=math.sqrt(1e-7))

        torch.nn.init.normal_(self.c3.weight, 0.0, std=math.sqrt(1.0 / (96 * 3 * 3)))
        torch.nn.init.normal_(self.c3.bias, 0.0, std=math.sqrt(1e-7))

        torch.nn.init.normal_(self.c4.weight, 0.0, std=math.sqrt(1.0 / (96 * 3 * 3)))
        torch.nn.init.normal_(self.c4.bias, 0.0, std=math.sqrt(1e-7))

        torch.nn.init.normal_(self.c5.weight, 0.0, std=math.sqrt(1.0 / (192 * 3 * 3)))
        torch.nn.init.normal_(self.c5.bias, 0.0, std=math.sqrt(1e-7))

        torch.nn.init.normal_(self.c6.weight, 0.0, std=math.sqrt(1.0 / (192 * 3 * 3)))
        torch.nn.init.normal_(self.c6.bias, 0.0, std=math.sqrt(1e-7))

        torch.nn.init.normal_(self.c7.weight, 0.0, std=math.sqrt(1.0 / (192 * 3 * 3)))
        torch.nn.init.normal_(self.c7.bias, 0.0, std=math.sqrt(1e-7))

    def forward(self, x):
        x = x.reshape(-1, 3, 32, 32)
        x = self.c1(x)
        x = torch.nn.functional.tanh(x)

        x = self.c2(x)
        x = torch.nn.functional.tanh(x)

        x = self.c3(x)
        x = torch.nn.functional.tanh(x)

        x = self.c4(x)
        x = torch.nn.functional.tanh(x)

        x = self.c5(x)
        x = torch.nn.functional.tanh(x)

        x = self.c6(x)
        x = torch.nn.functional.tanh(x)

        x = self.c7(x)
        x = torch.nn.functional.tanh(x)

        logits = torch.nn.functional.avg_pool2d(x, 4)
        logits = logits.squeeze()
        return dict(logits=logits, predictions=torch.softmax(logits.detach(), dim=-1))


# def create_small_conv_model(input_shape, model_params=None):
#     WD = 1e-3
#     kreg = None  # regularizers.l1(WD)
#     model = Sequential(
#         [
#             # keras.layers.Dropout(0.2, input_shape=(32, 32, 3)),
#             Conv2D(96, kernel_size=5, input_shape=input_shape, kernel_regularizer=kreg),
#             keras.layers.BatchNormalization(axis=-1),
#             ReLU(),
#             MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
#             # keras.layers.Dropout(0.5),
#             Conv2D(40, kernel_size=5, kernel_regularizer=kreg),
#             keras.layers.BatchNormalization(axis=-1),
#             ReLU(),
#             Conv2D(192, kernel_size=1, kernel_regularizer=kreg),
#             keras.layers.BatchNormalization(axis=-1),
#             ReLU(),
#             Conv2D(10, kernel_size=1, kernel_regularizer=kreg),
#             keras.layers.BatchNormalization(axis=-1),
#             ReLU(),
#             GlobalAveragePooling2D(),
#             Softmax(),
#         ]
#     )
#     # model.compile(loss=keras.losses.categorical_crossentropy,
#     #             optimizer=keras.optimizers.Adam(learning_rate=lr),#, amsgrad=True),
#     #             metrics=['accuracy'])
#     # model.summary()
#     return model

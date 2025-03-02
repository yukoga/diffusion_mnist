# Copyright 2023 yukoga
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class Dataset():
    def __init__(self, is_train=True):
        self.train = is_train

    def __call__(self):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1))
        ])

        return datasets.MNIST(
            root='data',
            train=self.train,
            download=True,
            transform=transform
        )


class MNISTDataLoader():
    def __init__(self):
        pass

    def __call__(self, is_train=True, batch_size=128, shuffle=True):
        dataset = Dataset(is_train=is_train)
        return DataLoader(
            dataset=dataset(),
            batch_size=batch_size,
            shuffle=shuffle)


if __name__ == '__main__':
    pass

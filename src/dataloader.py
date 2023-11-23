import torch
import os
from dataset import ChildrenDentalRadiographsDataset
from torchvision import transforms as T

TRAIN_PATH="train image path"
ANNOTATIONS_PATH="train mask path"

TEST_PATH="test image path"
TEST_ANNOTATIONS_PATH="test mask path"


def get_transform(train):
    transforms = []
    transforms.append(T.PILToTensor())
    transforms.append(T.ConvertImageDtype(torch.float))
    if train:
        transforms.append(T.RandomHorizontalFlip(p=.5))
    return T.Compose(transforms)


train_ds = ChildrenDentalRadiographsDataset(root=os.path.join(TRAIN_PATH, "images"),
                                            annotation=ANNOTATIONS_PATH,
                                            transforms=get_transform(train=True))

val_ds = ChildrenDentalRadiographsDataset(root=os.path.join(TEST_PATH, "images"),
                                          annotation=TEST_ANNOTATIONS_PATH,
                                          transforms=get_transform(train=False))
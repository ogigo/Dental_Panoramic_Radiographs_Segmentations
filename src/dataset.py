import os
import torch
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from torchvision import transforms as T



class ChildrenDentalRadiographsDataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.label_map = {
            1: "caries",
            2: "periapical infection",
            3: "pulpitis",
            4: "deep sulcus",
            5: "dental developmental abnormalities",
            6: "others"
        }
        self.label_map2 = {
            "caries": 1,
            "periapical infection": 2,
            "pulpitis": 3,
            "deep sulcus": 4,
            "dental developmental abnormalities": 5,
            "others": 6
        }

    def __getitem__(self, idx):
        coco = self.coco
        image_id = self.ids[idx]
        ann_ids = coco.getAnnIds(imgIds=image_id)
        coco_annotation = coco.loadAnns(ann_ids)
        path = coco.loadImgs(image_id)[0]["file_name"]
        img = Image.open(os.path.join(self.root, path)).convert("RGB")

        num_objs = len(coco_annotation)

        boxes = []
        labels = []
        masks = []
        areas = []

        for i in range(num_objs):
            xmin = coco_annotation[i]['bbox'][0]
            ymin = coco_annotation[i]['bbox'][1]
            xmax = xmin + coco_annotation[i]['bbox'][2]
            ymax = ymin + coco_annotation[i]['bbox'][3]
            boxes.append([xmin, ymin, xmax, ymax])

            category_id = coco_annotation[i]['category_id']
            label = self.label_map2[self.label_map[category_id]]
            labels.append(label)
            mask = coco.annToMask(coco_annotation[i])
            masks.append(mask)
            areas.append(coco_annotation[i]['area'])

        masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        areas = torch.as_tensor(areas, dtype=torch.float32)
        image_id = torch.tensor([image_id])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": image_id,
            "area": areas,
            "iscrowd": iscrowd
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)
    
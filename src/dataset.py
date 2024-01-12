# -*- coding: utf-8 -*-
# Import
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
import random

import numpy as np

import matplotlib.pyplot as plt

from PIL import Image
from typing import Any, Callable, Dict, List, Optional, Union, Tuple


from torchvision import datasets, transforms

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
class CustomCityscapes(datasets.Cityscapes):
    """
    Custom dataset class for Cityscapes dataset, extending torchvision's Cityscapes.

    Args:
        root (str): Root directory of the Cityscapes dataset.
        split (str): The split of the dataset (e.g., 'train', 'val', 'test').
        mode (str): The mode of the dataset ('fine' or 'coarse').
        target_type (str): The type of target to return ('semantic' or 'instance').
        transform (callable): A function/transform that takes in an PIL image and returns a transformed version.
        target_transform (callable): A function/transform that takes in the target and transforms it.

    Methods:
        plot_image(index: int = None) -> None: Plots an RGB image from the dataset.
        plot_segmentation(index: int = None) -> None: Plots the segmentation of an image from the dataset.
        plot_image_and_segmentation(index: Optional[int] = None) -> None: Plots both the RGB image and its segmentation.

    Example:
        dataset = CustomCityscapes('../data/', split='train', mode='fine', target_type='semantic', transform=transforms.ToTensor())
    """
    def __init__(self, root: str, split: str = 'train', mode: str = 'fine', target_type: str = 'semantic', transform=None, target_transform=None):
        super(CustomCityscapes, self).__init__(root, split=split, mode=mode, target_type=target_type, transform=transform, target_transform=target_transform)
        
        self.__init_mapping_class__()
        
    def __init_mapping_class__(self):
        self.ignore_index = 255
        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        self.valid_classes = [self.ignore_index, 7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33]
        self.class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic_light', \
                            'traffic_sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', \
                            'train', 'motorcycle', 'bicycle']
        self.class_map = dict(zip(self.valid_classes, range(len(self.valid_classes))))
        self.n_classes = len(self.valid_classes)

        self.colors = [
            [0, 0, 0], [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153], [153, 153, 153],
            [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152], [0, 130, 180], [220, 20, 60], [255, 0, 0],
            [0, 0, 142], [0, 0, 70], [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]
        ]
        self.label_colours = dict(zip(range(self.n_classes), self.colors))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        image = Image.open(self.images[index]).convert('RGB')

        targets: Any = []
        for i, t in enumerate(self.target_type):
            if t == 'polygon':
                target = self._load_json(self.targets[index][i])
            else:
                target = Image.open(self.targets[index][i])
            targets.append(target)
        target = tuple(targets) if len(targets) > 1 else targets[0]

        if self.transforms is not None:
            transformed=self.transform(image=np.array(image), mask=np.array(target))   
            image, target = transformed['image'],transformed['mask']
        return image, target
    
    
    def plot_image(self, index: Optional[int] = None) -> None:
        """
        Plots an RGB image from the dataset.

        Args:
            index (Optional[int]): Index of the image to plot. If None, a random image will be selected.

        Returns:
            None
        """
        if index is None:
            index = random.randint(0, len(self) - 1)
                
        image, _ = self.__getitem__(index)
        
        if self.transform is not None:
            image_np = image.detach().numpy().transpose(1, 2, 0)

        else:  
            # Convert PIL image to numpy array
            image_np = np.array(image)
    
        # Display the image
        plt.imshow(image_np)
        plt.title('RGB Image')
        plt.show()

    def plot_segmentation(self, index: Optional[int] = None) -> None:
        """
        Plots the segmentation of an image from the dataset.

        Args:
            index (Optional[int]): Index of the image to plot. If None, a random image will be selected.

        Returns:
            None
        """
        if index is None:
            index = random.randint(0, len(self) - 1)

        _, label = self.__getitem__(index)

        # Convert torch tensor to numpy array
        label_np = np.array(label)

        # Display the segmentation
        plt.imshow(label_np)
        plt.title('Segmentation')
        plt.show()
        

    def plot_image_and_segmentation(self, index: Optional[int] = None) -> None:
        """
        Plots both the RGB image and its segmentation from the dataset.

        Args:
            index (Optional[int]): Index of the image to plot. If None, a random image will be selected.

        Returns:
            None
        """
        if index is None:
            index = random.randint(0, len(self) - 1)

        image, label = self.__getitem__(index)

        # Convert PIL images to numpy arrays
        image_np = np.array(image)
        label_np = np.array(label)
        
        # Display both images side by side
        plt.subplot(1, 2, 1)
        plt.imshow(image_np)
        plt.title('RGB Image')

        plt.subplot(1, 2, 2)
        plt.imshow(label_np)
        plt.title('Segmentation')

        plt.show()
        
    
#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
if __name__=="__main__":
    import torch
    
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    transform=A.Compose(
    [
        A.Resize(256, 512),
        A.HorizontalFlip(),
        #A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
    )
    
    dataset = CustomCityscapes('../data/', split='train', mode='fine', target_type='semantic', transform=transform)
    
    
    img,seg= dataset[20]
    print(img.shape,seg.shape)
    
    print(torch.unique(seg))
    print(len(torch.unique(seg)))
    
    #class labels after label correction
    res=dataset.encode_segmap(seg.clone())
    print(res.shape)
    print(torch.unique(res))
    print(len(torch.unique(res)))
     
    
    #let do coloring
    res1=dataset.decode_segmap(res.clone())
    

    fig,ax=plt.subplots(ncols=2,figsize=(12,10))  
    ax[0].imshow(res,cmap='gray')
    ax[1].imshow(res1)


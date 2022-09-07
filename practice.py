from mask import *
import matplotlib.pyplot as plt
from glob import glob
import os
import cv2

import torch
from torchvision.transforms import transforms

from utils import *

# img_dir = '../all_datasets/SEM1/test_gt'
# img_paths = glob(os.path.join(img_dir, '*.png'))
# img_path = img_paths[0]
# img_sample = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#
# img_numpy = img_sample / 255.
# transform = transforms.Compose([transforms.ToTensor()])
# img = transform(img_numpy)
# img = torch.unsqueeze(img, dim=0)
#
# mask1, mask2 = generate_mask_pair(img)
# sub1 = generate_subimages(img, mask1)
# sub2 = generate_subimages(img, mask2)
#
# img, sub1, sub2 = torch.squeeze(img), torch.squeeze(sub1), torch.squeeze(sub2)
# plot_tensors([img, sub1, sub2], ['img', 'sub1', 'sub2'])

a = torch.ones(size=(1, 3, 2, 2))
print(a.item())

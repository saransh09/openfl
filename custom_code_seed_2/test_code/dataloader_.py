import torch
from torch import nn
import PIL
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as tsf
from skimage import io
import numpy as np
from sklearn.model_selection import train_test_split
import os
from openfl.interface.interactive_api.experiment import TaskInterface, DataInterface, ModelInterface, FLExperiment

DATA_PATH = '../data/segmented-images/'

def read_data(image_path, mask_path):
    """
    Read image and mask from disk.
    """
    img = io.imread(image_path)
    assert(img.shape[2] == 3)
    mask = io.imread(mask_path)
    return (img, mask[:, :, 0].astype(np.uint8))


class KvasirDataset(Dataset):
    """
    Kvasir dataset contains 1000 images for all collaborators.
    Args:
        data_path: path to dataset on disk
        collaborator_count: total number of collaborators
        collaborator_num: number of current collaborator
        is_validation: validation option
    """

    def __init__(self, images_path = '../data/segmented-images/images/', \
                        masks_path = '../data/segmented-images/masks/',
                        seed = 0, validation_fraction=1/8, is_validation=False):

        self.images_path = images_path
        self.masks_path = masks_path
        self.images_names = [img_name for img_name in sorted(os.listdir(
            self.images_path)) if len(img_name) > 3 and img_name[-3:] == 'jpg']
        self.seed = seed
        ## Seed the images
        np.random.seed(seed)

        assert(len(self.images_names) > 2), "Too few images"

        # Change the code here to randonly shuffle the data and then create the validation
        self.train_images, self.valid_images = train_test_split(self.images_names, test_size=validation_fraction, random_state=self.seed)

        # validation_size = max(1, int(len(self.images_names) * validation_fraction))

        # if is_validation:
            # self.images_names = self.images_names[-validation_size :]
        # else:
            # self.images_names = self.images_names[: -validation_size]

        # Finally set the self.images_names

        if is_validation:
            self.images_names = self.valid_images
        else:
            self.images_names = self.train_images

        # Prepare transforms
        self.img_trans = tsf.Compose([
            tsf.ToPILImage(),
            tsf.Resize((332, 332)),
            tsf.ToTensor(),
            tsf.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        self.mask_trans = tsf.Compose([
            tsf.ToPILImage(),
            tsf.Resize((332, 332), interpolation=PIL.Image.NEAREST),
            tsf.ToTensor()])


    def __getitem__(self, index):
        name = self.images_names[index]
        img, mask = read_data(self.images_path + name, self.masks_path + name)
        img = self.img_trans(img).numpy()
        mask = self.mask_trans(mask).numpy()
        return img, mask

    def __len__(self):
        return len(self.images_names)



class FedDataset(DataInterface):
    def __init__(self, UserDatasetClass, seed, **kwargs):
        self.UserDatasetClass = UserDatasetClass
        self.seed = seed
        self.kwargs = kwargs

    def _delayed_init(self, data_path=None):

        self.rank, self.world_size = [int(part) for part in data_path.split(',')]

        validation_fraction=1/8
        self.train_set = self.UserDatasetClass(validation_fraction=validation_fraction, seed=self.seed, is_validation=False)
        self.valid_set = self.UserDatasetClass(validation_fraction=validation_fraction, seed=self.seed, is_validation=True)

        # Do the actual sharding
        self._do_sharding( self.rank, self.world_size, self.seed)

    def _do_sharding(self, rank, world_size, seed):
        # This method relies on the dataset's implementation
        # i.e. coupled in a bad way
        np.random.seed(seed)
        np.random.shuffle(self.train_set.images_names)
        self.train_set.images_names = self.train_set.images_names[ rank-1 :: world_size ]

    def get_train_loader(self, **kwargs):
        """
        Output of this method will be provided to tasks with optimizer in contract
        """
        return DataLoader(
            self.train_set, num_workers=8, batch_size=self.kwargs['train_bs'], shuffle=True
            )

    def get_valid_loader(self, **kwargs):
        """
        Output of this method will be provided to tasks without optimizer in contract
        """
        return DataLoader(self.valid_set, num_workers=8, batch_size=self.kwargs['valid_bs'])

    def get_train_data_size(self):
        """
        Information for aggregation
        """
        return len(self.train_set)

    def get_valid_data_size(self):
        """
        Information for aggregation
        """
        return len(self.valid_set)

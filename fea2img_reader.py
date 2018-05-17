import torch.utils.data as data

from PIL import Image
import os
import os.path

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


def is_image_file(filename):
    """Checks if a file is an image.
    Args:
        filename (string): path to a file
    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)

def make_dataset(img_dir, fm_dir):
    data_list = []
    img_dir = os.path.expanduser(img_dir)
    fm_dir = os.path.expanduser(fm_dir)
    assert (os.path.exists(fm_dir) and os.path.exists(img_dir))

    for fname in sorted(os.listdir(fm_dir)):        
        if is_featuremap_file(fname):
            fm_path = os.path.join(fm_dir, fname)
            img_path = os.path.join(img_dir, fname)
            assert (os.path.exists(fm_path) and os.path.exists(img_path))
            item = (fm_path, img_path)
            data_list.append(item)

    return data_list


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

def h5_loader(path):
    assert path.endswith('.h5')
    return np.array(h5py.File(path, 'r')['data'])

class ImageFolder(data.Dataset):
    """A generic data loader where the images are arranged in this way: ::
        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png
        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png
    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.
     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """

    def __init__(self, img_dir, fm_dir, transform=None, img_loader=default_loader, fm_loader=h5_loader):
        data_list = make_dataset(img_dir, fm_dir)
        if len(data_list) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.data_list = data_list
        self.transform = transform
        self.img_loader = img_loader
        self.fm_loader = fm_loader

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        fm_path, img_path = self.data_list[index]
        fm = self.fm_loader(fm_path)
        img = self.img_loader(img_path)
        basename = os.path.basename(fm_path).strip('.xml')
        if self.transform is not None:
            img = self.transform(img)

        return fm, img, basename

    def __len__(self):
        return len(self.imgs)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str
import torchvision
import numpy as np

def get_argdict(dataset):
    if dataset == 'cifar10':
        argdict = {'root' : './data_cifar10', 
                   'tvds_cls' : torchvision.datasets.CIFAR10,
                   }
    elif dataset == 'cifar100':
        argdict = {'root' : './data_cifar100', 
                   'tvds_cls' : torchvision.datasets.CIFAR100,
                   }
    return argdict

def get_dataset(dataset, is_train, download, Dataset_class, transform):
    argdict = get_argdict(dataset)
    data = argdict['tvds_cls'](root=argdict['root'], train=is_train, download=download)

    inputs, target = [], []
    for dt in data:
        inp, tg = dt
        inputs.append(np.array(inp))
        target.append(tg)

    inputs = np.array(inputs)
    target = np.array(target).astype('int64')

    ds_instance = Dataset_class(inputs, target, transform)

    return ds_instance

def get_dataset_cifar100(is_train, download, Dataset_class, transform):
    return get_dataset('cifar100', is_train, download, Dataset_class, transform)

def get_dataset_cifar10(is_train, download, Dataset_class, transform):
    return get_dataset('cifar10', is_train, download, Dataset_class, transform)
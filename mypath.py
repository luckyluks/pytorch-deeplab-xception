import os
class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'pascal':
            return '/path/to/datasets/VOCdevkit/VOC2012/'  # folder that contains VOCdevkit/.
        elif dataset == 'sbd':
            return '/path/to/datasets/benchmark_RELEASE/'  # folder that contains dataset/.
        elif dataset == 'cityscapes':
            if os.path.isdir("/media/zed/Data/cityscapes"):
                print("using: /media/zed/Data/cityscapes")
                return '/media/zed/Data/cityscapes'
            elif os.path.isdir("/home/ubuntu/cityscapes/"):
                print("using: /home/ubuntu/cityscapes/")
                return '/home/ubuntu/cityscapes/'
            else: 
                print('Dataset {} not available.'.format(dataset))
                raise NotImplementedError
        elif dataset == 'coco':
            return '/media/zed/Data/g/'
        elif dataset == 'costum':
            if os.path.isdir("/media/zed/Data/gtdata"):
                print("using: /media/zed/Data/gtdata")
                return '/media/zed/Data/gtdata'
            elif os.path.isdir("/home/ubuntu/recorded_data/"):
                print("using: /home/ubuntu/recorded_data/")
                return '/home/ubuntu/recorded_data/'
            else: 
                print('Dataset {} not available.'.format(dataset))
                raise NotImplementedError
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError

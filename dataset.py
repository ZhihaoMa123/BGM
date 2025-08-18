from utils import *
import matplotlib.pyplot as plt
import os
import os.path as osp
import shutil
from PIL import Image, ImageOps, ImageFilter
from scipy.ndimage import uniform_filter
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'


class DataSetLoader(Dataset):
    def __init__(self, dataset_dir, dataset_name, patch_size, mode, img_norm_cfg=None):
        super(DataSetLoader).__init__()
        self.dataset_name = dataset_name
        dataset_dir = osp.join(dataset_dir, dataset_name)
        self.dataset_dir = dataset_dir
        self.patch_size = patch_size

        self.mode = mode

        self.files = []
        if mode == 'train':
            with open(osp.join(dataset_dir, 'img_idx', f'train_{dataset_name}.txt'), 'r') as f:
                self.files += [line.strip() for line in f.readlines()]   # self.train_list = f.read().splitlines()
        elif mode == 'val':
            with open(osp.join(dataset_dir, 'img_idx', f'test_{dataset_name}.txt'), 'r') as f:
                self.files += [line.strip() for line in f.readlines()]
        elif mode == 'test':
            with open(osp.join(dataset_dir, 'img_idx', f'train_{dataset_name}.txt'), 'r') as f:
                self.files += [line.strip() for line in f.readlines()]
            with open(osp.join(dataset_dir, 'img_idx', f'test_{dataset_name}.txt'), 'r') as f:
                self.files += [line.strip() for line in f.readlines()]
        else:
            raise NotImplementedError
        print(f'{len(self.files)} samples from {dataset_name} for {mode}')

        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg(dataset_name, dataset_dir)
        else:
            self.img_norm_cfg = img_norm_cfg
        self.tranform = augumentation()
    
    def generate_contrastive_mask(self, mask0, region_size=16, mask_ratio=0.2):  # mask0: [C, H, W]
        dh, dw = int(mask0.shape[0] // region_size), int(mask0.shape[0] // region_size)
        mask = np.zeros(dh * dw, dtype=np.float32)  
        mask[:round(dh * dw * mask_ratio)] = 1
        np.random.shuffle(mask)

        mask = mask.reshape(dh, dw)
        mask = mask.repeat(region_size, axis=0).repeat(region_size, axis=1)

        # 膨胀
        mask0 = nn.functional.max_pool2d(torch.from_numpy(mask0).unsqueeze(0), kernel_size=7, stride=1, padding=3).numpy()[0]

        mask[mask0 == 1] = 0  # positive 目标区域不进行mask
        return mask, mask0
    
    def __getitem__(self, idx):
        try:
            img = Image.open(osp.join(self.dataset_dir, 'images', self.files[idx] + '.png')).convert('I')
            mask = Image.open(osp.join(self.dataset_dir, 'masks', self.files[idx] + '.png'))
        except:
            img = Image.open(osp.join(self.dataset_dir, 'images', self.files[idx] + '.bmp')).convert('I')
            mask = Image.open(osp.join(self.dataset_dir, 'masks', self.files[idx] + '.bmp'))

        # print(self.files[idx])  # check for randomness, passed

        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
        # img = np.array(img, dtype=np.float32)
        mask = np.array(mask, dtype=np.float32)  / 255.0
        if len(mask.shape) > 2:
            mask = mask[:,:,0]
            
        if self.mode == 'train':
            img_patch, mask_patch = random_crop(img, mask, self.patch_size, pos_prob=0.5) 
            img_patch, mask_patch = self.tranform(img_patch, mask_patch)

            # # ====================================================================================
            # construct contrastive masking strategy
            # 1. 根据固定比例随机生成基于16*16 / 32*32 / 8*8的mask索引, 创建mask矩阵
            # 2. 构建正样本：将masking区域置为图像的均值, gt部分保留, label = label;
            # 3. 构建负样本：将gt区域膨胀5个像素点(基本充分掩盖目标), 将膨胀后的gt区域置为图像的均值, label = 0. 
            mask_positive, mask_negative = self.generate_contrastive_mask(mask_patch.copy(), region_size=16, mask_ratio=0.2)

            fill_ = img_patch.mean()
            img_positive = img_patch.copy()
            img_positive[mask_positive == 1] = fill_
            img_negative = img_patch.copy()
            img_negative[mask_negative == 1] = fill_


            img_positive, img_negative = img_positive[np.newaxis, :], img_negative[np.newaxis, :]
            img_positive = torch.from_numpy(np.ascontiguousarray(img_positive))
            img_negative = torch.from_numpy(np.ascontiguousarray(img_negative))

            # # # ====================================================================================
            
            img_patch, mask_patch = img_patch[np.newaxis,:], mask_patch[np.newaxis,:]
            img_patch = torch.from_numpy(np.ascontiguousarray(img_patch))
            mask_patch = torch.from_numpy(np.ascontiguousarray(mask_patch))

            return img_patch, mask_patch, img_positive, img_negative
        else:
            h, w = img.shape
            img = PadImg(img)
            mask = PadImg(mask)
            
            img, mask = img[np.newaxis,:], mask[np.newaxis,:]
            
            img = torch.from_numpy(np.ascontiguousarray(img))
            mask = torch.from_numpy(np.ascontiguousarray(mask))
            return img, mask, [h, w], self.files[idx]
    
    def __len__(self):
        return len(self.files)





    

class TestSetLoader(Dataset):
    def __init__(self, dataset_dir, train_dataset_name, test_dataset_name, img_norm_cfg=None):
        super(TestSetLoader).__init__()
        self.dataset_dir = dataset_dir + '/' + test_dataset_name
        with open(self.dataset_dir + '/img_idx/test_' + test_dataset_name + '.txt', 'r') as f:
            self.test_list = f.read().splitlines()
        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg(train_dataset_name, dataset_dir)
        else:
            self.img_norm_cfg = img_norm_cfg
        
    def __getitem__(self, idx):
        try:
            img = Image.open((self.dataset_dir + '/images/' + self.test_list[idx] + '.png').replace('//','/')).convert('I')
            mask = Image.open((self.dataset_dir + '/masks/' + self.test_list[idx] + '.png').replace('//','/'))
        except:
            img = Image.open((self.dataset_dir + '/images/' + self.test_list[idx] + '.bmp').replace('//','/')).convert('I')
            mask = Image.open((self.dataset_dir + '/masks/' + self.test_list[idx] + '.bmp').replace('//','/'))

        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
        mask = np.array(mask, dtype=np.float32)  / 255.0
        if len(mask.shape) > 2:
            mask = mask[:,:,0]
        
        h, w = img.shape
        img = PadImg(img)
        mask = PadImg(mask)
        
        img, mask = img[np.newaxis,:], mask[np.newaxis,:]
        
        img = torch.from_numpy(np.ascontiguousarray(img))
        mask = torch.from_numpy(np.ascontiguousarray(mask))
        return img, mask, [h,w], self.test_list[idx]
    def __len__(self):
        return len(self.test_list) 

class InferenceSetLoader(Dataset):
    def __init__(self, dataset_dir, train_dataset_name, test_dataset_name, img_norm_cfg=None):
        super(InferenceSetLoader).__init__()
        self.dataset_dir = dataset_dir + '/' + test_dataset_name
        with open(self.dataset_dir + '/img_idx/test_' + test_dataset_name + '.txt', 'r') as f:
            self.test_list = f.read().splitlines()
        if img_norm_cfg == None:
            self.img_norm_cfg = get_img_norm_cfg(train_dataset_name, dataset_dir)
        else:
            self.img_norm_cfg = img_norm_cfg
        
    def __getitem__(self, idx):
        try:
            img = Image.open((self.dataset_dir + '/images/' + self.test_list[idx] + '.png').replace('//','/')).convert('I')
        except:
            img = Image.open((self.dataset_dir + '/images/' + self.test_list[idx] + '.bmp').replace('//','/')).convert('I')
        img = Normalized(np.array(img, dtype=np.float32), self.img_norm_cfg)
        
        h, w = img.shape
        img = PadImg(img)
        
        img = img[np.newaxis,:]
        
        img = torch.from_numpy(np.ascontiguousarray(img))
        return img, [h,w], self.test_list[idx]
    def __len__(self):
        return len(self.test_list) 


class EvalSetLoader(Dataset):
    def __init__(self, dataset_dir, mask_pred_dir, test_dataset_name, model_name):
        super(EvalSetLoader).__init__()
        self.dataset_dir = dataset_dir
        self.mask_pred_dir = mask_pred_dir
        self.test_dataset_name = test_dataset_name
        self.model_name = model_name
        with open(self.dataset_dir+'/img_idx/test_' + test_dataset_name + '.txt', 'r') as f:
            self.test_list = f.read().splitlines()

    def __getitem__(self, idx):
        mask_pred = Image.open((self.mask_pred_dir + self.test_dataset_name + '/' + self.model_name + '/' + self.test_list[idx] + '.png').replace('//','/'))
        mask_gt = Image.open(self.dataset_dir + '/masks/' + self.test_list[idx] + '.png')

        mask_pred = np.array(mask_pred, dtype=np.float32)  / 255.0
        mask_gt = np.array(mask_gt, dtype=np.float32)  / 255.0
        
        if len(mask_pred.shape) == 3:
            mask_pred = mask_pred[:,:,0]
        
        h, w = mask_pred.shape
        
        mask_pred, mask_gt = mask_pred[np.newaxis,:], mask_gt[np.newaxis,:]
        
        mask_pred = torch.from_numpy(np.ascontiguousarray(mask_pred))
        mask_gt = torch.from_numpy(np.ascontiguousarray(mask_gt))
        return mask_pred, mask_gt, [h,w]
    def __len__(self):
        return len(self.test_list) 

class augumentation(object):
    def __call__(self, input, target):
        if random.random()<0.5:
            input = input[::-1, :]
            target = target[::-1, :]

            # print('aug: 1') # check for randomness, passed
        if random.random()<0.5:
            input = input[:, ::-1]
            target = target[:, ::-1]
            # print('aug: 2') # check for randomness, passed
        if random.random()<0.5:
            input = input.transpose(1, 0)
            target = target.transpose(1, 0)
            # print('aug: 3') # check for randomness, passed
        return input, target

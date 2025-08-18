import argparse
import time
import timeit
from torch.autograd import Variable
from torch.utils.data import DataLoader
from net import Net
from dataset import *
import matplotlib.pyplot as plt
from metrics import *
import numpy as np
import os


from PIL import Image

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

from torch.utils.tensorboard import SummaryWriter  


def save_checkpoint(state, save_path):
    if not os.path.exists(os.path.dirname(save_path)):
        os.makedirs(os.path.dirname(save_path))
    torch.save(state, save_path)
    return save_path


class Trainer(object):
    def __init__(self, opt):
        assert opt.mode == 'train' or opt.mode == 'test'
 
        self.mode = opt.mode
        # ------------------------
        seed = opt.seed

        def seed_worker(seed=42):
            random.seed(seed)
            np.random.seed(seed)
            os.environ['PYTHONHASHSEED'] = str(seed)
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=True)
        
        seed_worker(seed)
        g = torch.Generator()
        g.manual_seed(seed)

    
        self.trainset = opt.trainset
        train_set = DataSetLoader(dataset_dir=opt.dataset_dir, dataset_name=opt.trainset, patch_size=opt.patchSize, mode='train', img_norm_cfg=opt.img_norm_cfg)
        self.train_loader = DataLoader(dataset=train_set, num_workers=opt.num_workers, batch_size=opt.batchSize, shuffle=True, worker_init_fn=seed_worker, generator=g)

        self.val_loaders = {}
        self.best_metric_mIoU = {}
        self.best_metric_nIoU = {}
        for dkey in args.testset:
            valset = DataSetLoader(dataset_dir=opt.dataset_dir, dataset_name=dkey, patch_size=None, mode='val' if dkey == args.trainset else 'test', img_norm_cfg=opt.img_norm_cfg)
            self.val_loaders[dkey] = DataLoader(valset, 1, drop_last=False, num_workers=opt.num_workers, shuffle=False, worker_init_fn=seed_worker, generator=g)
            self.best_metric_mIoU[dkey] = 0.
            self.best_metric_nIoU[dkey] = 0.

        device = torch.device('cuda')
        self.device = device
        self.model = Net(model_name=opt.model).to(device)


        self.epoch_state = 0        
        if opt.resume:
            for resume_pth in opt.resume:
                if opt.dataset_name in resume_pth and opt.model in resume_pth:
                    ckpt = torch.load(resume_pth)
                    self.model.load_state_dict(ckpt['state_dict'])
                    self.epoch_state = ckpt['epoch']
                    total_loss_list = ckpt['total_loss']
                    for i in range(len(opt.scheduler_settings['step'])):
                        opt.scheduler_settings['step'][i] = opt.scheduler_settings['step'][i] - ckpt['epoch']
        if opt.pretrained:
            for pretrained_pth in opt.pretrained:
                if opt.dataset_name in pretrained_pth and opt.model in pretrained_pth:
                    ckpt = torch.load(resume_pth)
                    self.model.load_state_dict(ckpt['state_dict'])
                else:
                    raise NotImplementedError
        self.model = torch.nn.DataParallel(self.model)

        ### Default settings         
        if opt.model == 'SCTransNet' and opt.optimizer_name == 'Adam':
            opt.optimizer_name == 'Adam'
            opt.optimizer_settings = {'lr': 0.001}
            opt.scheduler_name = 'CosineAnnealingLRw10'
            opt.scheduler_settings = {'epochs': opt.epochs, 'eta_min': 1e-5, 'last_epoch': -1}
                
        elif opt.optimizer_name == 'Adam':
            opt.optimizer_settings = {'lr': 5e-4}
            opt.scheduler_name = 'MultiStepLR'
            opt.scheduler_settings = {'epochs':400, 'step': [200, 300], 'gamma': 0.1}
            opt.scheduler_settings['epochs'] = opt.epochs  
        ### Default settings of DNANet                
        elif opt.optimizer_name == 'Adagrad':
            opt.optimizer_settings = {'lr': 0.05}
            opt.scheduler_name = 'CosineAnnealingLR'
            opt.scheduler_settings = {'epochs':1500, 'min_lr':1e-5}
            opt.scheduler_settings['epochs'] = opt.epochs
        else:
            raise NotImplementedError
            
        opt.epochs = opt.scheduler_settings['epochs']

        self.optimizer, self.scheduler = get_optimizer(self.model, opt.optimizer_name, opt.scheduler_name, opt.optimizer_settings, opt.scheduler_settings)

        
        self.savename = opt.name + '-' + time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())) + '-' + opt.trainset 
        if self.mode == 'train':
            self.writer = SummaryWriter(f'./tf-logs/{self.savename}')
            self.save_folder = f'./weights/{self.savename}'
            if not osp.exists(self.save_folder):
                os.makedirs(self.save_folder)     

        self.opt = opt 

    def train(self, epoch):
        tic = timeit.default_timer()

        self.model.train()
        train_loss = 0


        for i, (data, mask, positive, negative) in enumerate(self.train_loader):
  
            batch = data.shape[0]

            data = data.to(self.device)
            labels = mask.to(self.device)


            pred  = self.model.forward(data)

            loss = self.model.module.loss(pred, labels)
            train_loss += loss.detach().cpu()
            
            self.optimizer.zero_grad()
            loss.backward()        
            self.optimizer.step()

        self.scheduler.step()
        train_loss /= len(self.train_loader)

        self.writer.add_scalar('train_loss', train_loss, epoch)
        toc = timeit.default_timer()
        print('Train epoch {:03d} on {}, loss: {:.4f}, time elapsed {}:{}s.'.format(epoch, self.trainset, train_loss, int((toc - tic) // 60), int(toc - tic) % 60))
    
    def test(self, epoch):
        self.model.eval()

        for dkey in self.val_loaders.keys():
            tic = timeit.default_timer()
            
            fn_mIoU = mIoU() 
            fn_PD_FA = PD_FA()
                        
            with torch.no_grad():
                for i, (data, mask, size, _) in enumerate(self.val_loaders[dkey]):
        
                    data = data.to(self.device)

                    pred = self.model.forward(data)                    
                    pred = pred[:, :, :size[0], :size[1]]
                    mask = mask[:, :, :size[0], :size[1]]
                    fn_mIoU.update((pred > self.opt.threshold).cpu(), mask)
                    fn_PD_FA.update((pred[0, 0, :, :] > self.opt.threshold).cpu(), mask[0, 0, :, :], size)     
                
            eval_pixAcc, eval_mIoU = fn_mIoU.get()
            eval_Pd, eval_Fa = fn_PD_FA.get()
            _, eval_nIoU = fn_mIoU.get_single()

            if self.mode == 'train':
                self.writer.add_scalar(f'eval-{dkey}-pixAcc', eval_pixAcc * 1e2, epoch)
                self.writer.add_scalar(f'eval-{dkey}-mIoU', eval_mIoU * 1e2, epoch)
                self.writer.add_scalar(f'eval-{dkey}-nIoU', eval_nIoU * 1e2, epoch)
                self.writer.add_scalar(f'eval-{dkey}-Pd', eval_Pd * 1e2, epoch)
                self.writer.add_scalar(f'eval-{dkey}-Fa', eval_Fa * 1e6, epoch)

                if eval_mIoU > self.best_metric_mIoU[dkey]:
                    self.best_metric_mIoU[dkey] = eval_mIoU
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': self.model.module.state_dict(),
                        'eval_pixAcc': eval_pixAcc, 
                        'eval_mIoU': eval_mIoU, 
                        'eval_nIoU': eval_nIoU, 
                        'eval_Pd': eval_Pd, 
                        'eval_Fa': eval_Fa}, 
                        osp.join(self.save_folder, f'best_mIoU_on_{dkey}.pth.tar'))
                    
                    with open(osp.join(self.save_folder, f'metrics_mIoU_on_{dkey}.log'), 'a') as f:
                        f.write('{} - {:04d}\t - pixAcc {:.4f}\t - mIoU {:.4f}\t nIoU {:.4f}\t - PD {:.4f}\t - FA {:.4f}\n' .
                            format(time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time())), 
                                epoch, eval_pixAcc * 1e2, eval_mIoU * 1e2, eval_nIoU * 1e2, eval_Pd * 1e2, eval_Fa * 1e6))
                        
                if eval_nIoU > self.best_metric_nIoU[dkey]:
                    self.best_metric_nIoU[dkey] = eval_nIoU
                    save_checkpoint({
                        'epoch': epoch + 1,
                        'state_dict': self.model.module.state_dict(),
                        'eval_pixAcc': eval_pixAcc, 
                        'eval_mIoU': eval_mIoU, 
                        'eval_nIoU': eval_nIoU, 
                        'eval_Pd': eval_Pd, 
                        'eval_Fa': eval_Fa}, 
                        osp.join(self.save_folder, f'best_nIoU_on_{dkey}.pth.tar'))
                    
                    with open(osp.join(self.save_folder, f'metrics_nIoU_on_{dkey}.log'), 'a') as f:
                        f.write('{} - {:04d}\t - pixAcc {:.4f}\t - mIoU {:.4f}\t nIoU {:.4f}\t - PD {:.4f}\t - FA {:.4f}\n' .
                            format(time.strftime('%Y-%m-%d-%H-%M-%S',time.localtime(time.time())), 
                                epoch, eval_pixAcc * 1e2, eval_mIoU * 1e2, eval_nIoU * 1e2, eval_Pd * 1e2, eval_Fa * 1e6))
                        
            toc = timeit.default_timer()

            print('Eval on {}, time elapsed {}:{}s.'.format(dkey, int((toc - tic) // 60), int(toc - tic) % 60))
            print('pixAcc: ' + str(eval_pixAcc * 1e2))
            print('mIoU: ' + str(eval_mIoU * 1e2) + '  Best_mIoU: ' + str(self.best_metric_mIoU[dkey] * 1e2))
            print('nIoU: ' + str(eval_nIoU * 1e2) + '  Best_nIoU: ' + str(self.best_metric_nIoU[dkey] * 1e2))
            print('Pd: ' + str(eval_Pd * 1e2))
            print('Fa: ' + str(eval_Fa * 1e6))
            print('')
        
    def inference(self, save_output=True):
        
        ToImg = transforms.ToPILImage()

        for imetric in ['mIoU', 'nIoU']:
            for dkey in self.val_loaders.keys():
                ckpt = torch.load(osp.join(self.save_folder, f'best_{imetric}_on_{dkey}.pth.tar'))
                self.model.module.load_state_dict(ckpt['state_dict'])
                
                self.model.eval()

                tic = timeit.default_timer()
                
                fn_mIoU = mIoU() 
                fn_PD_FA = PD_FA()

                output_path = f'./outputs_{imetric}/{self.savename}'
                if save_output and not osp.exists(output_path):
                    os.makedirs(output_path)

                with torch.no_grad():
                    for i, (data, mask, size, filename) in enumerate(self.val_loaders[dkey]):
            
                        data = data.to(self.device)


                        pred = self.model.forward(data)                    
                        pred = pred[:, :, :size[0], :size[1]]
                        mask = mask[:, :, :size[0], :size[1]]
                        fn_mIoU.update((pred > self.opt.threshold).cpu(), mask)
                        fn_PD_FA.update((pred[0, 0, :, :] > self.opt.threshold).cpu(), mask[0, 0, :, :], size)     

                        if save_output:
                            for j_ in range(pred.shape[0]):
                                j_pred = ToImg((pred[j_].detach().cpu() > self.opt.threshold).float())
                                j_pred.save(osp.join(output_path, filename[j_] + '.png'))
                    
                eval_pixAcc, eval_mIoU = fn_mIoU.get()
                eval_Pd, eval_Fa = fn_PD_FA.get()
                _, eval_nIoU = fn_mIoU.get_single()

                toc = timeit.default_timer()

                print('=========================')
                print('Inference on {} with {}, time elapsed {}:{}s.'.format(dkey, imetric, int((toc - tic) // 60), int(toc - tic) % 60))
                print('pixAcc: ' + str(eval_pixAcc * 1e2))
                print('mIoU: ' + str(eval_mIoU * 1e2))
                print('nIoU: ' + str(eval_nIoU * 1e2))
                print('Pd: ' + str(eval_Pd * 1e2))
                print('Fa: ' + str(eval_Fa * 1e6))
                print('')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="PyTorch BasicIRSTD train")
    parser.add_argument("--model", default='BasicUNet_plus', 
                        help="model: 'ACM', 'ALCNet', 'DNANet', 'ISNet', 'UIUNet', 'RDIAN', 'ISTDU-Net', 'U-Net', 'RISTDnet'")    
    parser.add_argument("--name", default='debug', 
                        help="experiment description")     

    parser.add_argument("--trainset", default='IRSTD-1K', 
                        help="dataset_name: 'NUAA-SIRST', 'NUDT-SIRST', 'IRSTD-1K', 'SIRST3', 'NUDT-SIRST-Sea', 'IRDST-real'")
    parser.add_argument("--testset", default='IRSTD-1K', 
                        help="dataset_name: 'NUAA-SIRST', 'NUDT-SIRST', 'IRSTD-1K', 'SIRST3', 'NUDT-SIRST-Sea', 'IRDST-real'")

    parser.add_argument("--img_norm_cfg", default=None, type=dict,
                        help="specific a img_norm_cfg, default=None (using img_norm_cfg values of each dataset)")
    parser.add_argument("--img_norm_cfg_mean", default=None, type=float,
                        help="specific a mean value img_norm_cfg, default=None (using img_norm_cfg values of each dataset)")
    parser.add_argument("--img_norm_cfg_std", default=None, type=float,
                        help="specific a std value img_norm_cfg, default=None (using img_norm_cfg values of each dataset)")

    parser.add_argument("--dataset_dir", default='/home/pc/work/BasicIRSTD-main/data', type=str, help="train_dataset_dir")

    parser.add_argument("--batchSize", type=int, default=16, help="Training batch sizse")
    parser.add_argument("--patchSize", type=int, default=256, help="Training patch size")

    parser.add_argument("--resume", default=None, type=str, help="Resume from exisiting checkpoints (default: None)")
    parser.add_argument("--pretrained", default=None, type=str, help="Load pretrained checkpoints (default: None)")

    parser.add_argument("--epochs", type=int, default=400, help="Number of epochs")
    parser.add_argument("--optimizer_name", default='Adam', type=str, help="optimizer name: Adam, Adagrad, SGD")
    parser.add_argument("--optimizer_settings", default={'lr': 5e-4}, type=dict, help="optimizer settings")

    parser.add_argument("--scheduler_name", default='MultiStepLR', type=str, help="scheduler name: MultiStepLR")
    parser.add_argument("--scheduler_settings", default={'step': [200, 300], 'gamma': 0.5}, type=dict, help="scheduler settings")

    parser.add_argument("--num_workers", type=int, default=0, help="Number of threads for data loader to use")
    parser.add_argument("--threshold", type=float, default=0.5, help="Threshold for test")
    parser.add_argument("--seed", type=int, default=42, help="random seed")

    parser.add_argument("--mode", type=str, default='train', help="train or test")
    parser.add_argument("--test_freq", type=int, default=10, help="frequency for evaluating method (efficient training)")
    args = parser.parse_args()

    args.testset = args.testset.split('/')

    ## Set img_norm_cfg
    if args.img_norm_cfg_mean != None and args.img_norm_cfg_std != None:
        args.img_norm_cfg = dict()
        args.img_norm_cfg['mean'] = args.img_norm_cfg_mean
        args.img_norm_cfg['std'] = args.img_norm_cfg_std

    for itrainset in ['IRSTD-1K','NUAA-SIRST','NUDT-SIRST' ]:
        args.trainset = itrainset
        args.testset = [itrainset]

        print('///////////////////////////////////////////////////////')
        print(args)

        trainer = Trainer(args)

        if trainer.mode=='train':
            print('\n========== Training ===========')
            for epoch in range(trainer.epoch_state, args.epochs):
                trainer.train(epoch)

                if (epoch + 1) % args.test_freq == 0:
                    print('-----------------------')
                    trainer.test(epoch)

        print('\n========== Inference ===========')
        trainer.inference()

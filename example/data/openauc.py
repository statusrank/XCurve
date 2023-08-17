import os
import argparse
import datetime
import time
import importlib
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

from XCurve.OpenAUC.models.wrapper_classes import TimmResNetWrapper, Classifier32Wrapper
from XCurve.OpenAUC.dataloaders.open_set_datasets import get_class_splits, get_datasets
from XCurve.OpenAUC.losses.OpenAUCLoss import StandardOpenAUCLoss
from XCurve.OpenAUC.utils.common_utils import AverageMeter, init_experiment, seed_torch, str2bool
from XCurve.OpenAUC.optimizers import get_scheduler, get_optimizer
from XCurve.OpenAUC.utils.model_utils import get_model, save_networks
from XCurve.OpenAUC.utils.config import exp_root
from XCurve.OpenAUC.metrics import OpenSetEvaluator, EnsembleModel


parser = argparse.ArgumentParser("Training")

# Dataset
parser.add_argument('--dataset', type=str, default='cub', help="")
parser.add_argument('--out-num', type=int, default=10, help='For cifar-10-100')
parser.add_argument('--image_size', type=int, default=64)

# optimization
parser.add_argument('--optim', type=str, default=None, help="Which optimizer to use {adam, sgd}")
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=0.1, help="learning rate for model")
parser.add_argument('--weight_decay', type=float, default=1e-4, help="LR regularisation on weights")
parser.add_argument('--max-epoch', type=int, default=600)
parser.add_argument('--scheduler', type=str, default='cosine_warm_restarts')
parser.add_argument('--num_restarts', type=int, default=2, help='How many restarts for cosine_warm_restarts schedule')

# model
parser.add_argument('--close_loss', type=str, default='Softmax')
parser.add_argument('--openauc', default=True, type=str2bool, metavar='BOOL')
parser.add_argument('--alpha', type=float, default=2, help="parameter for openauc loss")
parser.add_argument('--lambda', type=float, default=0.1, help="parameter for openauc loss")

parser.add_argument('--label_smoothing', type=float, default=None, help="Smoothing constant for label smoothing.")
parser.add_argument('--temp', type=float, default=1.0, help="temp for label_smoothing")
parser.add_argument('--model', type=str, default='classifier32')
parser.add_argument('--resnet50_pretrain', type=str, default='places_moco',
                        help='Which pretraining to use if --model=timm_resnet50_pretrained.'
                             'Options are: {iamgenet_moco, places_moco, places}', metavar='BOOL')
parser.add_argument('--feat_dim', type=int, default=128, help="Feature vector dim, only for classifier32 at the moment")

# aug
parser.add_argument('--transform', type=str, default='rand-augment')
parser.add_argument('--rand_aug_m', type=int, default=30)
parser.add_argument('--rand_aug_n', type=int, default=2)

# misc
parser.add_argument('--num_workers', default=16, type=int)
parser.add_argument('--split_train_val', default=False, type=str2bool,
                        help='Subsample training set to create validation set', metavar='BOOL')
parser.add_argument('--device', default='cuda:0', type=str, help='Which GPU to use')
parser.add_argument('--gpus', default=[0], type=int, nargs='+',
                        help='device ids assignment (e.g 0 1 2 3)')

parser.add_argument('--eval-freq', type=int, default=1)
parser.add_argument('--print-freq', type=int, default=100)
parser.add_argument('--checkpt_freq', type=int, default=20)
parser.add_argument('--gpu', type=str, default='0')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--use-cpu', action='store_true')
parser.add_argument('--eval', action='store_true', help="Eval", default=False)
parser.add_argument('--split_idx', default=0, type=int, help='0-4 OSR splits for each dataset')
parser.add_argument('--use_softmax_in_eval', default=False, type=str2bool,
                        help='Do we use softmax or logits for evaluation', metavar='BOOL')

def train(net, criterion, optimizer, trainloader, epoch=None, **options):
    net.train()
    losses = AverageMeter()
    torch.cuda.empty_cache()

    loss_all = 0
    criterion = StandardOpenAUCLoss(criterion, **options) if options['openauc'] else criterion
    for data, labels, _ in tqdm(trainloader):
        if options['use_gpu']:
            data, labels = data.cuda(), labels.cuda()

        with torch.set_grad_enabled(True):
            optimizer.zero_grad()
            embedding, logits = net(data, True)

            if options['openauc']:
                _, loss = criterion(logits, labels, net.net.fc, embedding)
            else:
                _, loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()
        
        losses.update(loss.item(), data.size(0))
        
        loss_all += losses.avg

    return loss_all

def test(net, testloader, outloader, **options):
    model = EnsembleModel(net)
    model.eval()
    if options['use_gpu']:
        model = model.cuda()

    # ------------------------
    # EVALUATE
    # ------------------------
    evaluate = OpenSetEvaluator(model=model, known_data_loader=testloader, unknown_data_loader=outloader)

    # Make predictions on test sets
    preds = evaluate.predict(save=False)
    results = evaluate.evaluate(evaluate, load=False, preds=preds, normalised_ap=False)

    return results

def main_worker(options, args):

    torch.manual_seed(options['seed'])
    os.environ['CUDA_VISIBLE_DEVICES'] = options['gpu']
    use_gpu = torch.cuda.is_available()
    if options['use_cpu']: use_gpu = False

    if use_gpu:
        print("Currently using GPU: {}".format(options['gpu']))
        cudnn.benchmark = False
        torch.cuda.manual_seed_all(options['seed'])
    else:
        print("Currently using CPU")

    # -----------------------------
    # DATALOADERS
    # -----------------------------
    trainloader = dataloaders['train']
    testloader = dataloaders['val']
    outloader = dataloaders['test_unknown']

    # -----------------------------
    # MODEL
    # -----------------------------
    print("Creating model: {}".format(options['model']))
    if args.model == 'timm_resnet50_pretrained':
        wrapper_class = TimmResNetWrapper
    else:
        wrapper_class = Classifier32Wrapper
    net = get_model(args, wrapper_class=wrapper_class)
    feat_dim = args.feat_dim

    # Loss
    options.update({
        'feat_dim': feat_dim,
        'use_gpu':  use_gpu
    })

    # -----------------------------
    # GET LOSS
    # -----------------------------
    Loss = importlib.import_module('loss.' + options['close_loss'])
    criterion = getattr(Loss, options['close_loss'])(**options)

    # -----------------------------
    # PREPARE EXPERIMENT
    # -----------------------------
    if use_gpu:
        net = net.cuda()
        criterion = criterion.cuda()
    
    model_path = os.path.join(args.log_dir)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    params_list = [{'params': net.parameters()},
                    {'params': criterion.parameters()}]
    
    # Get base network and criterion
    optimizer = get_optimizer(args=args, params_list=params_list, **options)

    # -----------------------------
    # GET SCHEDULER
    # ----------------------------
    scheduler = get_scheduler(optimizer, args)

    # -----------------------------
    # TRAIN
    # -----------------------------
    start_time = time.time()
    for epoch in range(options['max_epoch']):
        print("==> Epoch {}/{}".format(epoch+1, options['max_epoch']), end='\t')

        train(net, criterion, optimizer, trainloader, epoch=epoch, **options)

        if options['eval_freq'] > 0 and (epoch+1) % options['eval_freq'] == 0 or (epoch+1) == options['max_epoch']:
            print("==> Test", options['loss'], end='\t')
            results = test(net, testloader, outloader, **options)
            print("Acc:{:.3f}\tAUROC:{:.3f}\tOpenAUC:{:.3f}".format(results['Acc'], results['AUROC'], results['OpenAUC']))
                                                                                   
            if epoch % options['checkpt_freq'] == 0 or epoch == options['max_epoch'] - 1:
                save_networks(net, model_path, file_name.split('.')[0]+'_{}'.format(epoch), options['loss'], criterion=criterion)

        # -----------------------------
        # STEP SCHEDULER
        # ----------------------------
        if args.scheduler == 'plateau' or args.scheduler == 'warm_restarts_plateau':
            scheduler.step(results['Acc'], epoch)
        elif args.scheduler == 'multi_step':
            scheduler.step()
        else:
            scheduler.step(epoch=epoch)


    elapsed = round(time.time() - start_time)
    elapsed = str(datetime.timedelta(seconds=elapsed))
    print("Finished. Total elapsed time (h:m:s): {}".format(elapsed))

    return results

if __name__ == '__main__':
    args = parser.parse_args()
    args.exp_root = exp_root
    args.epochs = args.max_epoch
    img_size = args.image_size
    args.loss = 'OpenAUC-' + args.close_loss if args.openauc else args.close_loss

    # ------------------------
    # INIT
    # ------------------------
    if args.feat_dim is None:
        args.feat_dim = 128 if args.model == 'classifier32' else 2048

    args.train_classes, args.open_set_classes = get_class_splits(args.dataset, args.split_idx, cifar_plus_n=args.out_num)

    img_size = args.image_size

    args.save_name = '{}_{}_{}'.format(args.model, args.seed, args.dataset)
    args = init_experiment(args)

    # ------------------------
    # SEED
    # ------------------------
    seed_torch(args.seed)

    # ------------------------
    # DATASETS
    # ------------------------
    datasets = get_datasets(args.dataset, transform=args.transform, train_classes=args.train_classes,
                            open_set_classes=args.open_set_classes, balance_open_set_eval=True,
                            split_train_val=args.split_train_val, image_size=args.image_size, seed=args.seed,
                            args=args)

    # ------------------------
    # RANDAUG HYPERPARAM SWEEP
    # ------------------------
    if args.transform == 'rand-augment':
        if args.rand_aug_m is not None:
            if args.rand_aug_n is not None:
                datasets['train'].transform.transforms[0].m = args.rand_aug_m
                datasets['train'].transform.transforms[0].n = args.rand_aug_n

    # ------------------------
    # DATALOADER
    # ------------------------
    dataloaders = {}
    for k, v, in datasets.items():
        shuffle = True if k == 'train' else False
        dataloaders[k] = DataLoader(v, batch_size=args.batch_size,
                                    shuffle=shuffle, sampler=None, num_workers=args.num_workers)

    # ------------------------
    # SAVE PARAMS
    # ------------------------
    options = vars(args)
    options.update({
        'known':    args.train_classes,
        'unknown':  args.open_set_classes,
        'img_size': img_size,
        'dataloaders': dataloaders,
        'num_classes': len(args.train_classes)
    })

    dir_name = '{}_{}'.format(options['model'], options['loss'])
    dir_path = os.path.join(args.log_dir, 'results', dir_name)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    if options['dataset'] == 'cifar-10-100':
        file_name = '{}_{}.csv'.format(options['dataset'], options['out_num'])
    else:
        file_name = options['dataset'] + '.csv'

    print('result path:', os.path.join(dir_path, file_name))
    # ------------------------
    # TRAIN
    # ------------------------
    main_worker(options, args)

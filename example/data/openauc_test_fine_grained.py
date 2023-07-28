from XCurve.OpenAUC.metrics import OpenSetEvaluator, EnsembleModel
from XCurve.OpenAUC.utils.common_utils import strip_state_dict

import torch
import argparse
import pickle

from torch.utils.data import DataLoader
from XCurve.OpenAUC.dataloaders.open_set_datasets import get_datasets
from XCurve.OpenAUC.utils.model_utils import get_model
import os

from XCurve.OpenAUC.models.wrapper_classes import TimmResNetWrapper
from XCurve.OpenAUC.utils.config import save_dir, osr_split_dir, root_model_path


def load_model(path, args):
    model = get_model(args, wrapper_class=TimmResNetWrapper)
    state_dict = strip_state_dict(torch.load(path))
    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    return model

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='cls',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # General
    parser.add_argument('--gpus', default=[0], type=int, nargs='+',
                        help='device ids assignment (e.g 0 1 2 3)')
    parser.add_argument('--resnet50_pretrain', type=str, default='places_moco',
                        help='Which pretraining to use if --model=timm_resnet50_pretrained.'
                             'Options are: {iamgenet_moco, places_moco, places}', metavar='BOOL')
    parser.add_argument('--device', default='None', type=str, help='Which GPU to use')
    parser.add_argument('--seed', default=0, type=int)

    # Model
    parser.add_argument('--model', type=str, default='timm_resnet50_pretrained')
    parser.add_argument('--loss', type=str, default='Softmax')
    parser.add_argument('--feat_dim', default=2048, type=int)
    parser.add_argument('--max_epoch', default=599, type=int)

    # Data params
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--image_size', default=448, type=int)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--dataset', type=str, default='scars')
    parser.add_argument('--transform', type=str, default='rand-augment')
    parser.add_argument('--exp_id', type=str, default='')

    # Train params
    args = parser.parse_args()
    args.save_dir = save_dir + args.exp_id
    args.use_supervised_places = False

    device = torch.device('cuda:0')

    assert args.exp_id is not None

    # model path
    model_path = root_model_path.format(args.exp_id, args.dataset, args.max_epoch, args.loss)
    print(model_path)

    # Get OSR splits
    osr_path = os.path.join(osr_split_dir, '{}_osr_splits.pkl'.format(args.dataset))

    with open(osr_path, 'rb') as f:
        class_info = pickle.load(f)

    train_classes = class_info['known_classes']
    open_set_classes = class_info['unknown_classes']


    for difficulty in ('Easy', 'Medium', 'Hard'):

        # ------------------------
        # DATASETS
        # ------------------------
        args.train_classes, args.open_set_classes = train_classes, open_set_classes[difficulty]

        # if difficulty == 'Hard' and args.dataset != 'imagenet':
        #     args.open_set_classes += open_set_classes['Medium']

        datasets = get_datasets(args.dataset, transform=args.transform, train_classes=args.train_classes,
                                image_size=args.image_size, balance_open_set_eval=False,
                                split_train_val=False, open_set_classes=args.open_set_classes)

        # ------------------------
        # DATALOADERS
        # ------------------------
        dataloaders = {}
        for k, v, in datasets.items():
            shuffle = True if k == 'train' else False
            dataloaders[k] = DataLoader(v, batch_size=args.batch_size,
                                        shuffle=shuffle, sampler=None, num_workers=args.num_workers)

        # ------------------------
        # MODEL
        # ------------------------
        print('Loading model...')
        model = EnsembleModel(load_model(path=model_path, args=args))
        model.eval()
        model = model.to(device)

        # ------------------------
        # EVALUATE
        # ------------------------
        evaluate = OpenSetEvaluator(model=model, known_data_loader=dataloaders['test_known'],
                                   unknown_data_loader=dataloaders['test_unknown'], device=device, save_dir=args.save_dir)

        # Make predictions on test sets
        evaluate.predict()
        evaluate.evaluate(evaluate, normalised_ap=False)

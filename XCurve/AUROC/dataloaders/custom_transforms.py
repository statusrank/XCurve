import torch
import random
import numpy as np
import skimage as skimg
import cv2
import math
from PIL import Image, ImageOps, ImageFilter
import PIL.ImageEnhance as ImageEnhance
import time

class GaussianNoise(object):
    def __init__(self,mean=0,std=0):
        self.mean = mean
        self.std = std
    def __call__(self,sample):
        key_list = sample.keys()
        for key in key_list:
            if 'image' in key:
                img = sample[key]
                sample[key] = skimg.util.random_noise(img,mode='gaussian',mean=self.mean,var=(self.std)**2)

        return sample

class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self, mean=[123.675, 116.280, 103.530], std=[58.395, 57.120, 57.375]):
        if isinstance(std, (int, float)):
            std = [std, std, std]
        else:
            self.mean = np.array(mean)
            self.std = np.array(std)

    def __call__(self, sample):
        key_list = sample.keys()
        for key in key_list:
            if 'image' in key:
                img = sample[key].astype(np.float64)
                img -= self.mean
                img /= self.std
                sample[key] = img
        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        key_list = list(sample.keys())
        for key in key_list:
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
            if 'image' == key:
                img = sample[key]
                if len(img.shape) == 2:
                    img = np.expand_dims(img, axis=0).copy()
                else:
                    img = img.transpose((2, 0, 1)).copy()
                sample[key] = torch.from_numpy(img).float()
            elif 'audio' == key:
                aud = sample[key]
                # aud = aud.reshape((1,-1,1))
                sample[key] = torch.from_numpy(aud).float()
            elif 'label' in key:
                label = sample[key]
                # sample[key] = torch.from_numpy(label).long()
                sample[key] = torch.tensor(label).long()
        return sample


class RandomHorizontalFlip(object):
    def __call__(self, sample):
        if np.random.rand() < 0.5:
            key_list = sample.keys()
            for key in key_list:
                if 'image' not in key:
                    continue
                image = sample[key]
                image_flip = np.flip(image, axis=1)
                sample[key] = image_flip
        return sample


class RandomRotate(object):
    """Randomly rotate image"""
    def __init__(self, angle_r,image_value=127,is_continuous=True):
        self.angle_r = angle_r
        self.seg_interpolation = cv2.INTER_LINEAR if is_continuous else cv2.INTER_NEAREST
        self.IMAGE_VALUE = image_value

    def __call__(self, sample):
        # t0 = time.time()
        if np.random.rand() < 0.5:
            return sample
        rand_angle = np.random.randint(-self.angle_r, self.angle_r) if self.angle_r != 0 else 0
        PI = 3.141592653
        Hangle = rand_angle*PI / 180
        Hcos = math.cos(Hangle)
        Hsin = math.sin(Hangle)
        key_list = sample.keys()
        for key in key_list:
            if 'image' not in key:
                continue
            image = sample[key]
            imgsize = image.shape
            srcWidth = imgsize[1]
            srcHeight = imgsize[0]
            x = [0,0,0,0]
            y = [0,0,0,0]
            x1 = [0,0,0,0]
            y1 = [0,0,0,0]
            x[0] = -(srcWidth - 1) / 2
            x[1] = -x[0]
            x[2] = -x[0]
            x[3] = x[0]
            y[0] = -(srcHeight - 1) / 2
            y[1] = y[0]
            y[2] = -y[0]
            y[3] = -y[0]
            for i in range(4):
                x1[i] = int(x[i] * Hcos + y[i] * Hsin + 0.5)
                y1[i] = int(-x[i] * Hsin + y[i] * Hcos + 0.5)
            if (abs(y1[2] - y1[0]) > abs(y1[3] - y1[1])):
                Height = abs(y1[2] - y1[0])
                Width = abs(x1[3] - x1[1])
            else:
                Height = abs(y1[3] - y1[1])
                Width = abs(x1[2] - x1[0])
            row, col = image.shape[:2]
            m = cv2.getRotationMatrix2D(center=(col/2, row/2), angle=rand_angle, scale=1)
            new_image = cv2.warpAffine(image, m, (Width,Height), flags=cv2.INTER_LINEAR if 'image' in key else self.seg_interpolation, 
                borderValue=self.IMAGE_VALUE if 'image' in key else self.MASK_VALUE)
            sample[key] = new_image
        
        # t1 = time.time()
        # print('Rotate: %.4f'%(1000*(t1 - t0)))
        return sample

class Resize(object):
    def __init__(self,output_size,is_continuous=False,label_size=None):
        assert isinstance(output_size, (tuple,list))
        if len(output_size) == 1:
            self.output_size = (output_size[0],output_size[0])
        else:
            self.output_size = tuple(output_size)
        self.seg_interpolation = cv2.INTER_LINEAR if is_continuous else cv2.INTER_NEAREST
    
    def __call__(self,sample):
        # t0 = time.time()
        if not 'image' in sample.keys():
            return sample

        key_list = sample.keys()
        for key in key_list:
            if key != 'image':
                continue
            img = sample[key]
            h, w = img.shape[:2]
            img = cv2.resize(img, dsize=self.output_size, interpolation=cv2.INTER_LINEAR if 'image' in key else self.seg_interpolation)
            sample[key] = img

        # t1 = time.time()
        # print('Resize: %.4f'%(1000*(t1 - t0)))
        return sample

class RandomScale(object):
    def __init__(self,rand_resize,is_continuous=False):
        self.rand_resize = rand_resize
        self.seg_interpolation = cv2.INTER_LINEAR if is_continuous else cv2.INTER_NEAREST

    def __call__(self, sample):
        rand_scale = random.uniform(self.rand_resize[0], self.rand_resize[1])
        key_list = sample.keys()
        for key in key_list:
            if 'image' not in key and 'label' not in key:
                continue
            img = sample[key]
            img = cv2.resize(img, None, fx=rand_scale, fy=rand_scale, interpolation=cv2.INTER_LINEAR if 'image' in key else self.seg_interpolation)
            sample[key] = img

        if 'bbox' in key_list:
            x0, y0, x1, y1 = sample['bbox']
            x0 *= rand_scale
            x1 *= rand_scale
            y0 *= rand_scale
            y1 *= rand_scale
            sample['bbox'] = np.array([x0, y0, x1, y1])

        return sample

class RandomCrop(object):
    def __init__(self,crop_size, mask_value=0, image_value=127):
        assert isinstance(crop_size, (tuple,list))
        if len(crop_size) == 1:
            self.crop_size = (crop_size[0],crop_size[0])
        else:
            self.crop_size = crop_size
        self.MASK_VALUE = mask_value
        self.IMAGE_VALUE = image_value

    def __call__(self,sample):
        rand_pad = random.uniform(0, 1)
        key_list = sample.keys()
        for key in key_list:
            if 'image' not in key:
                continue
            img = sample[key]
            h,w = img.shape[:2]
            new_h,new_w = self.crop_size
            pad_w = new_w - w
            pad_h = new_h - h
            w_begin = max(0,-pad_w)
            h_begin = max(0,-pad_h)
            pad_w = max(0,pad_w)
            pad_h = max(0,pad_h)
            w_begin = int(w_begin * rand_pad)
            h_begin = int(h_begin * rand_pad)
            w_end = w_begin + min(w,new_w)
            h_end = h_begin + min(h,new_h)
            shape = list(img.shape)
            shape[0] = new_h
            shape[1] = new_w
            new_img = np.zeros(shape,dtype=np.float64)
            new_img.fill(self.IMAGE_VALUE)
            new_img[pad_h//2:min(h,new_h)+pad_h//2,pad_w//2:min(w,new_w)+pad_w//2] = img[h_begin:h_end,w_begin:w_end]
            sample[key] = new_img

        return sample

class RandomShift(object):
    def __init__(self,shift_pixel,mask_value,image_value):
        self.shift_pixel = shift_pixel
        self.MASK_VALUE = mask_value
        self.IMAGE_VALUE = image_value

    def __call__(self,sample):
        rand_x = int((random.uniform(0, 1)-0.5)*2*self.shift_pixel)
        rand_y = int((random.uniform(0, 1)-0.5)*2*self.shift_pixel)
        key_list = sample.keys()
        mean = sample['mean']
        for key in key_list:
            if 'image' not in key and 'label' not in key:
                continue
            img = sample[key]
            h,w = img.shape[:2]
            new_x0 = max(0,rand_x)
            new_y0 = max(0,rand_y)
            new_x1 = w + min(0, rand_x)
            new_y1 = h + min(0, rand_y)
            x0 = max(0, -rand_x)
            y0 = max(0, -rand_y)
            x1 = x0 + new_x1 - new_x0
            y1 = y0 + new_y1 - new_y0
            new_img = np.zeros_like(img, dtype=np.float)
            if 'image' in key:
                # if 'pad' in key_list:
                #     self.IMAGE_VALUE = sample['pad']
                new_img[...,0].fill(mean[0])
                new_img[...,1].fill(mean[1])
                new_img[...,2].fill(mean[2])
                # else:
                #     new_img.fill(self.IMAGE_VALUE)
            elif 'label' in key:
                new_img.fill(self.MASK_VALUE)
            new_img[new_y0:new_y1, new_x0:new_x1] = img[y0:y1, x0:x1]
            sample[key] = new_img
        
        if 'bbox' in key_list:
            x0, y0, x1, y1 = sample['bbox']
            x0 += rand_x
            x1 += rand_x
            y0 += rand_y
            y1 += rand_y
            sample['bbox'] = np.array([x0, y0, x1, y1])
        
        return sample

class CenterPad(object):
    def __init__(self, pad_size,mask_value,image_value):
        assert isinstance(pad_size, (tuple,list))
        if len(pad_size) == 1:
            self.pad_size = (pad_size[0],pad_size[0])
        else:
            self.pad_size = pad_size
        self.MASK_VALUE = mask_value
        self.IMAGE_VALUE = image_value

    def __call__(self,sample):
        key_list = sample.keys()
        for key in key_list:
            if 'image' not in key and 'label' not in key:
                continue
            img = sample[key]
            h,w = img.shape[:2]
            new_h,new_w = self.pad_size
            pad_w = new_w - w
            pad_h = new_h - h
            w_begin = max(0,-pad_w)
            h_begin = max(0,-pad_h)
            pad_w = max(0,pad_w)
            pad_h = max(0,pad_h)
            w_begin = int(w_begin * 0.5)
            h_begin = int(h_begin * 0.5)
            w_end = w_begin + min(w,new_w)
            h_end = h_begin + min(h,new_h)
            shape = list(img.shape)
            shape[0] = new_h
            shape[1] = new_w
            new_img = np.zeros(shape,dtype=np.float)
            if 'image' in key:
                new_img.fill(self.IMAGE_VALUE)
            elif 'label' in key:
                new_img.fill(self.MASK_VALUE)
            new_img[pad_h//2:min(h,new_h)+pad_h//2,pad_w//2:min(w,new_w)+pad_w//2] = img[h_begin:h_end,w_begin:w_end]
            sample[key] = new_img
        return sample

class CenterCrop(object):
    def __init__(self,crop_size,mask_value,image_value):
        assert isinstance(crop_size, (int, tuple,list))
        if isinstance(crop_size, int):
            self.crop_size = (crop_size,crop_size)
        elif isinstance(crop_size,list) and len(crop_size) == 1:
            self.crop_size = (crop_size[0],crop_size[0])
        else:
            self.crop_size = crop_size
        self.MASK_VALUE = mask_value
        self.IMAGE_VALUE = image_value
    
    def __call__(self,sample):
        key_list = sample.keys()
        mean = sample['mean']
        for key in key_list:
            if 'image' in key or 'label' in key:
                img = sample[key]
                h,w = img.shape[:2]
                new_h,new_w = self.crop_size
                
                if 'label' in key:
                    new_h = int(new_h * sample.get('resize_scale',1))
                    new_w = int(new_w * sample.get('resize_scale',1))
                pad_w = new_w - w
                pad_h = new_h - h
                w_begin = max(0,-pad_w)
                h_begin = max(0,-pad_h)
                pad_w = max(0,pad_w)
                pad_h = max(0,pad_h)
                w_begin = int(w_begin * 0.5)
                h_begin = int(h_begin * 0.5)
                w_end = w_begin + min(w,new_w)
                h_end = h_begin + min(h,new_h)
                shape = list(img.shape)
                shape[0] = new_h
                shape[1] = new_w
                new_img = np.zeros(shape,dtype=np.float)
                if 'image' in key:
                    new_img[...,0].fill(mean[0])
                    new_img[...,1].fill(mean[1])
                    new_img[...,2].fill(mean[2])
                    # new_img.fill(self.MASK_VALUE)
                elif 'label' in key:
                    new_img.fill(self.MASK_VALUE)
                new_img[pad_h//2:min(h,new_h)+pad_h//2,pad_w//2:min(w,new_w)+pad_w//2] = img[h_begin:h_end,w_begin:w_end]
                sample[key] = new_img

        if 'bbox' in key_list:
            x0, y0, x1, y1 = sample['bbox']
            x0 += pad_w//2 - w_begin
            x1 += pad_w//2 - w_begin
            y0 += pad_h//2 - h_begin
            y1 += pad_h//2 - h_begin
            sample['bbox'] = np.array([x0, y0, x1, y1])
                
        return sample

class BorderPad(object):
    def __init__(self,mask_value,image_value,size):
        self.MASK_VALUE = mask_value
        self.IMAGE_VALUE = image_value
        self.size = size
    
    def __call__(self, sample):
        key_list = sample.keys()
        mean = sample['mean']
        for key in key_list:
            if 'image' in key or 'label' in key:
                img = sample[key]
                h, w = img.shape[:2]
                if h >= self.size and w >= self.size:
                    return sample

                size = max(self.size,h,w)

                pad_w = (size - w)//2
                pad_h = (size - h)//2
                shape = list(img.shape)
                shape[0] = size
                shape[1] = size
                new_img = np.zeros(shape,dtype=np.float)
                if 'image' in key:
                    new_img[...,0].fill(mean[0])
                    new_img[...,1].fill(mean[1])
                    new_img[...,2].fill(mean[2])
                    # new_img.fill(self.MASK_VALUE)
                elif 'label' in key:
                    new_img.fill(self.MASK_VALUE)
                new_img[pad_h//2:h+pad_h//2,pad_w//2:w+pad_w//2] = img
                sample[key] = new_img

        if 'bbox' in key_list:
            x0, y0, x1, y1 = sample['bbox']
            x0 += pad_w//2
            x1 += pad_w//2
            y0 += pad_h//2
            y1 += pad_h//2
            sample['bbox'] = np.array([x0, y0, x1, y1])
                
        return sample

class CenterCropFromBBox(object):
    def __init__(self,k=1.5):
        self.k = k

    def __call__(self, sample):
        key_list = sample.keys()
        if not 'bbox' in key_list:
            return sample

        x0,y0,x1,y1 = sample['bbox']
        cx = (x0+x1)/2
        cy = (y0+y1)/2
        w = (x1 - x0)*self.k
        h = (y1 - y0)*self.k


        for key in key_list:
            if 'image' in key or 'label' in key:
                img = sample[key]
                H,W = img.shape[:2]

                x0 = max(0, int(cx - w/2))
                y0 = max(0, int(cy - h/2))
                x1 = min(W-1, int(cx + w/2))
                y1 = min(H-1, int(cy + h/2))
                
                new_img = img[y0:y1+1,x0:x1+1]
                new_img = cv2.resize(new_img, (W,H), \
                    interpolation=cv2.INTER_NEAREST if 'label' in key else cv2.INTER_LINEAR)

                sample[key] = new_img
            
        sample['bbox'] = np.array([0,0,W-1,H-1])

        return sample

class CenterCropFromRoI(object):
    def __init__(self,input_size,k=1.5):
        self.k = k
        self.input_size = input_size

    def __call__(self, sample):
        key_list = sample.keys()
        if not 'roi' in key_list:
            return sample

        roi = sample['roi']
        H,W = sample['image'].shape[:2]

        if np.random.rand() < 0.1:
            # sample['whole']
            roi = np.array([0,0,W-1,H-1])

        for key in key_list:
            if ('image' in key or 'label' in key) and not 'pre' in key:
                img = sample[key]
                H,W = img.shape[:2]

                x0 = max(0, roi[0])
                y0 = max(0, roi[1])
                x1 = min(W-1, roi[2])
                y1 = min(H-1, roi[3])
                new_img = img[y0:y1+1,x0:x1+1]
                # assert new_img.shape[0] > 0 and new_img.shape[1] > 0, '%d %d %d %d %d %d %d %d\n'%(x0,y0,x1,y1,W,H,img.shape[0], img.shape[1])
                new_img = cv2.resize(new_img, (W,H), \
                    interpolation=cv2.INTER_NEAREST if 'label' in key else cv2.INTER_LINEAR)

                sample[key] = new_img
        # sample['roi'] = np.array([x0, y0, x1, y1])
        # if not self.keep_roi:
        sample['roi'] = np.array([0,0,self.input_size[1],self.input_size[0]])

        return sample

class GetID(object):
    def __call__(self, sample):
        label = sample['label'].copy()
        sample['ID'] = list(set(label.flatten()))[1:]
        return sample

class GetHeatMap(object):
    def __init__(self, sigma=2.0):
        self.th = 4.6052 ## what's this??
        self.delta = math.sqrt(self.th * 2)
        self.sigma = sigma
        self.input_size = 481
        self.feat_size = 61
        self.radius = 3

    def __call__(self, sample):
        # t_kp = sample['top_keypoint']
        # l_kp = sample['left_keypoint']
        # b_kp = sample['bottom_keypoint']
        # r_kp = sample['right_keypoint']
        kp_key_words = ['top_keypoint', 'left_keypoint', 'bottom_keypoint', 'right_keypoint', 'center_keypoint']
        heat_key_words = ['top_heatmap', 'left_heatmap', 'bottom_heatmap', 'right_heatmap', 'center_heatmap']
        offset_key_words = ['top_offset', 'left_offset', 'bottom_offset', 'right_offset', 'to_del']
        weight_key_words = ['top_weight', 'left_weight', 'bottom_weight', 'right_weight', 'to_del']

        if sample['neg']:
            for h_key, os_key, w_key in zip(heat_key_words, offset_key_words, weight_key_words):
                sample[h_key] = np.zeros((self.feat_size, self.feat_size))
                sample[os_key] = np.zeros(2)
                sample[w_key] = np.zeros((self.feat_size, self.feat_size))
            
            sample.pop('to_del')
            return sample

        for key, h_key, os_key, w_key in zip(kp_key_words, heat_key_words, offset_key_words, weight_key_words):
            cx, cy = sample[key]
            cx, cy = cx*self.feat_size/self.input_size, cy*self.feat_size/self.input_size
            offset_x, offset_y = cx - int(cx), cy - int(cy)
            cx, cy = int(cx), int(cy)

            x0 = int(max(0, cx - self.delta*self.sigma+0.5))
            y0 = int(max(0, cy - self.delta*self.sigma+0.5))
            x1 = int(min(self.feat_size, cx + self.delta*self.sigma+1.5))
            y1 = int(min(self.feat_size, cy + self.delta*self.sigma+1.5))

            exp_factor = 1 / 2.0 / self.sigma / self.sigma
            heatmap = np.zeros((self.feat_size, self.feat_size))

            x_vec = (np.arange(x0, x1) - cx)**2
            y_vec = (np.arange(y0, y1) - cy)**2
            xv, yv = np.meshgrid(x_vec, y_vec)

            # arr_exp = (xv + yv) <= self.radius**2

            # arr_sum = exp_factor * np.maximum(xv + yv - self.radius**2, 0)
            arr_sum = exp_factor * (xv + yv)
            arr_exp = np.exp(-arr_sum)
            arr_exp[arr_sum > self.th] = 0
            heatmap[y0:y1, x0:x1] = arr_exp

            weight = np.zeros_like(heatmap)
            weight[cy,cx] = 1
            
            sample[h_key] = heatmap
            sample[os_key] = np.array([offset_x, offset_y])
            sample[w_key] = weight

        sample.pop('to_del')

        # img = sample['label']
        # n = img.sum()
        # h, w = img.shape
        # if n == 0:
        #     sample['center_keypoint'] = np.array([0.5, 0.5])
        #     sample['heatmap_label'] = np.zeros((self.size,self.size))
        #     return sample
        # gridx, gridy = np.meshgrid(range(w), range(h))
        # cx = (gridx*img).sum() / n / w
        # cy = (gridy*img).sum() / n / h
        # sample['center_keypoint'] = np.array([cx, cy])
        # cx,cy = int(cx*self.size), int(cy*self.size)

        # x0 = int(max(0, cx - self.delta*self.sigma+0.5))
        # y0 = int(max(0, cy - self.delta*self.sigma+0.5))
        # x1 = int(min(self.size, cx + self.delta*self.sigma+1.5))
        # y1 = int(min(self.size, cy + self.delta*self.sigma+1.5))

        # exp_factor = 1 / 2.0 / self.sigma / self.sigma
        # heatmap = np.zeros((self.size, self.size))

        # x_vec = (np.arange(x0, x1) - cx)**2
        # y_vec = (np.arange(y0, y1) - cy)**2
        # xv, yv = np.meshgrid(x_vec, y_vec)

        # arr_exp = (xv + yv) <= self.radius**2

        # # arr_sum = exp_factor * np.maximum(xv + yv - self.radius**2, 0)
        # arr_sum = exp_factor * (xv + yv)
        # arr_exp = np.exp(-arr_sum)
        # arr_exp[arr_sum > self.th] = 0
        # heatmap[y0:y1, x0:x1] = arr_exp
        # sample['heatmap_label'] = heatmap
        return sample

class GetMean(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        # img = sample['image']
        # mean = sample['image'].reshape(-1,3).mean(axis=0).astype(np.int)
        # mean = tuple([int(i) for i in mean])
        mean = (127,127,127)
        sample['mean'] = mean
        return sample

class GetBoundingBox(object):
    def __init__(self, with_heat=False):
        self.with_heat = with_heat

    def __call__(self, sample):
        label = sample['label']
        y, x = np.where(label > 0)
        if len(x) < 32:
            # sample['neg'] = True
            if self.with_heat:
                sample['top_keypoint'] = np.zeros(2)
                sample['left_keypoint'] = np.zeros(2)
                sample['bottom_keypoint'] = np.zeros(2)
                sample['right_keypoint'] = np.zeros(2)
                sample['center_keypoint'] = np.zeros(2)
            sample['bbox'] = np.array([0,0,label.shape[1]-1, label.shape[0]-1])
            return sample

        ind = np.argmin(x)
        l_kp = np.array([x[ind], y[ind]])
        ind = np.argmax(x)
        r_kp = np.array([x[ind], y[ind]])
        ind = np.argmin(y)
        t_kp = np.array([x[ind], y[ind]])
        ind = np.argmax(y)
        b_kp = np.array([x[ind], y[ind]])
        ct_kp = np.array([(l_kp[0] + r_kp[0]) // 2, (t_kp[1] + b_kp[1]) // 2])
        bbox = np.array([l_kp[0], t_kp[1], r_kp[0], b_kp[1]])

        if self.with_heat:
            sample['top_keypoint'] = t_kp
            sample['left_keypoint'] = l_kp
            sample['bottom_keypoint'] = b_kp
            sample['right_keypoint'] = r_kp
            sample['center_keypoint'] = ct_kp

        sample['bbox'] = bbox

        return sample

class GetROIs(object):
    def __init__(self, shift_ratio=0.2, scale=0.1):
        self.shift_ratio = shift_ratio
        self.scale = scale
        self.ext = 1.5

    def rand_shift(self):
        if np.random.rand() < 0.1:
            return 2*(np.random.rand()-0.5)*self.shift_ratio*2
        return 2*(np.random.rand()-0.5)*self.shift_ratio
    
    def rand_scale(self):
        return 1 + 2*(np.random.rand()-0.5)*self.scale

    def __call__(self, sample):
        x0, y0, x1, y1 = sample['bbox']
        shape = sample['label'].shape
        w = x1 - x0 + 1
        h = y1 - y0 + 1
        cx = (x1 + x0) / 2
        cy = (y1 + y0) / 2
        cx += self.rand_shift()*w
        cx = max(0,min(cx, shape[1]-1))
        cy += self.rand_shift()*h
        cy = max(0,min(cy, shape[0]-1))
        w = self.rand_scale()*(w*self.ext)
        h = self.rand_scale()*(h*self.ext)
        x0 = max(0,int(cx - w/2))
        x1 = min(shape[1]-1, int(x0 + w - 1))
        y0 = max(0,int(cy - h/2))
        y1 = min(shape[0]-1, int(y0 + h - 1))
        sample['roi'] = np.array([int(x0),int(y0),int(x1),int(y1)])
        return sample

class GetVerifyBBox(object):
    def __init__(self, K=1000, num_pos=16, num_total=64, thres_high=0.7, thres_low=0.5):
        self.K = K
        self.num_pos = num_pos
        self.num_total = num_total
        self.thres_high = thres_high
        self.thres_low = thres_low

    def __call__(self, sample):
        bbox = sample['bbox']
        cx_gt = (bbox[2] + bbox[0]) // 2
        cy_gt = (bbox[3] + bbox[1]) // 2
        w_gt = bbox[2] - bbox[0]
        h_gt = bbox[3] - bbox[1]

        if self.K == 1:
            sample['verify_label'] = np.array([1], dtype=np.int32)
            sample['verify_bbox'] = bbox.copy().reshape((1,4))
            return sample

        img = sample['image']
        image_h, image_w = img.shape[:2]

        cx = np.random.rand(self.K)*image_w
        cy = np.random.rand(self.K)*image_h
        w = np.maximum(1,np.random.rand(self.K)*image_w)
        h = np.maximum(1,np.random.rand(self.K)*image_h)


        cx_ = cx_gt + 0.2*(np.random.rand(self.K)-0.5)*w_gt
        cy_ = cy_gt + 0.2*(np.random.rand(self.K)-0.5)*h_gt
        w_ = np.maximum(1,w_gt*(np.random.rand(self.K)*1.5+0.5))
        h_ = np.maximum(1,h_gt*(np.random.rand(self.K)*1.5+0.5))

        cx = np.concatenate([cx, cx_], axis=0)
        cy = np.concatenate([cy, cy_], axis=0)
        w = np.concatenate([w, w_], axis=0)
        h = np.concatenate([h, h_], axis=0)

        x0 = np.maximum(0, cx - w/2)
        y0 = np.maximum(0, cy - h/2)
        x1 = np.minimum(image_w, cx + w/2)
        y1 = np.minimum(image_h, cy + h/2)

        area1 = np.maximum(0, x1 - x0) * np.maximum(0, y1 - y0)
        area2 = np.maximum(0, bbox[2] - bbox[0]) * np.maximum(0, bbox[3] - bbox[1])
        l = np.maximum(x0, bbox[0])
        r = np.minimum(x1, bbox[2])
        t = np.maximum(y0, bbox[1])
        d = np.minimum(y1, bbox[3])
        inter = np.maximum(0, r - l) * np.maximum(0, d - t)
        iou = inter / (area1 + area2 - inter)

        pos = np.where(iou > self.thres_high)
        neg = np.where(iou < self.thres_low)


        def select(x, keep_num):
            num = x[0].shape[0]
            if  num <= keep_num:
                return x, num
                
            keep = np.arange(num)
            np.random.shuffle(keep)
            keep = keep[:keep_num]
            return tuple(y[keep] for y in x), keep_num

        pos, num_pos = select(pos, self.num_pos)
        # print(iou[pos].mean(), num_pos)
        neg, num_neg = select(neg, self.num_total - num_pos)

        label = np.zeros(self.num_total)
        label[:num_pos] = 1

        x0_p = x0[pos]
        x1_p = x1[pos]
        y0_p = y0[pos]
        y1_p = y1[pos]

        x0_n = x0[neg]
        x1_n = x1[neg]
        y0_n = y0[neg]
        y1_n = y1[neg]

        bbox_pos = np.stack([x0_p, y0_p, x1_p, y1_p], axis=-1)
        bbox_neg = np.stack([x0_n, y0_n, x1_n, y1_n], axis=-1)

        sample['verify_label'] = label
        sample['verify_bbox'] = np.concatenate([bbox_pos, bbox_neg], axis=0)

        return sample

class RandomMorphologyEx(object):
    def __init__(self, max_times=5):
        self.max_times = max_times
    
    def __call__(self, sample):
        label = sample['label']
        kernel = np.ones((5,5),np.uint8) 
        if np.random.rand() > 0.5:
            func = [cv2.erode, cv2.dilate]
        else:
            func = [cv2.dilate, cv2.erode]
        for f in func:
            t = int(np.random.rand()*self.max_times + 1)
            label = f(label,kernel,iterations = t)
        sample['label'] = label
        return sample

class RandomTranspose(object):
    def __init__(self,trans_prob=0.0):
        self.trans_prob = 1.0 - trans_prob
    def __call__(self, sample):
        prob = np.random.rand()
        step = (1.0 - self.trans_prob)/3
        count = 0
        if prob <= self.trans_prob:
            count = 0
        elif prob > self.trans_prob and prob <= self.trans_prob+step:
            count = 1
        elif prob > self.trans_prob+step and prob <= self.trans_prob+2*step:
            count = 2
        else:
            count = 3
        key_list = sample.keys()
        for key in key_list:
            if 'image' not in key and 'label' not in key:
                continue
            image = sample[key]
            image_trans = np.rot90(image, k=count)
            sample[key] = image_trans
        return sample

class ColorJitter(object):
    def __init__(self, brightness=None, contrast=None, saturation=None, *args, **kwargs):
        if not brightness is None and brightness>0:
            self.brightness = [max(1-brightness, 0), 1+brightness]
        if not contrast is None and contrast>0:
            self.contrast = [max(1-contrast, 0), 1+contrast]
        if not saturation is None and saturation>0:
            self.saturation = [max(1-saturation, 0), 1+saturation]

    def __call__(self, im_lb):
        im = im_lb['image']
        temp_im = Image.fromarray(im.astype(dtype=np.uint8),mode='RGB')
        r_brightness = random.uniform(self.brightness[0], self.brightness[1])
        r_contrast = random.uniform(self.contrast[0], self.contrast[1])
        r_saturation = random.uniform(self.saturation[0], self.saturation[1])
        temp_im = ImageEnhance.Brightness(temp_im).enhance(r_brightness)
        temp_im = ImageEnhance.Contrast(temp_im).enhance(r_contrast)
        temp_im = ImageEnhance.Color(temp_im).enhance(r_saturation)
        im = np.array(temp_im)
        im_lb['image'] = im
        return im_lb

def onehot(label, num):
    m = label
    one_hot = np.eye(num)[m]
    return one_hot

class Multiscale(object):
    def __init__(self, rate_list):
        self.rate_list = rate_list

    def __call__(self, sample):
        image = sample['image']
        row, col, _ = image.shape
        image_multiscale = []
        for rate in self.rate_list:
            rescaled_image = cv2.resize(image, None, fx=rate, fy=rate, interpolation=cv2.INTER_LINEAR)
            sample['image_%f'%rate] = rescaled_image
        return sample

class AugLastFrame(object):
    def __init__(self):
        self.mor = RandomMorphologyEx()
        
    def __call__(self, sample):
        tmp_sample = {'label': sample['last_label']}
        tmp_sample = self.mor(tmp_sample)
        sample['last_label'] = tmp_sample['label']
        return sample

class GetOptFlow(object):
    def __call__(self,sample):
        img_p = sample['last_image']
        img = sample['image']
        flow = cv2.calcOpticalFlowFarneback(img_p,img, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mask = sample['last_label']
    
    
class RandomAffine(object):
    """Randomly affine label"""
    def __init__(self, empty_mask,affine_ratio, is_optflow=True, is_continuous=False):
        self.empty_mask = empty_mask
        self.affine_ratio = affine_ratio
        self.optflow = is_optflow
        self.dis_opt_flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
        self.seg_interpolation = cv2.INTER_LINEAR if is_continuous else cv2.INTER_NEAREST
    
    def getrand(self):
        return self.affine_ratio * (2*np.random.rand() - 1)
    
    def OpticalFlowFusion(self, prvs, next, mask):
        h,w,_ = prvs.shape
        prvs = cv2.cvtColor(prvs, cv2.COLOR_BGR2GRAY)
        next = cv2.cvtColor(next, cv2.COLOR_BGR2GRAY)
        flow = self.dis_opt_flow.calc(prvs, next, None)
        idx_x, idx_y = np.where(mask == 1)
            # new_mask = np.zeros_like(mask)
        for x,y in zip(idx_x, idx_y):
            dy, dx = flow[x,y]
            newx = min(max(0,int(x + dx)),h-1)
            newy = min(max(0,int(y + dy)),w-1)
            mask[newx, newy] = 1
            
        return mask
    
    def __call__(self, sample):
        is_empty = (np.random.rand() < self.empty_mask)
    
        if 'last_label' in sample.keys():
            label = sample['last_label']
            if is_empty:
                new_label = np.zeros_like(label)
            else:
                rows,cols = label.shape
                pts1 = np.float32([[0,0],[0,cols],[rows,0],[rows,cols]])
                pts2 = np.float32([[self.getrand()*rows,self.getrand()*cols], \
                                   [self.getrand()*rows,(1-self.getrand())*cols], \
                                   [(1-self.getrand())*rows,self.getrand()*cols], \
                                   [(1-self.getrand())*rows,(1-self.getrand())*cols]])
                M = cv2.getPerspectiveTransform(pts1,pts2)
                new_label = cv2.warpPerspective(label,M,(cols,rows),flags=self.seg_interpolation,borderValue=0)
                
                if self.optflow:
                    img = sample['image'].astype(np.uint8)
                    prvs = cv2.warpPerspective(img,M,(cols,rows),flags=self.seg_interpolation,borderValue=127)
                    new_label = self.OpticalFlowFusion(prvs, img, new_label)
    
            new_label[np.where(new_label == 255)] = 0
            sample['last_label'] = new_label
    
        if 'last_gray_image' in sample.keys():
            gray = sample['last_gray_image']
            if is_empty:
                new_gray = np.zeros_like(gray)
            else:
                new_gray = cv2.warpPerspective(gray,M,(cols,rows),flags=cv2.INTER_LINEAR,borderValue=0)
            sample['last_gray_image'] = new_gray
    
        return sample
    
class RandomHole(object):
    """Randomly crop holes in label"""
    def __init__(self, hole_p, hole_num, hole_ratio, hole_area, is_continuous=False):
        self.hole_num = hole_num
        self.hole_ratio = hole_ratio
        self.hole_area = hole_area
        self.hole_p = hole_p
        self.seg_interpolation = cv2.INTER_LINEAR if is_continuous else cv2.INTER_NEAREST
    
    def __call__(self, sample):
        if np.random.rand() > self.hole_p:
            return sample
    
        if 'last_label' in sample.keys():
            label = sample['last_label']
            rows,cols = label.shape
            hole = np.ones_like(label)
            num = int(np.random.rand()*self.hole_num + 0.5)
            for i in range(num):
                area = np.sum(label)
                area *= np.random.rand() * (self.hole_area[1] - self.hole_area[0]) + self.hole_area[0]
                r = (self.hole_ratio-1) * np.random.rand() + 1
                a = int(math.sqrt(r*area))
                b = int(math.sqrt(area/r))
                cx = int(np.random.rand()*cols)
                cy = int(np.random.rand()*rows)
                angle = int(np.random.rand()*180)
                cv2.ellipse(hole, (cx, cy), (a, b), angle, 0, 360, 0, -1)
    
            new_label = label * hole
            sample['last_label'] = new_label
    
        return sample

import numpy as np
import torch
import cv2
import pickle
# import imageio

'''
{ imread, imwrite, imsize, flipud, rot90, transpose, modcrop, augment
  float32, uint8, normalize, denormalize, ndarray2tensor, tensor2ndarray}
'''

'''
# --------------------------------------------
# functions used to read/write an image (size: C-H-W, class: numpy, type: uint8)
# --------------------------------------------
'''
def imread(path):
    extension = path.split('.')[-1].lower()
    if extension in ['jpg', 'jpeg', 'png', 'bmp', 'tif']:
        # read image with cv2
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        # for gray image, expand dimentions from (H, W) to (H, W, 1)
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)
        # from BGR to RGB
        if img.shape[2] == 3:
            img = img[:, :, [2, 1, 0]]
        # from H-W-C to C-H-W
        img = np.transpose(img, (2, 0, 1))
        # img = np.ascontiguousarray(np.transpose(img, (2, 0, 1)))
    elif extension in ['pt']:
        # read image with pickle
        with open(path, 'rb') as _f:
            img = pickle.load(_f)
        # for gray image, expand dimentions from (H, W) to (H, W, 1)
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)
        # from H-W-C to C-H-W
        img = np.transpose(img, (2, 0, 1))
    elif extension in ['npy']:
        # read image with numpy
        img = np.load(path)
        # for gray image, expand dimentions from (H, W) to (H, W, 1)
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)
        # from H-W-C to C-H-W
        img = np.transpose(img, (2, 0, 1))
    else:
        raise NotImplementedError
    return img


def imwrite(img, path):
    extension = path.split('.')[-1].lower()
    if extension in ['jpg', 'jpeg', 'png', 'bmp', 'tif']:
        # from C-H-W to H-W-C
        img = np.transpose(img, (1, 2, 0))
        # img = np.ascontiguousarray(np.transpose(img, (1, 2, 0)))
        # for color images, from RGB to BGR 
        if img.shape[2] == 3:
            img = img[:, :, [2, 1, 0]]
        # for gray images, squeeze the dimension
        if img.shape[2] == 1:
            img = np.squeeze(img)
        # save the specified image
        cv2.imwrite(path, img)
    # elif package == 'imageio':
    #     pass
    else:
        raise NotImplementedError

'''
# --------------------------------------------
# functions used to convert image to rgb/gray type
# --------------------------------------------
'''
def to_rgb(img):
    if img.shape[0] == 1:
        return np.concatenate([img] * 3, 0)
    else:
        return img

def to_gray(img):
    if img.shape[0] == 3:
        return img.mean(axis=0, keepdims=True).astype('uint8')
        # return np.expand_dims(np.mean(img, axis=0), axis=0).astype('uint8')
    else:
        return img

'''
# --------------------------------------------
# functio used to get size of an image
# --------------------------------------------
'''
def imsize(img):
    return list(img.shape)


'''
# --------------------------------------------
# functions used to flip/rotate/transpose/modcrop an image
# --------------------------------------------
'''
def flipud(img):
    if isinstance(img, np.ndarray):
        axis = len(img.shape) - 2
        return np.flip(img, axis=axis)
        # return np.flip(img, axis=axis).copy()
    if isinstance(img, torch.Tensor):
        dim = len(img.shape) - 2
        return torch.flip(img, dims=dim)

def rot90(img, k=1):
    if isinstance(img, np.ndarray):
        axes = (len(img.shape)-2, len(img.shape)-1)
        return np.rot90(img, k=k, axes=axes)
        # return np.rot90(img, k=k, axes=axes).copy()
    if isinstance(img, torch.Tensor):
        dims = [len(img.shape)-2, len(img.shape)-1]
        return torch.rot90(img, k=k, dims=dims)


# from (C, H, W) to (C, W, H) or from (H, W) to (W, H)
def transpose(img):
    if isinstance(img, np.ndarray):
        if len(img.shape) == 2:
            return np.transpose(img, (1, 0))
            # return np.transpose(img, (1, 0)).copy()
            # return np.ascontiguousarray(np.transpose(img, (1, 0)))
        elif len(img.shape) == 3:
            return np.transpose(img, (0, 2, 1))
            # return np.transpose(img, (0, 2, 1)).copy()
            # return np.ascontiguousarray(np.transpose(img, (0, 2, 1)))
        else:
            pass
    if isinstance(img, torch.Tensor):
        if len(img.shape) == 2:
            return torch.transpose(img, 0, 1)
        elif len(img.shape) == 3:
            return torch.transpose(img, 1, 2)
            # return torch.transpose(x, 1, 2).clone()
        else:
            pass


def modcrop(img, scale):
    if len(img.shape) == 2:
        H, W = imsize(img)
        H, W = H - H % scale, W - W % scale
        img = img[:H, :W]
    elif len(img.shape) == 3:
        _, H, W = imsize(img)
        H, W = H - H % scale, W - W % scale
        img = img[:, :H, :W]
    else:
        pass
    return img


'''
# --------------------------------------------
# functions used to augment data and add noise
# --------------------------------------------
'''
def augment(img, mode = 0):
    if mode == 0:
        return img
    elif mode == 1:
        return rot90(img, k=1)
    elif mode == 2:
        return rot90(img, k=2)
    elif mode == 3:
        return rot90(img, k=3)
    elif mode == 4:
        return flipud(img)
    elif mode == 5:
        return flipud(rot90(img, k=1))
    elif mode == 6:
        return flipud(rot90(img, k=2))
    elif mode == 7:
        return flipud(rot90(img, k=3))
    else:
        raise NotImplementedError


'''
# --------------------------------------------
# functions used to convert data type
# --------------------------------------------
'''
def float32(data):
    if isinstance(data, np.ndarray):
        return data.astype('float32')
    if isinstance(data, torch.Tensor):
        return data.float()

def uint8(data):
    if isinstance(data, np.ndarray):
        return data.astype('uint8')
    if isinstance(data, torch.Tensor):
        return data.byte()   # tensor.byte() --> uint8; tensor.char() --> int8


'''
# --------------------------------------------
# functions used for normalization
# --------------------------------------------
'''
def normalize(img, factor=255):
    if isinstance(img, np.ndarray):
        return img.astype('float32')/factor
    if isinstance(img, torch.Tensor):
        return img.float()/factor

def denormalize(img, factor=255):
    if isinstance(img, np.ndarray):
        return (img * factor).clip(0, factor).astype('uint8')
    if isinstance(img, torch.Tensor):
        return (img * factor).clamp(0, factor).byte()  


'''
# --------------------------------------------
# functions used to convert data class
# --------------------------------------------
'''
def ndarray2tensor(img): 
    # return torch.from_numpy(img)                          # Error: stride is negative
    return torch.from_numpy(img.copy())                     # Recommended solution
    # return torch.from_numpy(np.ascontiguousarray(img))    # Another solution

def tensor2ndarray(img):
    return img.numpy()


'''
# --------------------------------------------
# functions used to add noise
# --------------------------------------------
'''
def random_uniform(low=0.0, high=1.0, size=[], dclass='ndarray', device=torch.device("cpu")):
    if dclass == 'ndarray':
        return np.random.rand(*size).astype('float32') * (high - low) + low # '*' for '*args'
    elif dclass == 'tensor':
        return torch.rand(size, dtype=torch.float32, device = device) * (high - low) + low
    else:
        pass

def random_uniform_like(img, low=0.0, high=1.0):
    if isinstance(img, np.ndarray):
        return np.random.rand(*img.shape).astype(img.dtype) * (high - low) + low
    elif isinstance(img, torch.Tensor):
        return torch.rand_like(img) * (high - low) + low
    else:
        pass

def random_gauss(mu=0.0, sigma=1.0, size=[], dclass='ndarray', device=torch.device("cpu")):
    if dclass == 'ndarray':
        return np.random.randn(*size).astype('float32') * sigma + mu
    elif dclass == 'tensor':
        return torch.rand(size, dtype=torch.float32, device = device) * sigma + mu
    else:
        pass

def random_gauss_like(img, sigma=1.0, mu=0.0):
    if isinstance(img, np.ndarray):
        return np.random.randn(*img.shape).astype(img.dtype) * sigma + mu
    elif isinstance(img, torch.Tensor):
        return torch.randn_like(img) * sigma + mu
    else:
        pass

def random_poiss_like(img):
    if isinstance(img, np.ndarray):
        return np.random.poisson(lam=img).astype(img.dtype)
    elif isinstance(img, torch.Tensor):
        return torch.poisson(img)       # only test for torch.float32
    else:
        pass

def add_noise(img, has_Poisson = False, sigma_Gauss = None, sigma_pred = None, sigma_row = None, has_uniform = False):

    if has_Poisson:
        img = np.random.poisson(lam=img).astype(img.dtype)

    if sigma_Gauss is not None:
        noise = np.random.randn(*img.shape).astype(img.dtype) * sigma_Gauss
        img   = img + noise

    if sigma_pred is not None:
        if isinstance(sigma_pred, list):
            if len(sigma_pred) == 1:
                sigma_pred = [sigma_pred[0], sigma_pred[0]]
            elif len(sigma_pred) == 2:
                sigma_pred = sigma_pred
            else:
                raise NotImplementedError
        else:
            sigma_pred = [sigma_pred, sigma_pred]
        dim_x = np.arange(img.shape[2]).reshape(1,1,img.shape[2])
        for c in range(img.shape[0]):
            noise  = sigma_pred[0] * np.random.randn() * np.sin(np.pi/2*dim_x, dtype=img.dtype) # np.float32
            noise += sigma_pred[1] * np.random.randn() * np.cos(np.pi/2*dim_x, dtype=img.dtype)
            img[c] = img[c] + noise

    if sigma_row is not None:
        noise = np.random.randn(img.shape[0], 1, img.shape[2]).astype(img.dtype) * sigma_row
        img   = img + noise

    if has_uniform:
        noise = np.random.rand(*img.shape).astype(img.dtype) - 0.5
        img   = img + noise

    return img

# ---------------------------------------------------
# functions used to calculate PSNR with images (tensor version)
# ---------------------------------------------------
def cal_PSNR(img1, img2, peak, border=0, to_YCbCr=False):
    h, w  = img1.shape[-2], img1.shape[-1]
    diff  = (img1 - img2)/peak
    if to_YCbCr:
        diff = (65.738*diff[0] + 129.057*diff[1] + 25.064*diff[2])/256
    valid = diff[..., border:h-border, border:w-border]
    mse = valid.pow(2).mean()
    if mse == 0:
        return float('inf')
    else:
        return -10 * torch.log10(mse)

def cal_PSNRs(imgs1, imgs2, peak, border=0, to_YCbCr=False):
    N = imgs1.shape[0]
    psnrs = torch.zeros(N).float()
    for i in range(N):
        img1, img2 = imgs1[i], imgs2[i]
        psnr = cal_PSNR(img1, img2, peak, border, to_YCbCr)
        psnrs[i] = psnr
    return psnrs

def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)

# ---------------------------------------------------
# functions used to calculate PSNR with images (numpy version)
# ---------------------------------------------------
def cal_psnr_for_numpy(img1, img2, peak, border=0, to_YCbCr=False):
    h, w = img1.shape[-2], img1.shape[-1]
    diff = (img1 - img2)/peak
    if diff.shape[0] == 3 and to_YCbCr:
        diff = (65.738*diff[0] + 129.057*diff[1] + 25.064*diff[2])/256
    valid = diff[..., border:h-border, border:w-border]
    mse = np.mean(np.square(valid))
    return -10 * np.log10(mse)

def cal_ssim_for_numpy(img1, img2, border=0, to_YCbCr=False):
    h, w = img1.shape[-2], img1.shape[-1]
    img1 = img1[..., border:h-border, border:w-border]
    img2 = img2[..., border:h-border, border:w-border]
    if img1.shape[0] == 1:
        return ssim(np.squeeze(img1), np.squeeze(img2))
    elif img1.shape[0] == 3:
        if to_YCbCr:
            img1 = (65.738*img1[0] + 129.057*img1[1] + 25.064*img1[2])/256
            img2 = (65.738*img2[0] + 129.057*img2[1] + 25.064*img2[2])/256
            return ssim(img1, img2)
        else:
            ssims = []
            for c in range(3):
                ssims.append(ssim(img1[c], img2[c]))
            return np.array(ssims).mean()
    else:
        raise NotImplementedError

def ssim(img1, img2):
    C1 = (0.01 * 255)**2
    C2 = (0.03 * 255)**2

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())

    mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
    mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = cv2.filter2D(img1**2, -1, window)[5:-5, 5:-5] - mu1_sq
    sigma2_sq = cv2.filter2D(img2**2, -1, window)[5:-5, 5:-5] - mu2_sq
    sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                            (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean()
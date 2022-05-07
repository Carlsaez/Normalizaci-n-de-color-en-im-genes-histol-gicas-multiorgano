### Cálculo métrica SSIM
from skimage.metrics import structural_similarity as ssim
from skimage.color import rgb2gray
import numpy as np

def SSIM(x, y):
    x = np.array(x)
    y = np.array(y)
    ssim_const = ssim(x, y, data_range=y.max() - y.min(), multichannel=True)
    return ssim_const

### Pearson correlation coefficient
def pearson_correlation(x, y):
    x = rgb2gray(np.array(x))
    y = rgb2gray(np.array(y))
    corr = np.corrcoef(x.flatten(), y.flatten())[0, 1]
    return corr
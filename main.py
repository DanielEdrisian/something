import math
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import skimage as sk
import skimage.io as skio
import skimage.transform
import skimage.filters
import scipy.signal
import scipy.ndimage
from skimage.color import rgb2gray

subdirectory_name = ''

# Gets image from image name
def get_im_source(imname, gray = True):
    # read in the image
    im = skio.imread(f'inputs/{subdirectory_name}/{imname}', as_gray = gray)

    # convert to double (might want to do this later on to save memory)    
    im = sk.img_as_float(im)
    return im

# Saves image automatically using the sub-directory name specified globally
def save(imname, im):
  cleaned_imname = imname.replace('.jpeg', '').replace('.jpg', '').replace('.png', '')
  skio.imsave(f'outputs/{subdirectory_name}/{cleaned_imname}.jpeg', sk.img_as_ubyte(im))

# Convoles c onto input im
def convolve(im, c):
  return scipy.signal.convolve2d(im, c, mode='same', boundary='fill', fillvalue=0)

# Blurs im using sigma a
def blur(im, a):
  return skimage.filters.gaussian(im, a, mode='reflect', multichannel = True, truncate = 4)

def gaussian(a):
  x = np.zeros((6, 6))
  x[2, 2] = 1
  return scipy.ndimage.gaussian_filter(x, a, mode="constant")

# Interpolates image so that the minimum becomes 0 and max becomes 1.0 by default
def interp(im, min=0, max=1.0):
  return np.interp(im, (im.min(), im.max()), (min, max))

## Part 1.1

subdirectory_name = '1.1'

# Makes edges out of the image
def edgify(imname, im, t=0.15):
  im_dx = convolve(im, [[1, -1]])
  save('dx-' + imname, im_dx)

  im_dy = convolve(im, [[1], [-1]])
  save('dy-' + imname, im_dy)

  # Does the magnitude of gradient
  im_edge = (im_dx ** 2 + im_dy ** 2) ** 0.5
  save('edge-' + imname, interp(im_edge))
  
  # Binarizes the result
  im_edge = np.where(im_edge > t, 1.0, 0.0)
  save('binarized-edge-' + imname, im_edge)

# Edgify the camera man
cam_imname = 'cameraman.png'
im = get_im_source(cam_imname)
edgify(cam_imname, im)

## Part 1.2

subdirectory_name = '1.2'

# Get the camera man, the blur it, then edgify it
im = get_im_source(cam_imname)
im = convolve(im, gaussian(1))
save('cameraman-blur.jpeg', im)
edgify(cam_imname, im, 0.09)

# Makes edges using derivative of gaussian convolution
def dog_edgify(imname, im, t=0.09):
  g = gaussian(1)

  # Convolves Dx and Dy onto the gaussian
  dx, dy = np.array([[1, -1]]), np.array([[1],[-1]])
  dx, dy = convolve(g, dx), convolve(g, dy)
  save('blurrdy', interp(dy))
  save('blurrdx', interp(dx))

  im_dx = convolve(im, dx)
  save('dx-' + imname, interp(im_dx))

  im_dy = convolve(im, dy)
  save('dy-' + imname, interp(im_dy))

  # Does the magnitude of gradient
  im_edge = (im_dx ** 2 + im_dy ** 2) ** 0.5
  save('dog-edge-' + imname, interp(im_edge))

  # Binarizes the result
  im_edge = np.where(im_edge > t, 1.0, 0.0)
  save('binarized-dog-edge-' + imname, im_edge)

im = get_im_source(cam_imname)
dog_edgify(cam_imname, im)

## Part 2.1

subdirectory_name = '2.1'

# Sharpens the image using the image name and the factor
def sharpen(imname, factor):
  im = get_im_source(imname, False)
  im_blur = blur(im, im.shape[0]/500.0)
  im_high = im-im_blur
  im_sharp = im+factor*im_high
  im_sharp = np.interp(im_sharp, (im_sharp.min(), im_sharp.max()), (-0.2, 1.2))
  im_sharp = np.where(im_sharp < 0.0, 0.0, im_sharp)
  im_sharp = np.where(im_sharp > 1.0, 1.0, im_sharp)
  save('sharp-'+imname, im_sharp)

# Sharpen taj mahal and the dog
sharpen('taj.jpeg', 1.5)
sharpen('dog.jpeg', 1.0)

# Blur a sharp image, then sharpen it
blur_sharp_im = blur(get_im_source('sharp.jpeg', False), 3)
save('blur-sharp.jpeg', blur_sharp_im)
sharpen('blur-sharp.jpeg', 5.0)

## Part 2.2

subdirectory_name = '2.2'

# Gets the FFT of the image
def im_fft(im):
  im_fft = np.log(np.abs(np.fft.fftshift(np.fft.fft2(rgb2gray(im)))))
  return np.interp(im_fft, (im_fft.min(), im_fft.max()), (0.0, 1.0))

# Combines two images on top of each other and saves the intermediate steps
# Takes a blur sigma value for the two images
def combine_images(im1_name, im2_name, t1, t2, do_intermediates = False):
  im1, im2 = get_im_source(im1_name, False), get_im_source(im2_name, False)

  # Turns the first image into grayscale
  im1 = rgb2gray(im1)
  im1 = np.expand_dims(im1, 2)
  im1 = np.append(im1, np.append(im1, im1, axis=2), axis=2)
  im1_blur, im2_blur = blur(im1, t1), blur(im2, t2)
  im2_high = im2 - im2_blur
  im_combined = (im1_blur + im2_high) / 2
  save(f'{im1_name}-{im2_name}-combined.jpeg', im_combined)

  # Skips if this is false
  if not do_intermediates:
    return

  im_combined_fft = np.log(np.abs(np.fft.fftshift(np.fft.fft2(rgb2gray(im_combined)))))
  im_combined_fft = np.interp(im_combined_fft, (im_combined_fft.min(), im_combined_fft.max()), (0.0, 1.0))

  save(f'{im1_name}-fft.jpeg', im_fft(im1))
  save(f'{im2_name}-fft.jpeg', im_fft(im2))
  save(f'{im1_name}-blur-fft.jpeg', im_fft(im1_blur))
  save(f'{im2_name}-high-fft.jpeg', im_fft(im2_high))
  save(f'{im1_name}-{im2_name}-combined-fft.jpeg', im_combined_fft)

# Combines three sets og images on top of each other
combine_images('DerekPicture-aligned.jpeg', 'nutmeg-aligned.jpeg', 10, 10, True)
combine_images('eagle-aligned.jpeg', 'airplane-aligned.jpeg', 15, 15)
combine_images('denero-aligned.jpeg', 'josh-aligned.jpeg', 20, 20)


## Part 2.3

subdirectory_name = '2.3'

# Gets gaussian and laplacian stacks for input image
def stacks(im):
  curr = im
  blur_stack = [curr]
  lapl_stack = []
  for i in range(6):
    curr = blur(im, 2**i)
    blur_stack.append(curr)
  for i in range(6):
    lapl_stack.append(blur_stack[i]-blur_stack[i+1])
  blur_stack.pop(0)
  return blur_stack, lapl_stack

# Creates a rectangular mask given the vertical and horizontal boundaries
def mask(im, x0=0.0, x1=1.0, y0=0.0, y1=1.0):
  mask = np.zeros_like(im)
  def maskit(i, j, x):
    mask[i, j] = x
  w, h = mask.shape[0], mask.shape[1]
  [[maskit(j, i, 1.0) for i in range(w) if int(x0*w)+1 <= i <= int(x1*w) and int(y0*h) <= j <= int(y1*h)] for j in range(h)]
  return mask

# Alpha blends the two images and saves the laplacians in the middle
def alpha_blend(imname1, imname2):
  im1, im2 = get_im_source(imname1, False), get_im_source(imname2, False)
  im_mask = mask(im1, x1=0.5)
  im_mask = blur(im_mask, 16)
  
  # Splits in two
  im1, im2 = im_mask * im1, (1-im_mask) * im2

  save(f'{imname1}-blend', im1)
  save(f'{imname2}-blend', im2)

  # Save the first image's stack
  im_mask_stack_blur, _ = stacks(im_mask)
  im1_stack_blur, im1_stack_laplace = stacks(im1)
  [save(f'{imname1}-gaussian-stack-{i}', v) for i, v in enumerate(im1_stack_blur)]
  [save(f'{imname1}-laplacian-stack-{i}', interp(v)) for i, v in enumerate(im1_stack_laplace)]

  # Save second image's stack
  im2_stack_blur, im2_stack_laplace = stacks(im2)
  [save(f'{imname2}-gaussian-stack-{i}', v) for i, v in enumerate(im2_stack_blur)]
  [save(f'{imname2}-laplacian-stack-{i}', interp(v)) for i, v in enumerate(im2_stack_laplace)]
  save(f'{imname1}-{imname2}-blend', interp(im1 + im2))

  # Save the blended laplacians stack
  blended_stack = [interp(im_mask_stack_blur[i] * im1_stack_laplace[i] + (1 - im_mask_stack_blur[i]) * im2_stack_laplace[i]) for i in range(len(im1_stack_laplace))]
  [save(f'{imname1}-{imname2}-laplacian-blend-{i}', im) for i, im in enumerate(blended_stack)]

# Blends the Apple and Orange using alpha blending
alpha_blend('apple.jpeg', 'orange.jpeg')

## Part 2.4

subdirectory_name = '2.4'

# Creates a rectangular mask given the vertical and horizontal boundaries
def mask(im, x0=0.0, x1=1.0, y0=0.0, y1=1.0):
  mask = np.zeros_like(im)
  def maskit(i, j, x):
    mask[i, j] = x
  w, h = mask.shape[1], mask.shape[0]
  [[maskit(j, i, 1.0) for i in range(w) if int(x0*w) <= i <= int(x1*w) and int(y0*h) <= j <= int(y1*h)] for j in range(h)]
  return mask

# Modifies an input mask given the vertical and horizontal boundaries
def maskify(mask, x0=0.0, x1=1.0, y0=0.0, y1=1.0):
  def maskit(i, j, x):
    mask[i, j] = x
  w, h = mask.shape[1], mask.shape[0]
  [[maskit(j, i, 0.0) for i in range(w) if not (int(x0*w) <= i <= int(x1*w) and int(y0*h) <= j <= int(y1*h))] for j in range(h)]
  return mask

# Creates a circular mask given the radius and center offset
def circle_mask(im, r=0.5, dx=0.5, dy=0.5):
  mask = np.zeros_like(im)
  def maskit(i, j, x):
    mask[i, j] = x
  w, h = mask.shape[1], mask.shape[0]
  print(w, h)
  print(dx * w, dy * h)
  [[maskit(j, i, 1.0) for i in range(w) if math.sqrt((i - dx * w)**2 + (j - dy * h)**2) <= r*min(w,h)] for j in range(h)]
  return mask

# Modifies an input mask into a circular mask given the radius and center offset
def circlify_mask(mask, r=0.5, dx=0.5, dy=0.5):
  def maskit(i, j, x):
    mask[i, j] = x
  w, h = mask.shape[1], mask.shape[0]
  [[maskit(j, i, 0.0) for i in range(w) if math.sqrt((i - dx * w)**2 + (j - dy * h)**2) > r*min(w,h)] for j in range(h)]
  return mask

# Blends two images using laplacian stack blending as shown in the paper
# Also saves intermediate steps
def blend(imname1, imname2, im_mask, show_trace = False):
  subdir = f'{imname1}-{imname2}/'

  # Clean the sub directory name
  subdir = subdir.replace('.jpeg', '').replace('.jpg', '').replace('.png', '')
  im1, im2 = get_im_source(subdir + imname1, False), get_im_source(subdir + imname2, False)
  mask = im_mask(im1)
  save('mask', mask)

  # Create stacks for mask, image 1 and image 2
  im_mask_stack_blur, _ = stacks(mask)
  im1_stack_blur, im1_stack_laplace = stacks(im1)
  im2_stack_blur, im2_stack_laplace = stacks(im2)

  # Saves the trace if true
  if show_trace:
    [save(f'{subdir}/{imname1}-gaussian-stack-{i}', interp(v)) for i, v in enumerate(im1_stack_blur)]
    [save(f'{subdir}/{imname1}-laplacian-stack-{i}', interp(v)) for i, v in enumerate(im1_stack_laplace)]
    [save(f'{subdir}/{imname2}-gaussian-stack-{i}', interp(v)) for i, v in enumerate(im2_stack_blur)]
    [save(f'{subdir}/{imname2}-laplacian-stack-{i}', interp(v)) for i, v in enumerate(im2_stack_laplace)]
    [save(f'{subdir}/mask-stack-{i}', interp(v)) for i, v in enumerate(im_mask_stack_blur)]
    save(f'{subdir}/{imname1}-{imname2}-blend', np.clip(im1 + im2, 0.0, 1.0))

  blended_stack = [im_mask_stack_blur[i] * im1_stack_laplace[i] + (1 - im_mask_stack_blur[i]) * im2_stack_laplace[i] for i in range(len(im1_stack_laplace))]
  [save(f'{subdir}/{imname1}-{imname2}-laplacian-blend-{i}', interp(im)) for i, im in enumerate(blended_stack)]

  # Combines the laplacians together
  combined = blended_stack[0]
  for a in blended_stack[1:]:
    combined += a

  # Adds the base gaussians to the combinations
  combined += im1_stack_blur[-1] * im_mask_stack_blur[-1] + im2_stack_blur[-1] * (1 - im_mask_stack_blur[-1])
  combined = interp(combined)
  save(f'{subdir}/{imname1}-{imname2}-blend-combined', combined)

# Blends the oraple
blend('apple.jpeg', 'orange.jpeg', lambda x: mask(x, x1=0.5), True)

# Blends the sun in sunflower using a circle mask
blend('sun.jpeg', 'flower.jpeg', lambda x: circle_mask(x, r = 0.4))

# Blends the brain in walnut using a half circle made of two maskings
blend('brain.jpeg', 'walnut.jpeg', lambda x: circlify_mask(mask(x, x1=0.5), r=0.35, dx=0.5, dy=0.5), True)

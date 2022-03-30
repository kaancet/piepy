import numpy as np
from tqdm import tqdm
import tifffile as tf
import matplotlib.pyplot as plt
from os.path import join as pjoin
from skimage.measure import block_reduce

def get_vasculature(vasc_path):
    vasc_image = tf.imread(vasc_path)
    try:
        vasc_image = vasc_image[:,:,0]
    except:
        pass
    plt.imshow(vasc_image,cmap='gray')
    return vasc_image

def downsample_movie(movie,block_size=2,func=np.mean):
    if not isinstance(block_size,list):
        block_size = (block_size,block_size)
        
    temp_down = block_reduce(movie[0,:,:],block_size=block_size,func=func)
    
    downsampled_movie = np.zeros((movie.shape[0],temp_down.shape[0],temp_down.shape[1]))
    
    for f in range(movie.shape[0]):
        frame = movie[f,:,:]
        downsampled_movie[f,:,:] = block_reduce(frame,block_size=block_size,func=func)
        
    return downsampled_movie

def fft_movie(movie, component=1, output_raw=False):
    movief = np.fft.fft(movie,axis=0)
    if output_raw:
        return movief[component]
    phase = -1. * np.angle(movief[component]) % (2*np.pi)
    mag = (np.abs(movief[component]) * 2.)/len(movie)

    return mag,phase

def visual_sign_map(phasemap1, phasemap2):
    gradmap1 = np.gradient(phasemap1)
    gradmap2 = np.gradient(phasemap2)

    graddir1 = np.zeros(np.shape(gradmap1[0]))
    graddir2 = np.zeros(np.shape(gradmap2[0]))

    for i in range(phasemap1.shape[0]):
        for j in range(phasemap2.shape[1]):
            graddir1[i,j] = math.atan2(gradmap1[1][i,j],gradmap1[0][i,j])
            graddir2[i,j] = math.atan2(gradmap2[1][i,j],gradmap2[0][i,j])

    vdiff = np.multiply(np.exp(1j * graddir1),np.exp(-1j * graddir2))
    areamap = np.sin(np.angle(vdiff))
    return areamap

def make_square(img):
    """ Crops an image or image array to be a square"""
    if len(img.shape)>2:
        frames = img
    elif len(img.shape)==2:
        frames = img.reshape(1,img.shape[0],img.shape[1])
    else:
        raise ValueError(f'Input image shape is not valid f{img.shape}')
        
    im_height = frames.shape[1]
    im_width = frames.shape[2]
    if im_width == im_height:
        print(f'No need to crop, image is already square {im_height}x{im_width}')
        return 0
    elif im_width > im_height:
        shape_diff = im_width-im_height
        crop_points = [int(np.floor(shape_diff/2)),int(np.floor(shape_diff/2+im_height))]
        new_img = np.zeros((frames.shape[0],im_height,im_height))
    else:
        shape_diff = im_height - im_width
        crop_points = [int(np.floor(shape_diff/2)),int(np.floor(shape_diff/2+im_width))]
        new_img = np.zeros((frames.shape[0],im_width,im_width))
    
    for depth in range(frames.shape[0]):
        if im_width > im_height:
            new_img[depth,:,:] = frames[depth,:,crop_points[0]:crop_points[1]]
        elif im_width < im_height:
            new_img[depth,:,:] = frames[depth,crop_points[0]:crop_points[1],:]
    if len(img.shape)==2:
        return new_img[0,:,:]
    else:
        return new_img

def read_avgs(experiment_dir,downsample=0,downsample_func=np.mean):
    avg_mov_dict = {}
    if downsample:
        with tqdm(total=len(os.listdir(experiment_dir))) as pbar:
            for exp in os.listdir(experiment_dir):
                
                if '.DS' in exp:
                    continue
                elif '.tif' in exp:
                    continue
                else:
                    key = exp.split('_')[-1]
                    pbar.set_description(key)
                    avg_file = pjoin(experiment_dir,exp,'movies','avg_01.tif')
                    temp_mov = tf.imread(avg_file)

                    try:
                        avg_mov_dict[key] = temp_mov[:,0,:,:]
                    except:
                        avg_mov_dict[key] = temp_mov

                    if downsample:
                        avg_mov_dict[key] = downsample_movie(avg_mov_dict[key],block_size=downsample,func=downsample_func)
                pbar.update()

    return avg_mov_dict

def plot_mag_phase_maps(phase_dict,mag_dict,clim_percentages=[30,60]):
    f,axs = plt.subplots(2,len(phase_dict.keys()),figsize=(15,15))

    cr = [int(phase_dict['M2T'].shape[0]*clim_percentages[0]/100),int(phase_dict['M2T'].shape[1]*clim_percentages[1]/100)]

    for i,k in enumerate(phase_dict.keys()):
        p_img = phase_dict[k]
        ax_p = axs[0][i].imshow(p_img)
        ax_p.set_clim(np.percentile(p_img[cr[0]:cr[1],cr[0]:cr[1]],5),np.percentile(p_img[cr[0]:cr[1],cr[0]:cr[1]],95))
        axs[0][i].set_axis_off()

        m_img = mag_dict[k]
        ax_m = axs[1][i].imshow(m_img)
        ax_m.set_clim(np.percentile(m_img[cr[0]:cr[1],cr[0]:cr[1]],5),np.percentile(m_img[cr[0]:cr[1],cr[0]:cr[1]],95))
        axs[1][i].set_axis_off()

    f.colorbar(ax_p,ax=axs[0,:],pad=0.05,location='bottom')
    f.colorbar(ax_m,ax=axs[1,:],pad=0.05,location='bottom')




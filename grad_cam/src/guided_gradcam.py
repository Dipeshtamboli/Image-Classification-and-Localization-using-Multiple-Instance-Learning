"""
This programs helps to test the grad_cam on different examples

author: Dipesh Tamboli - https://github.com/Dipeshtamboli
author: Parth Patil    - https://github.com/Parth1811
"""
# sftp://test1@10.107.42.42/home/Drive2/amil/my_network/net.py
import numpy as np
from misc_functions import (get_example_params, convert_to_grayscale, save_gradient_images)
from gradcam import GradCam
from guided_backprop import GuidedBackprop
from net import Net

def guided_grad_cam(grad_cam_mask, guided_backprop_mask):
    """
        Guided grad cam is just pointwise multiplication of cam mask and
        guided backprop mask

    Args:
        grad_cam_mask (np_arr): Class activation map mask
        guided_backprop_mask (np_arr):Guided backprop mask
    """
    cam_gb = np.multiply(grad_cam_mask, guided_backprop_mask)
    return cam_gb


if __name__ == '__main__':
    # Get params
    class_no=6                  #denseresidential
    image_no=84
    check_target_class=1

    (original_image, prep_img, target_class, file_name_to_export, pretrained_model) =\
        get_example_params(class_no,image_no,check_target_class)

    # Grad cam
    target_layer=35
    gcv2 = GradCam(pretrained_model, target_layer=target_layer)
    # Generate cam mask
    cam = gcv2.generate_cam(prep_img, target_class)
    print('Grad cam completed')

    # Guided backprop
    GBP = GuidedBackprop(pretrained_model)
    # Get gradients
    guided_grads = GBP.generate_gradients(prep_img, target_class)
    print('Guided backpropagation completed')

    # Guided Grad cam
    cam_gb = guided_grad_cam(cam, guided_grads)
    save_gradient_images(cam_gb, file_name_to_export +",layer="+str(target_layer) +'_Guided_Grad_Cam')
    grayscale_cam_gb = convert_to_grayscale(cam_gb)
    save_gradient_images(grayscale_cam_gb, file_name_to_export +",layer="+str(target_layer)+ '_Guided_Grad_Cam_gray')
    save_gradient_images(guided_grads, file_name_to_export+",layer="+str(target_layer)+'_guided_grads   ')
    print('Guided grad cam completed')

import multiprocessing
import os
from tqdm import tqdm
import cv2 as cv2
import numpy as np
from scipy import ndimage as ndi 
from skimage import io, color, filters
from skimage.measure import regionprops, label
from skimage.morphology import disk, remove_small_objects, dilation, erosion
from skimage.segmentation import clear_border

import pandas as pd
import re

import math


def get_area(a, b): # returns None if rectangles don't intersect
#     dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin)
#     dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin)
    dx = min(a[2], b[2]) - max(a[0], b[0])
    dy = min(a[3], b[3]) - max(a[1], b[1])
    if (dx>=0) and (dy>=0):
        return dx*dy

# Check whether two bounding boxes intersect. If there is an intersection, 
# return the overlap degree IOU; if there is no intersection, return 0.
def bb_overlab(x1, y1, w1, h1, x2, y2, w2, h2):
    '''
    In the image, the x-axis ranges from 0 to positive infinity from left to right, 
    the y-axis ranges from 0 to positive infinity from top to bottom, 
    the width w extends from left to right, 
    and the height h extends from top to bottom. 

    '''
    if(x1>x2+w2):
        return 0
    if(y1>y2+h2):
        return 0
    if(x1+w1<x2):
        return 0
    if(y1+h1<y2):
        return 0
    colInt = abs(min(x1 +w1 ,x2+w2) - max(x1, x2))
    rowInt = abs(min(y1 + h1, y2 +h2) - max(y1, y2))
    overlap_area = colInt * rowInt
    area1 = w1 * h1
    area2 = w2 * h2
    return round((overlap_area / (area1 + area2 - overlap_area)), 2)

def display_object_from_original(img, label_img):

    mask = label_img > 0

    r = img[:, :, 0] * mask
    g = img[:, :, 1] * mask
    b = img[:, :, 2] * mask

    img_temp = np.dstack([r, g, b])
    
    return img_temp


base_mask_folder = "***" # the folder of the series benchmark
base_mask_list = os.listdir(base_mask_folder)


series_image_path = "***" # the folder of the series image
series_final_mask = "***" # the folder of the series mask
series_traits_folder = "***" # the output folder of the series traits

for base_mask_name in tqdm(base_mask_list):


    folder_name = base_mask_name.split('_')[0]

    time_list = []
    num_list = []
    area_list = []
    perimeter_list = []
    diameter_list = []
    compacthness_list = []
    length_list = []
    width_list = []
    l_w_list = []
    roundness_list = []
    centroid_list = []
    color_list = []

    base_mask = io.imread(os.path.join(base_mask_folder, base_mask_name))

    img = color.rgb2gray(base_mask)
    thresh_yen_img = filters.threshold_yen(img)
    yen_binary_img = img>thresh_yen_img
    get_base_mask = dilation(yen_binary_img, disk(1))
    get_base_mask = get_base_mask.astype(bool)
    get_base_mask = remove_small_objects(get_base_mask, min_size=50)

    mask_folder = os.path.join(series_final_mask, folder_name)
    series_mask_list = os.listdir(mask_folder)

    image_folder = os.path.join(series_image_path, folder_name)
    series_image_list = os.listdir(image_folder)

    traits_folder = os.path.join(series_traits_folder, folder_name)
    os.makedirs(traits_folder, exist_ok=True)

    for image_name in tqdm(series_mask_list):

        nan_area_list = []
        nan_perimeter_list = []
        nan_diameter_list = []
        nan_compacthness_list = []
        nan_length_list = []
        nan_width_list = []
        nan_l_w_list = []
        nan_roundness_list = []
        nan_centroid_list = []
        nan_color_list = []

        nan_bbox_list = []

        base_bbox_list = []
        base_num_list = []

        seg_mask, num =ndi.label(get_base_mask)
        seg_regions = regionprops(seg_mask)

        base_num = 1
        for final_seg_region in seg_regions:
            
            minr, minc, maxr, maxc = final_seg_region.bbox
            x1 = minr-25
            x2 = minc-25
            y1 = maxr+25
            y2 = maxc+25
            base_bbox = x1, x2, y1, y2
            base_bbox_list.append(base_bbox)
            base_num_list.append(base_num)
            
            base_num = base_num + 1
            
            nan_area_list.append('nan')
            nan_bbox_list.append('nan')
            nan_perimeter_list.append('nan')
            nan_diameter_list.append('nan')
            nan_compacthness_list.append('nan')
            nan_length_list.append('nan')
            nan_width_list.append('nan')
            nan_l_w_list.append('nan')
            nan_roundness_list.append('nan')
            nan_centroid_list.append('nan')
            nan_color_list.append('nan')

        test_area_list = nan_area_list
        test_perimeter_list = nan_perimeter_list
        test_diameter_list = nan_diameter_list
        test_compacthness_list = nan_compacthness_list
        test_length_list = nan_length_list
        test_width_list = nan_width_list
        test_l_w_list = nan_l_w_list
        test_roundness_list = nan_roundness_list
        test_centroid_list = nan_centroid_list
        test_color_list = nan_color_list

        get_test_bbox_list = nan_bbox_list

        get_date = image_name.split('.')[0]
        image_path = os.path.join(mask_folder, get_date+'.png')
        image = io.imread(image_path)

        series_image_path = os.path.join(image_folder, image_name)
        series_image = io.imread(series_image_path)

        #####add#####
        if len(image.shape) == 3:

            img = color.rgb2gray(image)
            thresh_yen_img = filters.threshold_yen(img)
            image = img>thresh_yen_img

        else:
            image = image
        #####add end#####
        
        image = dilation(image, disk(1))
        image = image.astype(bool)
        image = remove_small_objects(image, min_size=50)

        test_bbox_list = []

        seg_mask, num =ndi.label(image)
        seg_regions = regionprops(seg_mask)

        for final_seg_region in seg_regions:

            minr, minc, maxr, maxc = final_seg_region.bbox
            
            x1 = minr-25
            x2 = minc-25
            y1 = maxr+25
            y2 = maxc+25
            test_bbox = x1, x2, y1, y2
            test_bbox_list.append(test_bbox)
        
        for i in range(len(base_bbox_list)):
            
            num_id = i + 1
            
            bbox1 = base_bbox_list[i]
            new_centroid_mask = np.zeros(seg_mask.shape)
            
            for j in range(len(test_bbox_list)):
                bbox2 = test_bbox_list[j]
                overlap = get_area(bbox1, bbox2)
                
                if overlap!=None:
                    if overlap > 0.65 and test_area_list[i]=='nan':
                        minr, minc, maxr, maxc = bbox2
                        if minr-5<0 or minc-5<0 or maxr+5>seg_mask.shape[0] or maxc+5>seg_mask.shape[1]:
                            continue
                        else:
                            nematode_part = series_image[minr-5:maxr+5, minc-5:maxc+5]

                            mask2 = image[minr-5:maxr+5, minc-5:maxc+5]
                            mask2 = clear_border(mask2)

                            color_part = display_object_from_original(nematode_part, erosion(mask2, disk(1)))
                            color_area = erosion(mask2, disk(1)).sum()
                            nematode_hsv = color.rgb2hsv(color_part)
                            nematode_saturation = nematode_hsv[:, :, 1]
                            region_saturation = nematode_saturation.sum()/(color_area)  
                            test_color_list[i] = region_saturation

                            new_centroid_mask[minr-5:maxr+5, minc-5:maxc+5] = mask2
                            
                            new_test_label_blob, num = ndi.label(new_centroid_mask)
                            new_test_regions = regionprops(new_test_label_blob)
                            for new_test_region in new_test_regions:
                                
                                pi = math.pi
                                area = new_test_region.area
                                centroid = (round(new_test_region.centroid[0],2), round(new_test_region.centroid[1],2))

                                perimeter = new_test_region.perimeter
                                diameter = 2 * ((area / math.pi) ** (1 / 2))
                                compacthness = (perimeter ** 2) / area  #
                                major_length = new_test_region.major_axis_length  #length

                                minor_length = new_test_region.minor_axis_length   #width
                                if minor_length == 0:
                                    major_minor = 0
                                else:
                                    major_minor = major_length/minor_length           
                                roundness = (4*pi*area)/(perimeter*perimeter)


                                coord = new_test_region.coords
                                if area in test_area_list:
                                    continue
                                else:

                                    test_area_list[i] = area
                                    test_centroid_list[i] = centroid
                                    test_perimeter_list[i] = np.round(perimeter, 3)
                                    test_diameter_list[i] = np.round(diameter, 3)
                                    test_compacthness_list[i] = np.round(compacthness, 3)
                                    test_length_list[i] = np.round(major_length, 3)
                                    test_width_list[i] = np.round(minor_length, 3)
                                    test_l_w_list[i] = np.round(major_minor, 3)
                                    test_roundness_list[i] = np.round(roundness, 3)

                                    get_test_bbox_list[i] = bbox2
                    else:
                        continue

                else:
                    continue
        
        time_list.append(get_date)
        area_list.append(test_area_list)
        perimeter_list.append(test_perimeter_list)
        diameter_list.append(test_diameter_list)
        compacthness_list.append(test_compacthness_list)
        length_list.append(test_length_list)
        width_list.append(test_width_list)
        l_w_list.append(test_l_w_list)
        roundness_list.append(test_roundness_list)
        centroid_list.append(test_centroid_list)
        color_list.append(test_color_list)

    for i in range(len(area_list)):
        count_times = []
        for j in area_list[i]:
            count_times.append(area_list[i].count(j))
        m = max(count_times)
        number = len(area_list[i]) - m
        num_list.append(number)

    df_number = pd.DataFrame({'Qunatity': num_list})
    df_number.insert(0, 'Date', time_list)
    sorted_df_number = df_number.sort_values(by='Date')
    sorted_df_number.to_csv(os.path.join(traits_folder, 'Quantity.csv'), encoding='gbk')

    df_perimeter = pd.DataFrame(perimeter_list)
    df_perimeter.insert(0, 'Date', time_list)
    sorted_df_perimeter = df_perimeter.sort_values(by='Date')
    sorted_df_perimeter.to_csv(os.path.join(traits_folder, 'perimeter.csv'), encoding='gbk')

    df_diameter = pd.DataFrame(diameter_list)
    df_diameter.insert(0, 'Date', time_list)
    sorted_df_diameter = df_diameter.sort_values(by='Date')
    sorted_df_diameter.to_csv(os.path.join(traits_folder, 'diameter.csv'), encoding='gbk')

    df_compacthness = pd.DataFrame(compacthness_list)
    df_compacthness.insert(0, 'Date', time_list)
    sorted_df_compacthness = df_compacthness.sort_values(by='Date')
    sorted_df_compacthness.to_csv(os.path.join(traits_folder, 'compacthness.csv'), encoding='gbk')

    df_length = pd.DataFrame(length_list)
    df_length.insert(0, 'Date', time_list)
    sorted_df_length = df_length.sort_values(by='Date')
    sorted_df_length.to_csv(os.path.join(traits_folder, 'length.csv'), encoding='gbk')

    df_width = pd.DataFrame(width_list)
    df_width.insert(0, 'Date', time_list)
    sorted_df_width = df_width.sort_values(by='Date')
    sorted_df_width.to_csv(os.path.join(traits_folder, 'width.csv'), encoding='gbk')

    df_l_w = pd.DataFrame(l_w_list)
    df_l_w.insert(0, 'Date', time_list)
    sorted_df_l_w = df_l_w.sort_values(by='Date')
    sorted_df_l_w.to_csv(os.path.join(traits_folder, 'l_w_ratio.csv'), encoding='gbk')

    df_roundness = pd.DataFrame(roundness_list)
    df_roundness.insert(0, 'Date', time_list)
    sorted_df_roundness = df_roundness.sort_values(by='Date')
    sorted_df_roundness.to_csv(os.path.join(traits_folder, 'roundness.csv'), encoding='gbk')
        
    df_location = pd.DataFrame(centroid_list)
    df_location.insert(0, 'Date', time_list)
    sorted_df_location = df_location.sort_values(by='Date')
    sorted_df_location.to_csv(os.path.join(traits_folder, 'location.csv'), encoding='gbk')

    df_color = pd.DataFrame(color_list)
    df_color.insert(0, 'Date', time_list)
    sorted_df_color = df_color.sort_values(by='Date')
    sorted_df_color.to_csv(os.path.join(traits_folder, 'saturation.csv'), encoding='gbk')
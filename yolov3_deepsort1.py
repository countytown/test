import cv2
import numpy  as np
import torch
from detector.yolov3.utils.utils import  xyxy2xywh
from seetings_globals import   Global
G = Global

def run(rtsp_info,deepsort ,target_exits,index ):
    '''
    all_rtsp_detect_result = [
                [[p1,conf1,img1，cls1],[p2,conf2,,img2，cls2]],#rtsp1
                [[p3,conf3,,img3，cls3],[p4,conf4,,img4，cls4]] #rtsp2
                ]
    rtsp_i_detect_result  = [
                [p1,conf1,img1，cls1],[p2,conf2,,img2，cls2]
                ]#rtsp1
    rtsp_i_detect_result[0] =
    '''
    '''
    1. 显示图片和obj无关，只需一张图片即可
    2. 每次执行都只针对一个rtsp的话，全局变量内每次都只有[0]，也就不需要遍历了，每次都取G[0][][]
    3. else中的全局变量形式和if中不同，所以取[]的方法也不一样
        cv2.imshow(str(index),G.info_for_sort[0][0][3]) #if
        cv2.waitKey(1)
    '''
    if target_exits == True:
        list_boxs = []
        list_confs = []
        list_clss = []
        for obj in G.info_for_sort[0]:
            bbox = np.array(obj[0])  #list->numpy ndarray
            conf_cpu_tensor = obj[1].cpu()
            cls_cpu_tensor = obj[2].cpu()
            np_conf = conf_cpu_tensor.numpy()
            np_cls = cls_cpu_tensor.numpy()

            list_boxs.append(bbox)
            list_confs.append(np_conf)
            list_clss.append(np_cls)
        nparr_list_boxs = np.array(list_boxs)
        nparr_list_boxs.reshape((-1,4))
        print("nparr_matrix_box 是", nparr_list_boxs)
        print("nparr_matrix_box 类型是", type(nparr_list_boxs))

        nparr_list_boxs = xyxy2xywh(nparr_list_boxs)
        deepsort.update(nparr_list_boxs,
                        list_confs,
                        G.info_for_sort[0][0][3],
                        list_clss,index)
    else:
        '''
        如果发现未检测出目标时图像卡住，说明上面if只处理了反之的情况，就要打开else的注释
        '''
        # pass
        img = G.info_for_sort[0][3]
        cv2.imshow(str(index),img)
        cv2.waitKey(1)

'''
cannot unpack non-iterable numpy.float64 object
'''




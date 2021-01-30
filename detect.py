from detector.yolov3.models import  *
from detector.yolov3.utils.datasets import  *
import trackor.deep_sort.deep_sort
import yolov3_deepsort1 as ds
import trackor.deep_sort.deep_sort as dp
from prepare_deepsort import prepare_deepsort
from seetings_globals import   Global
G = Global
import argparse

def detect(save_img=False):
    # deepsort = init_deepsort()
    imgsz = (320, 192) if ONNX_EXPORT else opt.img_size  # (320, 192) or (416, 256) or (608, 352) for (height, width)
    out, source, weights, half, view_img, save_txt = opt.output, opt.source, opt.weights, opt.half, opt.view_img, opt.save_txt
    webcam = source == '0' or source.startswith('rtsp') or source.startswith('http') or source.endswith('.txt') #规定输入视频的格式
    if os.path.isfile(source):
        G.rtsp_nums = len(open(opt.source, 'rU').readlines())
        if G.rtsp_nums:
            for i in range(G.rtsp_nums):
                G.info_for_sort.append([]) #初始化info数组为2维
                # print(len( G.info_for_sort),"---------")
    # Initialize
    device = torch_utils.select_device(device='cpu' if ONNX_EXPORT else opt.device)
    if os.path.exists(out):
        shutil.rmtree(out)  # delete output folder if the path already exits
    os.makedirs(out)  # make new output folder

    # Initialize model
    model = Darknet(opt.cfg, imgsz)

    # Load weights;   2 kinds of weights file
    attempt_download(weights)
    if weights.endswith('.pt'):  # pytorch format
        model.load_state_dict(torch.load(weights, map_location=device)['model'])
    else:  # darknet format
        load_darknet_weights(model, weights)

    # Second-stage classifier
    #第二阶段分类器 "第二阶段"是指在对图片进行裁剪后再进行分类？
    classify = False
    if classify:
        modelc = torch_utils.load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])  # load weights
        modelc.to(device).eval()

    # Eval mode
    model.to(device).eval()

    #2D卷积+BatchNorm2d  为什么要归一化 处理？特征差距过大导致网络性能不稳定 https://zhuanlan.zhihu.com/p/27627299
    # fuse什么意思？  模型融合，结合多个模型的优点，综合判断分类结果。 这里用了哪些模型怎么看？？？
    # Fuse Conv2d + BatchNorm2d layers
    # model.fuse()

    # Export mode
    if ONNX_EXPORT:
        model.fuse()
        img = torch.zeros((1, 3) + imgsz)  # (1, 3, 320, 192)
        f = opt.weights.replace(opt.weights.split('.')[-1], 'onnx')  # *.onnx filename
        torch.onnx.export(model, img, f, verbose=False, opset_version=11,
                          input_names=['images'], output_names=['classes', 'boxes'])

        # Validate exported model
        import onnx
        model = onnx.load(f)  # Load the ONNX model
        onnx.checker.check_model(model)  # Check that the IR is well formed
        print(onnx.helper.printable_graph(model.graph))  # Print a human readable representation of the graph
        return

    # Half precision
    half = half and device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        print("执行了webcam")
        view_img = True
        torch.backends.cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz)
    else:
        print("没有执行webcam")
        #save_img = True
        view_img = True #存储改为实时显示
        dataset = LoadImages(source, img_size=imgsz)

    # Get names and colors
    names = load_classes(opt.names)
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    # Run inference  识别/预测  ？？
    t0 = time.time()
    img = torch.zeros((1, 3, imgsz, imgsz), device=device)  # init img
    _ = model(img.half() if half else img.float()) if device.type != 'cpu' else None  # run once
    # for i, value in enumerate(dataset):
    #     path = value[0]
    #     img = value[1]
    #     im0s = value[2]
    #     vid_cap = value[3]
    #     print(path,"[[[[[[[[[[[")
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = torch_utils.time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        t2 = torch_utils.time_synchronized()

        # to float
        if half:
            pred = pred.float()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres,
                                   multi_label=False, classes=opt.classes, agnostic=opt.agnostic_nms)

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections

        '''
        pred是两个摄像头
        '''
        for i, det in enumerate(pred):  # detections for image i  i表示第i个rtsp
            # if G.rtsp_nums:
            #     print(len(G.info_for_sort),"llllll")
            # print(len(pred)) #pred长度为2
            if webcam:  # batch_size >= 1
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
            else:
                p, s, im0 = path, '', im0s

            save_path = str(Path(out) / Path(p).name)
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  #  normalization gain whwh
            if det is not None and len(det):  #注意这个if!!!!!!!
                target_exits = True #用来标志有无检测出目标

                # Rescale boxes from imgsz to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string


                # Write results
                '''
                xyxy是画的框 xywh是存的框0~1
                xyxy只是一个框，怎么把一帧里面的所有框都找出来
                det是一个rtsp内的一帧的检测结果，在1个det循环内append所有xyxy，conf,cls即可？
                '''
                 #当前rtsp帧的信息
                this_frame_info_for_sort  = []
                print("这一帧有",len(det),"个box信息")
                for *xyxy, conf, cls in reversed(det):
                    print("执行det循环")
                    # if save_txt:  # Write to file
                    #     xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                    #     info_for_sort = (xywh, conf, im0, cls)
                    #     print(info_for_sort,"222222222")
                    #     print("66666666")
                    #     with open(save_path[:save_path.rfind('.')] + '.txt', 'a') as file:
                    #         file.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format
                    info_for_sort = []
                    '''  
                    检测结果在这显示，可暂时不删
                    '''
                    # print(len(xyxy),len(conf),len(xyxy),len(xyxy),"00000")
                    # if save_img or view_img:  # Add bbox to image
                    # label = '%s %.2f' % (names[int(cls)], conf)
                    # plot_one_box(xyxy, im0, label=label, color=colors[int(cls)])
                    xyxy = (torch.tensor(xyxy).view(1, 4)).view(-1).tolist()  # 这行不可少
                    # print(len(det),"44444444") #一帧内的所有目标
                    info_for_sort.append(xyxy)
                    info_for_sort.append(conf)
                    info_for_sort.append(cls)
                    info_for_sort.append(im0)
                    this_frame_info_for_sort.append(info_for_sort) #一帧内的所有box的信息
                    G.info_for_sort[0] = this_frame_info_for_sort #测试总是0的情况
                    # print(this_frame_info_for_sort[0][1],"一帧内某个box的conf")
                # cv2.imshow(str(i), G.info_for_sort[i][0][3])
                # cv2.waitKey(1)
                ds.run(G.info_for_sort,deepsort,target_exits = True,index = i)   #放在这应该是可以的，因为已经执行完det的循环了
            else: #当前没有检测出目标 det为空
                print("未检测出目标")
                this_frame_info_for_sort = []
                for j in range(G.rtsp_nums):
                    this_frame_info_for_sort.append(0)
                    this_frame_info_for_sort.append(0)
                    this_frame_info_for_sort.append(0)
                    this_frame_info_for_sort.append(im0)
                G.info_for_sort[0] = this_frame_info_for_sort #第i个rtsp的信息
                # cv2.imshow(str(i), G.info_for_sort[i][3])
                # cv2.waitKey(1)
                ds.run(G.info_for_sort, deepsort, target_exits = False,index = i)
                # pass

        # for i in range(len(G.info_for_sort)):
        #     print( G.info_for_sort[i][0][0]) #三维坐标访问第一个rtsp的第一个box的置信度
        # ds.run(G.info_for_sort,deepsort)
        #应该在这还是在上面？？？
        # ds.run(G.info_for_sort,deepsort,target_exits)
        # G.info_for_sort = []
            # Print time (inference + NMS)
            # print('%sDone. (%.3fs)' % (s, t2 - t1))

            # Stream results
            # if view_img:
            #     cv2.imshow(p, im0)
            #     if cv2.waitKey(1) == ord('q'):  # q to quit
            #         raise StopIteration
            #
            # # Save results (image with detections)
            # if save_img:
            #     if dataset.mode == 'images':
            #         cv2.imwrite(save_path, im0)
            #     else:
            #         if vid_path != save_path:  # new video
            #             vid_path = save_path
            #             if isinstance(vid_writer, cv2.VideoWriter):
            #                 vid_writer.release()  # release previous video writer
            #
            #             fps = vid_cap.get(cv2.CAP_PROP_FPS)
            #             w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            #             h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            #             vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*opt.fourcc), fps, (w, h))
            #         vid_writer.write(im0)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    #parser.add_argument('--cfg', type=str, default='cfg/yolov3-spp.cfg', help='*.cfg path')
    parser.add_argument('--cfg', type=str, default='./detector/yolov3/cfg/yolov3-2.cfg', help='*.cfg path')
    parser.add_argument('--names', type=str, default='./detector/yolov3/data/hat.names', help='*.names path')
    #parser.add_argument('--weights', type=str, default='weights/yolov3-spp-ultralytics.pt', help='weights path')
    parser.add_argument('--weights', type=str, default="./detector/weights/best.pt", help='weights path')
    #改为无人机视频流地址
    # parser.add_argument('--source', type=str, default='rtsp://192.168.1.168/0', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--source', type=str, default='rtsp_home.txt', help='source')  # input file/folder, 0 for webcam
    #parser.add_argument('--source', type=str, default='data/samples', help='source')  # input file/folder, 0 for webcam
    parser.add_argument('--output', type=str, default='output', help='output folder')  # output folder
    parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.3, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--fourcc', type=str, default='mp4v', help='output video codec (verify ffmpeg support)')
    parser.add_argument('--half', action='store_true', help='half precision FP16 inference')
    parser.add_argument('--device', default='', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument("--deepsortweight", type=str, default="./trackor/weights/ckpt.t7")
    parser.add_argument("--config_deepsort", type=str, default="./trackor/deep_sort/cfg/deep_sort.yaml")


    opt = parser.parse_args()
    opt.cfg = check_file(opt.cfg)  # check file
    opt.names = check_file(opt.names)  # check file
    print(opt)


    deepsort = prepare_deepsort(opt.deepsortweight,opt.config_deepsort)
    with torch.no_grad():
        detect()


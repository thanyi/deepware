import cv2
import dlib
import os
import datetime
import time
import threading

dlib_classifier_path = "shape_predictor_68_face_landmarks.dat"  # 人脸识别模型路径
LOCAL_VIDEO = 0
CAMERA_STREAM = 1


class dealImgThread(threading.Thread):
    image_count = 0
    face_details = []

    def __init__(self, frame, outPath, detector, predictor, timeF, frame_count):
        threading.Thread.__init__(self)
        self.frame = frame
        self.outPath = outPath
        self.detector = detector
        self.predictor = predictor
        self.timeF = timeF
        self.frame_count = frame_count

    def run(self):
        dots = self.detector(self.frame, 1)
        backup = dealImgThread.face_details[:]
        for k, d in enumerate(dots):
            shape = self.predictor(self.frame, d)
            # 排除静态人脸,如画像
            isSame = False
            for i in range(len(backup)):
                same_count = 0
                for p_pt, n_pt in zip(backup[i].parts(), shape.parts()):
                    if p_pt.x == n_pt.x and p_pt.y == n_pt.y:
                        same_count += 1
                if same_count >= 10:
                    isSame = True
                    break
            if self.frame_count == self.timeF:
                dealImgThread.face_details.append(shape)

            if not isSame and self.frame_count != self.timeF:
                save_img = self.frame[int(d.top()):int(d.top() + d.height()), int(d.left()):int(d.left() + d.width())]
                if len(self.detector(save_img, 1)) == 1:  # 再次识别,提高精度
                    try:
                        threading.Lock().acquire()
                        save_img = cv2.resize(save_img, (299, 299))
                        dealImgThread.image_count += 1
                        cv2.imwrite(self.outPath + "/" + str(dealImgThread.image_count) + '.png', save_img)
                        threading.Lock().release()
                    except Exception:
                        pass


def setModelPath(path):
    global dlib_classifier_path
    dlib_classifier_path = path


def loadModel():
    global dlib_classifier_path
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(dlib_classifier_path)
    return detector, predictor


def recognition(vc, outPath, detector, predictor, timeF, mode, timeout=60, total=100, multiThread=False):
    # 加载模型
    if mode:
        begin_time = time.time()
    if not os.path.exists(outPath):  # 如果不存在就创建文件夹
        os.mkdir(outPath)
    if vc.isOpened():
        res = True
    else:
        res = False
    image_count = 0  # 图片计数
    frame_count = 0  # 帧数计数
    face_details = []  # 辅助排除静态人脸
    while res:
        if mode:
            now_time = time.time()
            if now_time - begin_time > timeout:
                break
        res, frame = vc.read()  # 分帧读取视频
        # cv2.imshow("img", frame)
        if not res:
            break
        frame_count += 1
        if frame_count % timeF == 0:
            if multiThread:
                dealImgThread(frame, outPath, detector, predictor, timeF, frame_count).start()
            else:
                dots = detector(frame, 1)
                backup = face_details[:]
                face_details.clear()
                for k, d in enumerate(dots):
                    shape = predictor(frame, d)

                    # 排除静态人脸,如画像
                    isSame = False
                    for i in range(len(backup)):
                        same_count = 0
                        for p_pt, n_pt in zip(backup[i].parts(), shape.parts()):
                            if p_pt.x == n_pt.x and p_pt.y == n_pt.y:
                                same_count += 1
                        if same_count >= 10:
                            isSame = True
                            break
                    face_details.append(shape)

                    if not isSame and frame_count != timeF:
                        height, width = frame.shape[:2]

                        top_cross = max(0-d.top()-d.height()/2, 0)
                        bottom_cross = max(d.bottom()+d.height()/2-height, 0)
                        top = max(d.top()-d.height()/2-bottom_cross, 0)
                        bottom = min(d.bottom()+d.height()/2+top_cross, height)
                        # print(top_cross, bottom_cross, top, bottom)

                        left_cross = max(0-d.left()-d.width()/2, 0)
                        right_cross = max(d.right()+d.width()/2-width, 0)
                        left = max(d.left()-d.width()/2-right_cross, 0)
                        right = min(d.right()+d.width()/2+left_cross, width)
                        # print(left_cross, right_cross, left, right)

                        save_img = frame[int(top):int(bottom), int(left):int(right)]
                        # cv2.imshow("save", save_img)
                        if len(detector(save_img, 1)) == 1:  # 再次识别,提高精度
                            try:
                                save_img = cv2.resize(save_img, (299, 299))
                                image_count += 1
                                cv2.imwrite(outPath + "/" + str(image_count) + '.png', save_img)
                            except Exception:
                                pass
        if mode and image_count >= total:
            break
        if multiThread and mode and dealImgThread.image_count >= total:
            break
        # if cv2.waitKey(33) == 27:
        #     break
    vc.release()


def dealLocalVideos(inPath, outPath, timeF=7, multiThread=False):
    # 预处理
    if not os.path.exists(outPath):
        os.mkdir(outPath)

    # 加载模型
    detector, predictor = loadModel()

    # 视频处理
    videos = sorted(os.listdir(inPath))
    for video in videos[:]:
        video = os.path.join(inPath, video)
        img_dir = os.path.join(outPath, video.split('\\')[-1].split('.')[0])
        vc = cv2.VideoCapture(video)
        recognition(vc, img_dir, detector, predictor, timeF, LOCAL_VIDEO, multiThread)


def dealCameraStream(outPath, timeF=7, deviceId=0, timeout=60, total=100, multiThread=False):
    # 预处理
    if not os.path.exists(outPath):
        os.mkdir(outPath)

    # 加载模型
    detector, predictor = loadModel()

    # 视频处理
    now = datetime.datetime.now()
    img_dir = os.path.join(outPath, now.strftime("%Y-%m-%d_%H-%M-%S"))
    vc = cv2.VideoCapture(deviceId, cv2.CAP_DSHOW)
    recognition(vc, img_dir, detector, predictor, timeF, CAMERA_STREAM, timeout, total, multiThread)


if __name__ == '__main__':
    in_dir = r'F:\dataset\Celeb-DF-v2\Celeb-synthesis   '
    out_dir = r'F:\dataset\Celeb-DF-v2\celeb_syn_dlib'  # 处理后图片存放位置
    dealLocalVideos(in_dir, out_dir)
    # dealCameraStream(out_dir, deviceId=0, timeout=30)

from mtcnn.mtcnn import MTCNN
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from keras_vggface.utils import decode_predictions
import PIL
import numpy as np
import cv2
import json
from scipy.spatial.distance import cosine

detector = MTCNN()

model_extractor = VGGFace(model='resnet50',
                          include_top=False,
                          input_shape=(224, 224, 3),
                          pooling='avg')

border_rel = 0
base_imgs_scores = []
video_result = []
base_imgs = []
blur_boxes = []
check_blur = []
thres_cosine = 0.38
fps = 25
overall_frames = 0
overall_time = 0


def json_creator(start_frame, end_frame, box):
    result_temp = dict()
    time_start = start_frame / overall_frames * overall_time
    time_end = end_frame / overall_frames * overall_time
    result_temp["time_start"] = round(time_start, 5)
    result_temp["time_end"] = round(time_end, 5)
    result_temp["corner_1"] = list((box[0], box[1]))
    result_temp["corner_2"] = list((box[0] + box[2], box[1] + box[3]))

    video_result.append(result_temp)


def get_model_scores(faces):
    samples = np.asarray(faces, 'float32')
    samples = preprocess_input(samples, version=2)
    return model_extractor.predict(samples)


def compare_face(model_scores_img1, model_scores_img2):
    for idx, face_score_1 in enumerate(model_scores_img1):
        for idy, face_score_2 in enumerate(model_scores_img2):
            score = cosine(face_score_1, face_score_2)
            if score <= thres_cosine:
                return True
            else:
                continue
    return False


def populate_base_imgs(img):
    detections = detector.detect_faces(img)

    for i in range(len(detections)):
        x1, y1, width, height = detections[i]['box']
        dw = round(width * border_rel)
        dh = round(height * border_rel)
        x2, y2 = x1 + width + dw, y1 + height + dh

        face = img[y1:y2, x1:x2]
        face = PIL.Image.fromarray(face)
        face = face.resize((224, 224))
        face = np.asarray(face)
        face = face.astype('float32')
        face = np.expand_dims(face, axis=0)
        base_imgs.append(face)


def predict_face(img):
    detections = detector.detect_faces(img)
    for i in range(len(detections)):
        x1, y1, width, height = detections[i]['box']
        dw = round(width * border_rel)
        dh = round(height * border_rel)
        x2, y2 = x1 + width + dw, y1 + height + dh

        face = img[y1:y2, x1:x2]
        face = PIL.Image.fromarray(face)
        face = face.resize((224, 224))
        face = np.asarray(face)
        face = face.astype('float32')
        face = np.expand_dims(face, axis=0)

        face = preprocess_input(face, version=2)

        prediction = model_extractor.predict(face)
        results = decode_predictions(prediction)

        for result in results[0]:
            if result[1] * 100 > 70:
                ROI = img[y1 - 40:y1 + height + 40, x1 - 35:x1 + width + 35]
                blur = cv2.GaussianBlur(ROI, (101, 101), 0)
                img[y1 - 40:y1 + height + 40, x1 - 35:x1 + width + 35] = blur

    return img


def predict_face_comparator(img, frame_num):
    detections = detector.detect_faces(img)

    if frame_num % 8 == 0:
        step_boxes = []
        for i in range(len(detections)):
            step_boxes.append(detections[i]['box'][0] + detections[i]['box'][1])

        for blur_box in blur_boxes:
            s = blur_box[0][0] + blur_box[0][1]
            flag = False
            for value in step_boxes:
                if abs(s - value) < 10:
                    flag = True
            if not flag:
                end_time = frame_num
                json_creator(blur_box[1], end_time, blur_box[0])
                blur_boxes.remove(blur_box)
                check_blur.remove(s)
                print('deleted')

    for i in range(len(detections)):

        x1, y1, width, height = detections[i]['box']
        dw = round(width * border_rel)
        dh = round(height * border_rel)
        x2, y2 = x1 + width + dw, y1 + height + dh

        face = img[y1:y2, x1:x2]
        face = PIL.Image.fromarray(face)
        face = face.resize((224, 224))
        face = np.asarray(face)
        face = face.astype('float32')
        face = np.expand_dims(face, axis=0)

        frame_scores = get_model_scores(face)

        if not compare_face(frame_scores, base_imgs_scores):
            if frame_num % 3 == 0:
                s = detections[i]['box'][0] + detections[i]['box'][1]
                flag = False
                for value in check_blur:
                    # print("new blur:", detections[i]['box'][0:2])
                    if abs(s - value) < 10:
                        flag = True
                        break
                if not flag:
                    blur_boxes.append((detections[i]['box'], frame_num))
                    check_blur.append(detections[i]['box'][0] + detections[i]['box'][1])
                ROI = img[y1:y1 + height, x1:x1 + width]
                blur = cv2.GaussianBlur(ROI, (51, 51), 0)
                img[y1:y1 + height, x1:x1 + width] = blur

    return img


def get_main_faces(source):
    cap_first = cv2.VideoCapture(source)
    for i in range(0, 201, 5):
        ret, frame = cap_first.read()
        if ret == True:
            populate_base_imgs(frame)
        else:
            break
    cap_first.release()

    for img in base_imgs:
        base_imgs_scores.append(get_model_scores(img))


def create_video(source, prefix):
    counter = 0
    frame_no = 1

    get_main_faces(source)

    cap = cv2.VideoCapture(source)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    width = int(cap.get(3))
    height = int(cap.get(4))
    size = (width, height)
    out = cv2.VideoWriter(f'{prefix}_result.mp4', fourcc, 25.0, size)

    overall_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    overall_time = overall_frames / fps

    while cap.isOpened():
        ret, frame = cap.read()
        if ret == True:
            if counter % 2 == 0 and counter <= 200:
                img = predict_face_comparator(frame, frame_no)
                frame_no += 1
                out.write(img)

            elif counter > 200:
                img = predict_face_comparator(frame, frame_no)
                frame_no += 1
                out.write(img)

            else:
                out.write(frame)
            counter += 1
        else:
            break


    cap.release()
    out.release()

    with open(f'{prefix}_video.json', 'w') as f:
        f.write(json.dumps(video_result))
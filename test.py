from sklearn import svm

import joblib
import sys
sys.path.append('../')
import argparse
import numpy as np
import pickle
import time

from utils import evaluate

#from model.CAE import CAE
#from model.CAE import CAE_encoder


from utils import util
import os
import tensorflow as tf
#import inference
tf.autograph.set_verbosity(3)

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


from model import CAE

import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils


prefix = "D:\\Paper\\"

feature_list=[]
def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-k', '--K', type=int, default='2', help='K value')
    parser.add_argument('-g', '--gpu', type=str, default='0', help='Use which gpu?')
    parser.add_argument('-d', '--dataset', type=str, help='Train on which dataset')
    parser.add_argument('-b','--bn',type=bool,default=True,help='whether to use BN layer')
    parser.add_argument('--model_path',type=str,help='Path to saved tensorflow CAE model')
    parser.add_argument('--graph_path',type=str,help='Path to saved object detection frozen graph model')
    parser.add_argument('--svm_model',type=str,help='Path to saved svm model')
    parser.add_argument('--dataset_folder',type=str,help='Dataset Fodlder Path')
    parser.add_argument('-c','--class_add',type=bool,default=True,help='Whether to add class one-hot embedding to the featrue')
    parser.add_argument('-n','--norm',type=int,default=0,help='Whether to use Normalization to the Feature and the normalization level')
    parser.add_argument('--test_CAE',type=bool,default=False,help='Whether to test CAE')
    parser.add_argument('--matlab',type=bool,default=False,help='Whether to use matlab weights and biases to test')
    args = parser.parse_args()
    return args

batch_size=64
learning_rate=[1e-3,1e-4]
lr_decay_epochs=[100]
epochs=300



#PACK_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
CKPT_DIR = "D:\\Paper\\"+'\\Checkpoint'

PATH_TO_MODEL_DIR = "D:\Paper\efficientdet_d1_coco17_tpu-32"

PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "\saved_model"

LABEL_FILENAME = 'mscoco_label_map.pbtxt'

PATH_TO_LABELS="D:\\Paper\\data\\"+LABEL_FILENAME

####Image Paths####
base_url = "D:\\Paper\\dataset\\avenue\\training\\frames\\02\\"

filenames = '1421.jpg'

IMAGE_PATHS = [base_url+filenames]


####----------------------####

file_path = IMAGE_PATHS[0]
print(os.path.isfile(file_path))

detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                    use_display_name=True)

nn_1 = CAE.MemAE(height=64, width=64, channel=1, leaning_rate=learning_rate[0], ckpt_dir=CKPT_DIR)

def test(OVR_SVM_path, args,gap=2, score_threshold=0.4):
    





    temp_imp_list=[]

    anomaly_scores_records = []

    nn_1.load_params()

    image_folder = prefix +"dataset\\"+ args.dataset + '\\testing\\frames\\'

    vids_paths = util.get_vids_paths(image_folder)

    (image_height, image_width) = util.image_size_map[args.dataset]

    for frame_paths in vids_paths:
        anomaly_scores = np.empty(shape=(len(frame_paths),), dtype=np.float32)
        
        for frame_iter in range(gap, len(frame_paths) - gap):
            img = util.data_preprocessing(frame_paths[frame_iter],target_size=640)
            # Things to try:
            # Flip horizontally
            # image_np = np.fliplr(image_np).copy()

            # Convert image to grayscale
            # image_np = np.tile(
            #     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

            # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
            input_tensor = tf.convert_to_tensor(img)
            # The model expects a batch of images, so add an axis with `tf.newaxis`.
            input_tensor = input_tensor[tf.newaxis, ...]

            # input_tensor = np.expand_dims(image_np, 0)
            detections = detect_fn(input_tensor)

            # All outputs are batches tensors.
            # Convert to numpy arrays, and take index [0] to remove the batch dimension.
            # We're only interested in the first num_detections.


            num_detections = int(detections.pop('num_detections'))
            detections = {key: value[0, :num_detections].numpy()
                        for key, value in detections.items()}
            detections['num_detections'] = num_detections

            # detection_classes should be ints.
            detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

            image_np_with_detections = img.copy()

            viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes'],
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=200,
                min_score_thresh=.30,
                agnostic_mode=False)

            #plt.figure()
            #plt.imshow(image_np_with_detections)
            #result = Image.fromarray(image_np_with_detections)
            #result.save("det.jpg")
            
            #print('Done')

            #plt.show()

            _temp_anomaly_scores = []
            _temp_anomaly_score = 10000.
        
            for score,box in zip(detections['detection_scores'],detections['detection_boxes']):
            #print(score)
                if score>=score_threshold:
                    print(score)
                    box = [int(box[0] * image_height), int(box[1] * image_height), int(box[2] * image_height),
                            int(box[3] * image_width)]
                    img_gray = util.box_image_crop(frame_paths[frame_iter], box)
                    img_former = util.box_image_crop(frame_paths[frame_iter - gap], box)
                    img_back = util.box_image_crop(frame_paths[frame_iter + gap], box)

                    feed_dict={'former_batch': np.expand_dims(img_former, 0),
                                'gray_batch': np.expand_dims(img_gray, 0),
                                'back_batch': np.expand_dims(img_back, 0)}
                    
                    former_batch=tf.Variable(feed_dict['former_batch'])
                    gray_batch=tf.Variable(feed_dict['gray_batch'])
                    back_batch=tf.Variable(feed_dict['back_batch'])

                    grad1_x, grad1_y = tf.image.image_gradients(former_batch)
                    grad1= tf.concat([grad1_x,grad1_y],axis=-1)

                    grad3_x, grad3_y = tf.image.image_gradients(back_batch)
                    grad3= tf.concat([grad3_x,grad3_y],axis=-1)

                    former_feat=nn_1.half_step(grad1,frame_iter)
                    gray_feat=nn_1.half_step(gray_batch,frame_iter)
                    back_feat=nn_1.half_step(grad3,frame_iter)
                    # [batch_size,3072]

                    feat=tf.concat([tf.keras.backend.flatten(former_feat),tf.keras.backend.flatten(gray_feat),tf.keras.backend.flatten(back_feat)],axis=0)
                    feat=tf.reshape(feat,[feat.shape[0],1])
                    clf = joblib.load(OVR_SVM_path)
                    timestamp = time.time()
                    num_videos = len(vids_paths)
                    total = 0

                    feature_list.append(feat)

                    scores = clf.decision_function(feat)

                    _temp_anomaly_scores.append(-max(scores[0]))
            if _temp_anomaly_scores.__len__() != 0:
                 _temp_anomaly_score = max(_temp_anomaly_scores)

            print('video = {} / {}, i = {} / {}, score = {:.6f}'.format(
                frame_paths[0].split('\\')[-2], num_videos, frame_iter, len(frame_paths), _temp_anomaly_score))
            temp_imp_list.append([frame_paths[0].split('\\')[-2], num_videos, frame_iter, len(frame_paths), _temp_anomaly_score])
            anomaly_scores[frame_iter] = _temp_anomaly_score

        anomaly_scores[:gap] = anomaly_scores[gap]
        anomaly_scores[-gap:] = anomaly_scores[-gap-1]

        min_score=min(anomaly_scores)
        for i,_s in enumerate(anomaly_scores):
            if _s==10000.:
                anomaly_scores[i]=min_score
        anomaly_scores_records.append(anomaly_scores)
        total += len(frame_paths)

    # use the evaluation functions from github.com/StevenLiuWen/ano_pred_cvpr2018
    result_dict = {'dataset': args.dataset, 'psnr': anomaly_scores_records, 'flow': [], 'names': [],
                   'diff_mask': []}
    used_time = time.time() - timestamp

    print('total time = {}, fps = {}'.format(used_time, total / used_time))
    np.save("D:\\Paper\\veryimportantscores.npy",temp_imp_list)
    # TODO specify what's the actual name of ckpt.
    if not args.bn:
        pickle_path = "D:\\Paper\\Score\\" + args.dataset + str(args.K) + '.pkl'
    else:
        pickle_path = "D:\\Paper\\Score\\" + args.dataset +'_bn'+ '.pkl'

    with open(pickle_path, 'wb') as writer:
        pickle.dump(result_dict, writer, pickle.HIGHEST_PROTOCOL)
    np.save("feature_list.npy",feature_list)
    results = evaluate.evaluate_all( pickle_path,reverse=True,smoothing=True)
    print(results)

def evaluator(pickle_path,reverse=True,smoothing=True):
    
    results = evaluate.evaluate_all( pickle_path,reverse=True,smoothing=True)
    print(results)



def test_CAE(CAE_model_path,args,gap=2, score_threshold=0.4):
    temp_imp_list=[]
    image_folder = prefix +"conda prog\\VAD\\object_centric_VAD-master\\scripts\\Data\\"+ args.dataset + '\\testing\\frames\\'
    vids_paths = util.get_vids_paths(image_folder)
    # to set gpu visible
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    
    # to load the ssd fpn model, and get related tensor
    #object_detection_graph = inference.load_frozen_graph(args.graph_path)
    with object_detection_graph.as_default():
        ops = object_detection_graph.get_operations()
        all_tensor_names = {output.name for op in ops for output in op.outputs}
        tensor_dict = {}
        for key in [
            'num_detections', 'detection_boxes', 'detection_scores',
            'detection_classes'
        ]:
            tensor_name = key + ':0'
            if tensor_name in all_tensor_names:
                tensor_dict[key] = object_detection_graph.get_tensor_by_name(tensor_name)

        image_tensor = object_detection_graph.get_tensor_by_name('image_tensor:0')

        former_batch = tf.placeholder(dtype=tf.float32, shape=[1, 64, 64, 1], name='former_batch')
        gray_batch = tf.placeholder(dtype=tf.float32, shape=[1, 64, 64, 1], name='gray_batch')
        back_batch = tf.placeholder(dtype=tf.float32, shape=[1, 64, 64, 1], name='back_batch')

        grad1_x, grad1_y = tf.image.image_gradients(former_batch)
        grad1 = tf.concat([grad1_x, grad1_y], axis=-1)
        # grad2_x,grad2_y=tf.image.image_gradients(gray_batch)
        grad3_x, grad3_y = tf.image.image_gradients(back_batch)
        grad3 = tf.concat([grad3_x, grad3_y], axis=-1)

        grad_dis_1 = tf.sqrt(tf.square(grad1_x) + tf.square(grad1_y))
        grad_dis_2 = tf.sqrt(tf.square(grad3_x) + tf.square(grad3_y))
        print(grad1, gray_batch, grad3)
        former_output = CAE(grad1, 'former', bn=args.bn, training=False)
        gray_output = CAE(gray_batch, 'gray', bn=args.bn, training=False)
        back_output = CAE(grad3, 'back', bn=args.bn, training=False)
        
        outputs=tf.concat([tf.layers.flatten(former_output),tf.layers.flatten(gray_output),tf.layers.flatten(back_output)],axis=1)
        targets=tf.concat([tf.layers.flatten(grad1),tf.layers.flatten(gray_batch),tf.layers.flatten(grad3)],axis=1)
        L2_dis=tf.reduce_sum(tf.square(outputs-targets))

        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='former_encoder')
        var_list.extend(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='gray_encoder'))
        var_list.extend(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='back_encoder'))
        var_list.extend(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='former_decoder'))
        var_list.extend(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='gray_decoder'))
        var_list.extend(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='back_decoder'))

        if args.bn:
            g_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='former_encoder')
            g_list.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='gray_encoder'))
            g_list.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='back_encoder'))
            g_list.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='former_decoder'))
            g_list.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='gray_decoder'))
            g_list.extend(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='back_decoder'))
            bn_list = [g for g in g_list if 'moving_mean' in g.name or 'moving_variance' in g.name]
            var_list += bn_list
        restorer = tf.train.Saver(var_list=var_list)

        (image_height, image_width) = util.image_size_map[args.dataset]
        # image_height,image_width=640,640
        anomaly_scores_records = []

        timestamp = time.time()
        num_videos = len(vids_paths)
        total = 0

        with tf.Session() as sess:
            if args.bn:
                restorer.restore(sess, CAE_model_path + '_bn')
            else:
                restorer.restore(sess, CAE_model_path)

            for frame_paths in vids_paths:
                anomaly_scores = np.empty(shape=(len(frame_paths),), dtype=np.float32)

                for frame_iter in range(gap, len(frame_paths) - gap):
                    img = np.expand_dims(util.data_preprocessing(frame_paths[frame_iter], target_size=640), axis=0)
                    output_dict = sess.run(tensor_dict,
                                           feed_dict={image_tensor: img})

                    # all outputs are float32 numpy arrays, so convert types as appropriate
                    output_dict['num_detections'] = int(output_dict['num_detections'][0])
                    output_dict['detection_classes'] = output_dict[
                        'detection_classes'][0].astype(np.int8)
                    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
                    output_dict['detection_scores'] = output_dict['detection_scores'][0]

                    _temp_anomaly_scores = []
                    _temp_anomaly_score = 10000.

                    for score, box in zip(output_dict['detection_scores'], output_dict['detection_boxes']):
                        if score >= score_threshold:
                            box = [int(box[0] * image_height), int(box[1] * image_height), int(box[2] * image_height),
                                   int(box[3] * image_width)]
                            img_gray = util.box_image_crop(frame_paths[frame_iter], box)
                            img_former = util.box_image_crop(frame_paths[frame_iter - gap], box)
                            img_back = util.box_image_crop(frame_paths[frame_iter + gap], box)

                            l2_dis = sess.run(L2_dis, feed_dict={former_batch: np.expand_dims(img_former, 0),
                                                              gray_batch: np.expand_dims(img_gray, 0),
                                                              back_batch: np.expand_dims(img_back, 0)})

                            _temp_anomaly_scores.append(l2_dis)
                    if _temp_anomaly_scores.__len__() != 0:
                        _temp_anomaly_score = max(_temp_anomaly_scores)

                    print('video = {} / {}, i = {} / {}, score = {:.6f}'.format(
                        frame_paths[0].split('\\')[-2], num_videos, frame_iter, len(frame_paths),
                        _temp_anomaly_score))
                    temp_imp_list.append([frame_paths[0].split('\\')[-2], num_videos, frame_iter, len(frame_paths),
                        _temp_anomaly_score])

                    anomaly_scores[frame_iter] = _temp_anomaly_score

                anomaly_scores[:gap] = anomaly_scores[gap]
                anomaly_scores[-gap:] = anomaly_scores[-gap - 1]

                min_score=np.min(anomaly_scores)
                for i,_s in enumerate(anomaly_scores):
                    if _s==10000.:
                        anomaly_scores[i]=min_score
                anomaly_scores_records.append(anomaly_scores)
                total += len(frame_paths)

                # use the evaluation functions from github.com/StevenLiuWen/ano_pred_cvpr2018
            result_dict = {'dataset': args.dataset, 'psnr': anomaly_scores_records, 'flow': [], 'names': [],
                           'diff_mask': []}
            used_time = time.time() - timestamp

            print('total time = {}, fps = {}'.format(used_time, total / used_time))
            np.save("C:\\Users\\srini\\conda prog\\VAD\\object_centric_VAD-master\\scripts\\veryimportantscore.npy",temp_imp_list)
            # TODO specify what's the actual name of ckpt.
            if not args.bn:
                pickle_path = "C:\\Users\\srini\\conda prog\\VAD\\object_centric_VAD-master\\scripts\\Score\\" + args.dataset+'_CAE_only' + '.pkl'
            else:
                pickle_path = "C:\\Users\\srini\\conda prog\\VAD\\object_centric_VAD-master\\scripts\\Score\\" + args.dataset + '_CAE_only'+'_bn' + '.pkl'

            with open(pickle_path, 'wb') as writer:
                pickle.dump(result_dict, writer, pickle.HIGHEST_PROTOCOL)

            results = evaluate.evaluate_all(pickle_path, reverse=True, smoothing=True)
            print(results)

if __name__=='__main__':

    args=arg_parse()
    pickle_path = "D:\\Paper\\Score\\" + "avenue"+"10"+ '.pkl'
    #evaluator(pickle_path,True,True)
    if not args.test_CAE:
        test(args.svm_model,args,score_threshold=0.4)
    else:
        test_CAE(args.model_path,args)

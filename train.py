import numpy as np

import os, inspect, time, math
os.environ['CUDA_VISIBLE_DEVICES'] = "0"


import sys
sys.path.append('../')
import argparse
import tensorflow as tf
tf.autograph.set_verbosity(3)

physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


import random
from utils import util
from PIL import Image

from sklearn.cluster import KMeans
import joblib
from model import CAE
#from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsRestClassifier



summary_save_path_pre="D:\\Paper\\CAE\\CAE_" #'/home/jiachang/summary/CAE_'
svm_save_dir="D:\\Paper\\SVM\\"#'/home/jiachang/clfs/'
prefix = "D:\\Paper\\dataset"


model_save_path_pre='CAE_'


batch_size=64
learning_rate=[1e-3,1e-4]
lr_decay_epochs=[100]
epochs=300

#PACK_PATH = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
CKPT_DIR = "D:\\Paper\\"+'\\Checkpoint'

def arg_parse():
    parser=argparse.ArgumentParser()
    parser.add_argument('-g','--gpu',type=str,default='0',help='Use which gpu?')
    parser.add_argument('-d','--dataset',type=str,help='Train on which dataset')
    parser.add_argument('-t','--train',type=str,help='Train on SVM / CAE')
    parser.add_argument('-b','--bn',type=bool,default=True,help='whether to use BN layer')
    parser.add_argument('--dataset_folder',type=str,help='Dataset Fodlder Path')
    parser.add_argument('--model_dir',type=str,help='Folder to save tensorflow CAE model')
    parser.add_argument('-c','--class_add',type=bool,default=True,help='Whether to add class one-hot embedding to the featrue')
    parser.add_argument('-n','--norm',type=int,default=0,help='Whether to use Normalization to the Feature and the normalization level')
    parser.add_argument('--box_imgs_npy_path',type=str,help='Path for npy file that store the \(box,img_path\)')
    parser.add_argument('--weight_reg',type=float,default=0,help='weight regularization for training CAE')
    parser.add_argument('--matlab',type=bool,default=False,help='Whether to use matlab to train SVMs')
    args=parser.parse_args()
    return args

def make_dir(path):

    try: os.mkdir(path)
    except: pass

def gray2rgb(gray):

    rgb = np.ones((gray.shape[0], gray.shape[1], 3)).astype(np.float32)
    rgb[:, :, 0] = gray[:, :, 0]
    rgb[:, :, 1] = gray[:, :, 0]
    rgb[:, :, 2] = gray[:, :, 0]

    return rgb

def dat2canvas(data):

    numd = math.ceil(np.sqrt(data.shape[0]))
    [dn, dh, dw, dc] = data.shape
    canvas = np.ones((dh*numd, dw*numd, dc)).astype(np.float32)

    for y in range(numd):
        for x in range(numd):
            try: tmp = data[x+(y*numd)]
            except: pass
            else: canvas[(y*dh):(y*dh)+28, (x*dw):(x*dw)+28, :] = tmp
    if(dc == 1):
        canvas = gray2rgb(gray=canvas)

    return canvas

def save_img(contents, names=["", "", ""], savename=""):

    num_cont = len(contents)
    plt.figure(figsize=(5*num_cont+2, 5))

    for i in range(num_cont):
        plt.subplot(1,num_cont,i+1)
        plt.title(names[i])
        plt.imshow(dat2canvas(data=contents[i]))

    plt.tight_layout()
    plt.savefig(savename)
    plt.close()

def discrete_cmap(N, base_cmap=None):

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)

    return base.from_list(cmap_name, color_list, N)

def latent_plot(latent, y, n, savename=""):

    plt.figure(figsize=(6, 5))
    plt.scatter(latent[:, 0], latent[:, 1], c=y, \
        marker='o', edgecolor='none', cmap=discrete_cmap(n, 'jet'))
    plt.colorbar(ticks=range(n))
    plt.grid()
    plt.tight_layout()
    plt.savefig(savename)
    plt.close()

def boxplot(contents, savename=""):

    data, label = [], []
    for cidx, content in enumerate(contents):
        data.append(content)
        label.append("class-%d" %(cidx))

    plt.clf()
    fig, ax1 = plt.subplots()
    bp = ax1.boxplot(data, showfliers=True, whis=3)
    ax1.set_xticklabels(label, rotation=45)

    plt.tight_layout()
    plt.savefig(savename)
    plt.close()

def histogram(contents, savename=""):

    n1, _, _ = plt.hist(contents[0], bins=100, alpha=0.5, label='Normal')
    n2, _, _ = plt.hist(contents[1], bins=100, alpha=0.5, label='Abnormal')
    h_inter = np.sum(np.minimum(n1, n2)) / np.sum(n1)
    plt.xlabel("MSE")
    plt.ylabel("Number of Data")
    xmax = max(contents[0].max(), contents[1].max())
    plt.xlim(0, xmax)
    plt.text(x=xmax*0.01, y=max(n1.max(), n2.max()), s="Histogram Intersection: %.3f" %(h_inter))
    plt.legend(loc='upper right')
    plt.savefig(savename)
    plt.close()



nn_1 = CAE.MemAE(height=64, width=64, channel=1, leaning_rate=learning_rate[0], ckpt_dir=CKPT_DIR)

def train_CAE(path_boxes_np,args):
    
    epoch_len=len(np.load(path_boxes_np))


    f_imgs,g_imgs,b_imgs,class_indexes=util.CAE_dataset_feed_dict(prefix,path_boxes_np,dataset_name=args.dataset)

    print("Images have been extracted.....")



    #nn_2 = CAE.MemAE(height=64, width=64, channel=1, leaning_rate=learning_rate[0], ckpt_dir=CKPT_DIR)
    #nn_3 = CAE.MemAE(height=64, width=64, channel=1, leaning_rate=learning_rate[0], ckpt_dir=CKPT_DIR)

    print("MemAE initialised.....")

    indices=list(range(epoch_len))



    make_dir(path="results")
    result_list = ["tr_resotring"]
    for result_name in result_list: make_dir(path=os.path.join("results", result_name))

    #print(np.load(path_boxes_np))
    start_time = time.time()

    for epoch in range(epochs):
        random.shuffle(indices)
        for i in range(epoch_len//batch_size):

            feed_dict={'former_batch':[f_imgs[d] for d in indices[i*batch_size:(i+1)*batch_size]],
                       'gray_batch':[g_imgs[d] for d in indices[i*batch_size:(i+1)*batch_size]],
                       'back_batch':[b_imgs[d] for d in indices[i*batch_size:(i+1)*batch_size]]
                      }

            

            former_batch=tf.Variable(feed_dict['former_batch'])
            gray_batch=tf.Variable(feed_dict['gray_batch'])
            back_batch=tf.Variable(feed_dict['back_batch'])

            grad1_x, grad1_y = tf.image.image_gradients(former_batch)
            grad1= tf.concat([grad1_x,grad1_y],axis=-1)

            grad3_x, grad3_y = tf.image.image_gradients(back_batch)
            grad3= tf.concat([grad3_x,grad3_y],axis=-1)

            print("\nTraining to %d epochs (%d of minibatch size)" %(epoch, batch_size))

            print(grad1.shape)
            former_output, f_mse, f_mem_etrp, f_loss=nn_1.step(grad1,i,True)
            gray_output, g_mse, g_mem_etrp, g_loss=nn_1.step(gray_batch,i,True)
            back_output, b_mse, b_mem_etrp, b_loss=nn_1.step(grad3,i,True)

            nn_1.save_params()
            print("Epoch [%d / %d] (%d iteration)  F_MSE:%.3f, F_W-ETRP:%.3f, F_Total:%.3f, G_MSE:%.3f, G_W-ETRP:%.3f, G_Total:%.3f, B_MSE:%.3f, B_W-ETRP:%.3f, B_Total:%.3f"\
                %(epoch, epochs, i, np.sum(f_mse), np.sum(f_mem_etrp), f_loss, np.sum(g_mse), np.sum(g_mem_etrp), g_loss, np.sum(b_mse), np.sum(b_mem_etrp), b_loss))


    
    print('Done')
           
    print("train")

def extract_features(path_boxes_np,args):

    f_imgs,g_imgs,b_imgs,class_indexes=util.CAE_dataset_feed_dict(prefix,path_boxes_np,args.dataset)

    print('dataset loaded!')

    iters=np.load(path_boxes_np).__len__()

    print("Images have been extracted.....")

    nn_1.load_params()
    #nn_2 = CAE.MemAE(height=64, width=64, channel=1, leaning_rate=learning_rate[0], ckpt_dir=CKPT_DIR)
    #nn_3 = CAE.MemAE(height=64, width=64, channel=1, leaning_rate=learning_rate[0], ckpt_dir=CKPT_DIR)

    print("MemAE initialised.....")

    for i in range(iters):
        feed_dict={'former_batch':np.expand_dims(f_imgs[i],0),
                    'gray_batch':np.expand_dims(g_imgs[i],0),
                   'back_batch':np.expand_dims(b_imgs[i],0)}

        former_batch=tf.Variable(feed_dict['former_batch'])
        gray_batch=tf.Variable(feed_dict['gray_batch'])
        back_batch=tf.Variable(feed_dict['back_batch'])

        grad1_x, grad1_y = tf.image.image_gradients(former_batch)
        grad1=tf.concat([grad1_x,grad1_y],axis=-1)

        grad3_x, grad3_y = tf.image.image_gradients(back_batch)
        grad3=tf.concat([grad3_x,grad3_y],axis=-1)

        former_feat=nn_1.half_step(grad1,i)
        gray_feat=nn_1.half_step(gray_batch,i)
        back_feat=nn_1.half_step(grad3,i)





    # [batch_size,3072]
        feat=tf.concat([tf.keras.backend.flatten(former_feat),tf.keras.backend.flatten(gray_feat),tf.keras.backend.flatten(back_feat)],axis=0)
        #print(feat.shape)


    return feat

def train_one_vs_rest_SVM(path_boxes_np,K,args):  
    data=extract_features(path_boxes_np,args)
    print(data)  
    data=tf.reshape(data,[data.shape[0],1])
    print(data.shape)
    print('feature extraction finish!')
    labels=KMeans(n_clusters=K,init='k-means++',n_init=10,algorithm='full',max_iter=300).fit(data)
    labels=labels.labels_
    print(labels)
    sparse_labels=np.eye(K)[labels]
    sparse_labels=(sparse_labels-0.5)*2
    print('clustering finished!')
    base_estimizer=SVC(kernel='rbf',max_iter=10000)
    print(base_estimizer)
    ovr_classifer=OneVsRestClassifier(base_estimizer)
    print(ovr_classifer)
    ovr_classifer.fit(data,sparse_labels)
    joblib.dump(ovr_classifer,svm_save_dir+args.dataset+str(K)+'.m')
    print('train finished!')

if __name__=='__main__':
    args=arg_parse()
    os.environ['CUDA_VISIBLE_DEVICES']=args.gpu
    K=10
    if args.train=='CAE':
        train_CAE(args.box_imgs_npy_path,args)
    
    else:
        print(args.model_dir)
        print("Training"+str(K)+" th svm .......")
        train_one_vs_rest_SVM(args.box_imgs_npy_path,K,args)

    


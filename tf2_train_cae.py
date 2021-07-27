import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers

def train_CAE(path_boxes_np,args):
    
    epoch_len=len(np.load(path_boxes_np))
    f_imgs,g_imgs,b_imgs,class_indexes=util.CAE_dataset_feed_dict(prefix,path_boxes_np,dataset_name=args.dataset)

    #former_batch,gray_batch,back_batch=util.CAE_dataset(path_boxes_np,args.dataset,epochs,batch_size)

    former_batch=tf.keras.Input(dtype=tf.float32,shape=[batch_size,64,64,1],name='former_batch')#not sure if there's a argument called name otherwise chill
    gray_batch=tf.keras.Input(dtype=tf.float32,shape=[batch_size,64,64,1],name='gray_batch')
    back_batch=tf.keras.Input(dtype=tf.float32,shape=[batch_size,64,64,1],name='back_batch')

    grad1_x, grad1_y = tf.image.image_gradients(former_batch)
    grad1=tf.concat([grad1_x,grad1_y],axis=-1)

    # grad2_x,grad2_y=tf.image.image_gradients(gray_batch)

    grad3_x, grad3_y = tf.image.image_gradients(back_batch)
    grad3=tf.concat([grad3_x,grad3_y],axis=-1)

    #grad_dis_1 = tf.sqrt(tf.square(grad1_x) + tf.square(grad1_y))
    #grad_dis_2 = tf.sqrt(tf.square(grad3_x) + tf.square(grad3_y))

    former_outputs=CAE.CAE(grad1,'former',bn=args.bn,training=True)
    gray_outputs=CAE.CAE(gray_batch,'gray',bn=args.bn,training=True)
    back_outputs=CAE.CAE(grad3,'back',bn=args.bn,training=True)

    former_loss=CAE.pixel_wise_L2_loss(former_outputs,grad1)#this function doesn't exist in neuralnet.py
    gray_loss=CAE.pixel_wise_L2_loss(gray_outputs,gray_batch)
    back_loss=CAE.pixel_wise_L2_loss(back_outputs,grad3)

    global_step=tf.Variable(0,dtype=tf.int32,trainable=False)
    global_step_a=tf.Variable(0,dtype=tf.int32,trainable=False)
    global_step_b=tf.Variable(0,dtype=tf.int32,trainable=False)

    lr_decay_epochs[0] =int(epoch_len//batch_size)*lr_decay_epochs[0]

    lr=tf.train.piecewise_constant(global_step,boundaries=lr_decay_epochs,values=learning_rate)
    #I think we have to hardcode it but do check this out https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/schedules/ExponentialDecay

    former_vars=tf.Graph.get_collection(name=tf.GraphKeys.TRAINABLE_VARIABLES,scope='former_')
    gray_vars=tf.Graph.get_collection(name=tf.GraphKeys.TRAINABLE_VARIABLES,scope='gray_')
    back_vars=tf.Graph.get_collection(name=tf.GraphKeys.TRAINABLE_VARIABLES,scope='back_')
    # print(former_vars)
    if args.weight_reg!=0:
        former_loss=former_loss+args.weight_reg*weiht_regualized_loss(former_vars)
        gray_loss=gray_loss+args.weight_reg*weiht_regualized_loss(gray_vars)
        back_loss=back_loss+args.weight_reg*weiht_regualized_loss(back_vars)
        
    former_op=tf.keras.optimizers.Adam(learning_rate=lr).minimize(former_loss,var_list=former_vars,global_step=global_step)
    gray_op=tf.keras.optimizers.Adam(learning_rate=lr).minimize(gray_loss,var_list=gray_vars,global_step=global_step_a)
    back_op=tf.keras.optimizers.Adam(learning_rate=lr).minimize(back_loss,var_list=back_vars,global_step=global_step_b)

    step=0
    if not args.bn:
        writer=tf.summary.create_file_writer(logdir=summary_save_path_pre+args.dataset)
    else:
        writer=tf.summary.create_file_writer(logdir=summary_save_path_pre+args.dataset+'_bn')

    tf.summary.scalar('loss/former_loss',former_loss)
    tf.summary.scalar('loss/gray_loss',gray_loss)
    tf.summary.scalar('loss/back_loss',back_loss)
    #tf.summary.image('inputs/former',grad_dis_1)
    tf.summary.image('inputs/gray',gray_batch)
    #tf.summary.image('inputs/back',grad_dis_2)
    #tf.summary.image('outputs/former',former_outputs)
    tf.summary.image('outputs/gray',gray_outputs)
    #tf.summary.image('outputs/back',back_outputs)
    summary_op=tf.summary.merge_all()# there is nothing called .merge_all() but apparently it does join on its own or something but check this https://www.tensorflow.org/tensorboard/migrate

    saver=tf.train.Saver(var_list=tf.global_variables())#adei this saver thingy there's a lot to be changed nu nenaikaran. i'll look into it after i wake up. ippo thembu illa https://www.tensorflow.org/api_docs/python/tf/train/Checkpoint#save
    indices=list(range(epoch_len))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(epochs):
            print("Epoch : ",epoch)
            random.shuffle(indices)
            for i in range(epoch_len//batch_size):
                print(" i : ",i)
                feed_dict={former_batch:[f_imgs[d] for d in indices[i*batch_size:(i+1)*batch_size]],
                           gray_batch:[g_imgs[d] for d in indices[i*batch_size:(i+1)*batch_size]],
                           back_batch:[b_imgs[d] for d in indices[i*batch_size:(i+1)*batch_size]]
                           }
                step,_lr,_,_,_,_former_loss,_gray_loss,_back_loss=sess.run([global_step,lr,former_op,gray_op,back_op,former_loss,gray_loss,back_loss],feed_dict=feed_dict)
                if step%10==0:
                    print('At step {}'.format(step))
                    print('\tLearning Rate {:.4f}'.format(_lr))
                    print('\tFormer Loss {:.4f}'.format(_former_loss))
                    print('\tGray Loss {:.4f}'.format(_gray_loss))
                    print('\tBack Loss {:.4f}'.format(_back_loss))
                    
                if step%50==0:
                    _summary=sess.run(summary_op,feed_dict=feed_dict)
                    plt.plot(_summary)
                    writer.add_summary(_summary,global_step=step)
                if step%10000==0:
                    if not args.bn:
                        saver.save(sess,model_save_path_pre+args.dataset)
                    else:
                        saver.save(sess,model_save_path_pre+args.dataset+'_bn')
        if not args.bn:
            saver.save(sess,model_save_path_pre+args.dataset)
        else:
            saver.save(sess,model_save_path_pre+args.dataset+'_bn')

        print('train finished!')
        sess.close()
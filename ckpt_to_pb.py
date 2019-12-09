#coding:utf-8
###################################################
# File Name: test.py
# Author: Meng Zhao
# mail: @
# Created Time: 2019年03月11日 星期一 15时16分58秒
#=============================================================
import os
import sys
import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.saved_model import tag_constants


sys.path.append('../')

from setting import *



os.environ["CUDA_VISIBLE_DEVICES"] = "" #不使用GPU



def get_graph_def_from_saved_model(saved_model_dir): 
    with tf.Session() as session:
        meta_graph_def = tf.saved_model.loader.load(
        session,
        tags=[tag_constants.SERVING],
        export_dir=saved_model_dir
        ) 
    return meta_graph_def.graph_def


def get_size(model_dir, model_file='saved_model.pb'):
    model_file_path = os.path.join(model_dir, model_file)
    print(model_file_path, '')
    pb_size = os.path.getsize(model_file_path)
    variables_size = 0
    if os.path.exists(
        os.path.join(model_dir,'variables/variables.data-00000-of-00001')):
        variables_size = os.path.getsize(os.path.join(
            model_dir,'variables/variables.data-00000-of-00001'))
        variables_size += os.path.getsize(os.path.join(
            model_dir,'variables/variables.index'))
    print('Model size: {} KB'.format(round(pb_size/(1024.0),3)))
    print('Variables size: {} KB'.format(round( variables_size/(1024.0),3)))
    print('Total Size: {} KB'.format(round((pb_size + variables_size)/(1024.0),3)))

def freeze_graph(input_checkpoint_dir, output_graph):
    '''
    :param input_checkpoint_dir:
    :param output_graph: PB模型保存路径
    :return:
    '''
 
    # 指定输出的节点名称,该节点名称必须是原模型中存在的节点
    # 直接用最后输出的节点，可以在tensorboard中查找到，tensorboard只能在linux中使用
    output_node_names = ['input_ids', 'input_mask', 'segment_ids', 'is_training', 'crf_pred_labels', 'crf_probs', 'logits']
    cp_file = tf.train.latest_checkpoint(input_checkpoint_dir)

    graph = tf.Graph()

    with graph.as_default():
        saver = tf.train.import_meta_graph('{}.meta'.format(cp_file))
        input_graph_def = graph.as_graph_def()  # 返回一个序列化的图代表当前的图
        with tf.Session() as sess:
            saver.restore(sess, cp_file) #恢复图并得到数据
            output_graph_def = graph_util.convert_variables_to_constants(  # 模型持久化，将变量值固定
                sess=sess,
                input_graph_def=input_graph_def,# 等于:sess.graph_def
                output_node_names=output_node_names)# 如果有多个输出节点，以逗号隔开
 
            with tf.gfile.GFile(output_graph, "wb") as f: #保存模型
                f.write(output_graph_def.SerializeToString()) #序列化输出
            print("%d ops in the final graph." % len(output_graph_def.node)) #得到当前图有几个操作节点



def tf_serving_model(pb_path, tf_serving_model_path):
    restore_graph_def = tf.GraphDef()
    restore_graph_def.ParseFromString(open(pb_path, 'rb').read())
    tf.import_graph_def(restore_graph_def, name='')

    graph = tf.get_default_graph()

    if tf.gfile.Exists(tf_serving_model_path):
        tf.gfile.DeleteRecursively(tf_serving_model_path)

    builder = tf.saved_model.builder.SavedModelBuilder(tf_serving_model_path)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #saver.restore(sess, cp_file) #恢复图并得到数据
        input_ids = graph.get_operation_by_name('input_ids').outputs[0]
        input_mask = graph.get_operation_by_name('input_mask').outputs[0]
        segment_ids = graph.get_operation_by_name('segment_ids').outputs[0]
        is_training = graph.get_operation_by_name('is_training').outputs[0]

        probs =  graph.get_operation_by_name('crf_probs').outputs[0]
        pred_labels = graph.get_operation_by_name('crf_pred_labels').outputs[0]

        tensor_info_input_ids = tf.saved_model.utils.build_tensor_info(input_ids)
        tensor_info_input_mask = tf.saved_model.utils.build_tensor_info(input_mask)
        tensor_info_segment_ids = tf.saved_model.utils.build_tensor_info(segment_ids)
        tensor_info_is_training = tf.saved_model.utils.build_tensor_info(is_training)
        
        tensor_info_probs = tf.saved_model.utils.build_tensor_info(probs)
        tensor_info_pred_labels = tf.saved_model.utils.build_tensor_info(pred_labels)

        prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs={'input_ids': tensor_info_input_ids,
                        'input_mask': tensor_info_input_mask,
                        'segment_ids': tensor_info_segment_ids,
                        'is_training': tensor_info_is_training},
                outputs={'probs': tensor_info_probs,
                         'pred_labels': tensor_info_pred_labels},
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)) 

        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={
                'predict_text': prediction_signature,
            },
            main_op=tf.tables_initializer()) 


        builder.save()


if __name__ == '__main__':
    checkpoint_path = MODEL_DIR + '/checkpoints/'
    pb_path = MODEL_DIR + '/checkpoints/frozen_model.pb'
    tf_serving_model_version = '1'
    tf_serving_model_path = MODEL_DIR + '/checkpoints/' + tf_serving_model_version
 
    freeze_graph(checkpoint_path, pb_path)
    tf_serving_model(pb_path, tf_serving_model_path)


    #test = get_graph_def_from_saved_model('../example/runs/v0.91/checkpoints')
    #test = get_size(MODEL_DIR+'/checkpoints', 'frozen_model.pb')



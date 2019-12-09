#coding:utf-8
###################################################
# File Name: eval.py
# Author: Meng Zhao
# mail: @
# Created Time: Fri 23 Mar 2018 09:27:09 AM CST
#=============================================================
import os
import sys
import csv
import datetime
import logging
import numpy as np
import codecs


sys.path.append('../')

from preprocess import bert_data_utils
from preprocess import dataloader
from preprocess import tokenization
from preprocess import ner_utils
from setting import *
from tensorflow.contrib import learn


#os.environ["CUDA_VISIBLE_DEVICES"] = "" #不使用GPU



def cosine(q, a):
    pooled_len_1 = np.sqrt(np.sum(np.multiply(q, q), 1))
    pooled_len_2 = np.sqrt(np.sum(np.multiply(a, a), 1))

    pooled_mul_12 = np.sum(np.multiply(q, a), 1)
    score = np.divide(pooled_mul_12, np.multiply(pooled_len_1, pooled_len_2) + 1e-8)
    return score

def softmax(x):
    """Compute softmax values for each sets of scores in x."""

    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def sigmoid(x):
    output = 1.0 / (1.0 + 1.0 / np.exp(x))
    return output

def norm(x):
    x[0] = x[0] * 10
    return x / x.sum()


class Evaluator(object):
    def __init__(self, config):
        #self.top_k_num = config['top_k']
        self.model_dir = config['model_dir']
        self.max_seq_length = config['max_seq_length']
        self.vocab_file = config['vocab_file']
        self.label_map_file = config['label_map_file']
        self.model_checkpoints_dir = config['model_checkpoints_dir']
        self.model_pb_path = config['model_pb_path']


        #init label dict and processors
        label2idx, idx2label = bert_data_utils.read_ner_label_map_file(self.label_map_file)
        self.idx2label = idx2label
        self.label2idx = label2idx
        
        #self.label2code = bert_data_utils.read_code_file(self.code_file)

        self.tokenizer = tokenization.FullTokenizer(vocab_file=self.vocab_file, do_lower_case=True)

    
        #init stop set
        self.stop_set = dataloader.get_stopwords_set(STOPWORD_FILE)

        #use default graph
        self.graph = tf.get_default_graph()
        restore_graph_def = tf.GraphDef()
        restore_graph_def.ParseFromString(open(self.model_pb_path, 'rb').read())
        tf.import_graph_def(restore_graph_def, name='')

        session_conf = tf.ConfigProto()
        self.sess = tf.Session(config=session_conf)
        self.sess.as_default()
        self.sess.run(tf.global_variables_initializer())

        #restore model
        #cp_file = tf.train.latest_checkpoint(self.model_checkpoints_dir)
        #saver = tf.train.import_meta_graph('{}.meta'.format(cp_file))
        #saver.restore(self.sess, cp_file)

        #get the placeholders from graph by name
        self.input_ids_tensor = self.graph.get_operation_by_name('input_ids').outputs[0]
        self.input_mask_tensor = self.graph.get_operation_by_name('input_mask').outputs[0]
        self.segment_ids_tensor = self.graph.get_operation_by_name('segment_ids').outputs[0]
        self.is_training_tensor = self.graph.get_operation_by_name('is_training').outputs[0]


        #tensors we want to evaluate
        self.pred_labels_tensor = self.graph.get_operation_by_name('crf_pred_labels').outputs[0]
        self.probabilities_tensor = self.graph.get_operation_by_name('crf_probs').outputs[0]
        self.logits_tensor = self.graph.get_operation_by_name('logits').outputs[0]

        #self.uuid2features = self.get_memory()
   

    def get_memory(self):
        memory = {}
        with codecs.open(self.memory_file, 'r', 'utf8') as fr:
            for line in fr:
                line = line.strip()
                info = json.loads(line)
                text = info['text']
                uuid = info['uuid_code']
                features = info['features']
                memory[uuid] = features
        return memory    

    def close_session(self):
        self.sess.close()

    


    def evaluate(self, text):
        input_ids, input_mask, segment_ids = self.trans_text2ids(text)

        feed_dict = {
                self.input_ids_tensor: input_ids,
                self.input_mask_tensor: input_mask,
                self.segment_ids_tensor: segment_ids,
                self.is_training_tensor: False}
        cur_pred_labels, cur_probabilities, cur_logits = self.sess.run([self.pred_labels_tensor,
                                                                        self.probabilities_tensor, 
                                                                        self.logits_tensor],
                                                                       feed_dict)
        print(cur_pred_labels)
        print(cur_probabilities)
        tags = [self.idx2label[t].upper() for t in cur_pred_labels[0]]
        print(tags, len(tags))
        tags = tags[1: len(text) + 1]
        print(text, len(text))
        print(tags, len(tags))
        tags = ner_utils.bert_result_to_json(text, tags) 
        print(text, len(text))
        print(tags, len(tags))

        return tags 


    def trans_text2ids(self, text):
        if text[-1] in self.stop_set:
            text = text[: -1]
        example = bert_data_utils.InputExample(guid='1', text_a=text)
        #seq_length = min(self.max_seq_length, len(text) + 2)
        seq_length = self.max_seq_length
        feature = bert_data_utils.convert_online_example(example,
                                                seq_length, self.tokenizer)
        input_ids = [feature.input_ids]
        input_mask = [feature.input_mask]
        segment_ids = [feature.segment_ids]
        #print(input_ids)
        #print(input_mask)
        #print(segment_ids)
        return input_ids, input_mask, segment_ids 


if __name__ == '__main__':
    config = {}
    '''
    config['model_dir'] = MODEL_DIR
    config['model_checkpoints_dir'] = MODEL_DIR + '/checkpoints/'
    config['max_seq_length'] = 128
    config['top_k'] = 3
    config['label_map_file'] = LABEL_MAP_FILE 
    config['vocab_file'] = MODEL_DIR + '/vocab.txt'
    config['model_pb_path'] = MODEL_DIR + '/checkpoints/frozen_model.pb'
    config['memory_file'] = MODEL_DIR + '/memory.tsv'
    '''
    config['model_dir'] = MODEL_DIR 
    config['model_checkpoints_dir'] = MODEL_DIR + '/checkpoints'
    config['max_seq_length'] = 64
    config['label_map_file'] = MODEL_DIR + '/label_map'
    config['vocab_file'] = MODEL_DIR + '/vocab.txt'
    config['model_pb_path'] = MODEL_DIR + '/checkpoints/frozen_model.pb'




    pred_instance = Evaluator(config)
    rs = pred_instance.evaluate('明天我要去万达')
    print(rs)
    print(np.shape(rs))
    
    rs = pred_instance.evaluate('马思文要跟老板去上海的田亩公司')
    print(rs)


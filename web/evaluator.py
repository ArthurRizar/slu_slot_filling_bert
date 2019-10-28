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
        label2idx, idx2label = bert_data_utils.read_label_map_file(self.label_map_file)
        self.idx2label = idx2label
        self.label2idx = label2idx
        
        #self.label2code = bert_data_utils.read_code_file(self.code_file)

        self.tokenizer = tokenization.FullTokenizer(vocab_file=self.vocab_file, do_lower_case=True)

    
        #init stop set
        #self.stop_set = dataloader.get_stopwords_set(STOPWORD_FILE)

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
        self.pred_labels_tensor = self.graph.get_operation_by_name('loss/pred_labels').outputs[0]
        self.probabilities_tensor = self.graph.get_operation_by_name('loss/probs').outputs[0]
        self.logits_tensor = self.graph.get_operation_by_name('loss/logits').outputs[0]
        self.sentence_features_tensor = self.graph.get_operation_by_name('sentence_features').outputs[0]

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

    
    def extract_features(self, text, uuid=None):
        if uuid and uuid in self.uuid2features:
            return self.uuid2features[uuid]
        input_ids, input_mask, segment_ids = self.trans_text2ids(text)
        feed_dict = {         
                self.input_ids_tensor: input_ids,
                self.input_mask_tensor: input_mask,
                self.segment_ids_tensor: segment_ids,
                self.is_training_tensor: False}
        batch_sentence_features = self.sess.run(self.sentence_features_tensor, feed_dict)
        
        sentence_features = batch_sentence_features[0] 
        return sentence_features

    def ranking(self, text, cand_pool):
        start_time = datetime.datetime.now()
        cand_features = [cand['features'] for cand in cand_pool]
        end_time = datetime.datetime.now()
        cost = (end_time - start_time).total_seconds() * 1000
        logging.info('prepare cost' + str(cost))

        input_ids, input_mask, segment_ids = self.trans_text2ids(text)
        start_time = end_time
        end_time = datetime.datetime.now()
        cost = (end_time - start_time).total_seconds() * 1000
        logging.info('text to ids cost' + str(cost))

        cur_features = self.extract_features(text)
        start_time = end_time
        end_time = datetime.datetime.now()
        cost = (end_time - start_time).total_seconds() * 1000
        logging.info('extract features cost' + str(cost))

        cur_features = np.reshape(cur_features, [1, -1])
        sim_scores = cosine(cur_features, cand_features)
        start_time = end_time                        
        end_time = datetime.datetime.now()
        cost = (end_time - start_time).total_seconds() * 1000
        logging.info('cosine cost' + str(cost))

        sorted_ids = np.argsort(-sim_scores)
        start_time = end_time                        
        end_time = datetime.datetime.now()
        cost = (end_time - start_time).total_seconds() * 1000
        logging.info('sort cost' + str(cost))
        return sorted_ids, sim_scores


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
        #print(cur_pred_labels)
        #print(cur_probabilities)
        
        cur_probabilities = sigmoid(cur_logits)
        best_label = self.idx2label[cur_pred_labels[0]]
        
        all_ids = np.argsort(-cur_probabilities, 1)
        top_k_ids = all_ids[:, :self.top_k_num][0]

        top_k_labels = [self.idx2label[idx] for idx in top_k_ids]
        top_k_probs = cur_probabilities[0][top_k_ids]
        top_k_probs = norm(top_k_probs)
        top_k_code = [self.label2code[label] for label in top_k_labels]

        return zip(top_k_labels, top_k_code, top_k_probs) 


    def trans_text2ids(self, text):
        if text[-1] in self.stop_set:
            text = text[: -1]
        example = bert_data_utils.InputExample(guid='1', text_a=text)
        seq_length = min(self.max_seq_length, len(text) + 2)
        feature = bert_data_utils.convert_single_example(1, example, self.label2idx,
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
    MODEL_DIR = '../output'
    config['model_dir'] = MODEL_DIR 
    config['model_checkpoints_dir'] = MODEL_DIR + '/checkpoints'
    config['max_seq_length'] = 64
    config['label_map_file'] = MODEL_DIR + '/label_map'
    config['vocab_file'] = MODEL_DIR + '/vocab.txt'
    config['model_pb_path'] = MODEL_DIR + '/checkpoints/frozen_model.pb'




    pred_instance = Evaluator(config)
    rs = pred_instance.extract_features('明天我要去万达')
    print(rs)
    print(np.shape(rs))



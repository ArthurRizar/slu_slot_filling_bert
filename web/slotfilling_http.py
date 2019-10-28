#coding:utf-8
###################################################
# File Name: httpserver.py
# Author: Meng Zhao
# mail: @
# Created Time: 2018年06月06日 星期三 15时24分44秒
#=============================================================

import os
import sys
import ujson as json
import tornado.web
import tornado.ioloop


import numpy as np
import datetime
import logging
import traceback


sys.path.append('../')


import evaluator

#from tornado.concurrent import run_on_executor
#from concurrent.futures import ThreadPoolExecutor


from multiprocessing import Pool, TimeoutError

from setting import *
from evaluator import Evaluator
from preprocess import dataloader
from common.segment.segment_client import SegClient


os.environ["CUDA_VISIBLE_DEVICES"] = "" #不使用GPU




class NERHttpServer(tornado.web.RequestHandler):

    def initialize(self, pred_instance):
        self.pred_instance = pred_instance

    @tornado.gen.coroutine
    def head(self):
        self.write('OK')

    def online_extract_features(self):
        query = self.data['query']
        logging.info('extracted query:' + query)
        result = {}
        if 'uuid_code' not in self.data:
            features = self.pred_instance.extract_features(query)
        else:
            uuid_code = self.data['uuid_code']
            features = self.pred_instance.extract_features(query, uuid_code)
        result['features'] = [float(feature) for feature in features]
        return result
    
    def online_ranking(self):
        start_time = datetime.datetime.now()
        query = self.data['query']
        cand_pool = self.data['candidates']
        logging.info('ranking query:' + query)
        end_time = datetime.datetime.now()
        cost = (end_time - start_time).total_seconds() * 1000
        logging.info('candidates param cost' + str(cost))
        sorted_indices, sim_scores = self.pred_instance.ranking(query, cand_pool)
        
        sim_scores = list(sim_scores)
        sorted_indices = list(sorted_indices)
        top_sorted_indices = sorted_indices[: 5]
        logging.info(type(sim_scores))
        logging.info(type(sorted_indices))

        result = []
        start_time = end_time
        for idx in top_sorted_indices:
            title = cand_pool[idx]['text']
            name = cand_pool[idx]['userId']
            score = sim_scores[idx]
            item = {}
            item['title'] = title
            item['name'] = name
            item['score'] = score
            result.append(item)
        end_time = datetime.datetime.now()
        cost = (end_time - start_time).total_seconds() * 1000
        logging.info('wrap result cost' + str(cost))
        return result
            
    def prepare(self):
        start_time = datetime.datetime.now()
        if self.request.headers.get("Content-Type", "").startswith("application/json"):
            self.data = json.loads(self.request.body)
        else:
            self.data = {}
            for k in self.request.arguments:
                self.data[k] = self.get_argument(k)
        end_time = datetime.datetime.now()
        cost = (end_time - start_time).total_seconds() * 1000
        logging.info('json convert cost' + str(cost))

    @tornado.gen.coroutine   
    def main_process(self):
        err_dict = {}
        try:
            start_time = datetime.datetime.now()
            end_time = datetime.datetime.now()
            cost = (end_time - start_time).total_seconds() * 1000
            logging.info('check param cost' + str(cost))
            start_time = end_time
            method_type = self.data['method']
            end_time = datetime.datetime.now()
            cost = (end_time - start_time).total_seconds() * 1000
            logging.info('get method param and query param cost' + str(cost))
            if method_type == 'extract':
                response = self.online_extract_features()
            else:
                response = self.online_ranking()
            response_json = json.dumps(response, ensure_ascii=False)
            self.write(response_json)

        except Exception as err:
            err_dict['errMsg'] = traceback.format_exc()
            self.write(json.dumps(err_dict, ensure_ascii=False))
            logging.warning(traceback.format_exc())

    def get(self):
        self.main_process()

    def post(self):
        self.main_process()




if __name__ == '__main__':


    config = {}
    config['model_dir'] = MODEL_DIR
    config['model_checkpoints_dir'] = MODEL_DIR + '/checkpoints/'
    config['max_seq_length'] = 32
    config['top_k'] = 3
    config['code_file'] = CODE_FILE
    config['label_map_file'] = LABEL_MAP_FILE
    config['vocab_file'] = MODEL_DIR + '/vocab.txt'
    config['model_pb_path'] = MODEL_DIR + '/checkpoints/frozen_model.pb'
    config['memory_file'] = MODEL_DIR + '/memory.tsv'

    pred_instance = Evaluator(config)
    pred_instance.extract_features('班车报表')


    application = tornado.web.Application([
            (r"/ner", NERHttpServer,
            dict(pred_instance=pred_instance))
        ])
    application.listen(6379)
   
    tornado.ioloop.IOLoop.instance().start()

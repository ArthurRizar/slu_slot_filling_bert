#coding:utf-8
###################################################
# File Name: ner_utils.py
# Author: Meng Zhao
# mail: @
# Created Time: 2019年10月28日 星期一 15时28分03秒
#=============================================================
def result_to_json(string, tags):
    item = {"string": string, "entities": []}
    entity_name = ""
    entity_start = 0
    idx = 0
    for char, tag in zip(string, tags):
        if tag[0] == "S":
            item["entities"].append({"word": char, "start": idx, "end": idx + 1, "type": tag[2: ]})
        elif tag[0] == "B":
            entity_name += char
            entity_start = idx
        elif tag[0] == "I" or tag[0] == 'M':
            entity_name += char
        elif tag[0] == "E":
            entity_name += char
            item["entities"].append({"word": entity_name, "start": entity_start, "end": idx + 1, "type": tag[2: ]})
            entity_name = ""
        else:
            entity_name = ""
            entity_start = idx
        idx += 1
    return item

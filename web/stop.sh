#coding:utf-8
###################################################
# File Name: stop.sh
# Author: Meng Zhao
# mail: @
# Created Time: 2019年10月28日 星期一 15时58分42秒
#=============================================================
CLIENT_NAME=ner_http.py
ps -ef | grep "python $CLIENT_NAME" | grep -v 'grep' 
ps -ef | grep "python $CLIENT_NAME" | grep -v 'grep' | awk '{print $2}' | xargs -n1 kill -9

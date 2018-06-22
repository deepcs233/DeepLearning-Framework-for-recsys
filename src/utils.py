#encoding=utf-8

#import psutil
import os
import re

from ioManger import Vec

RE_SLOT = re.compile(r'slot_(\d)+')

def extractInputNode(node_str_list, iM):
    nodes = []
    for node_str in node_str_list:
        if 'bias' in node_str:
            nodes.append(iM.initBias())
        elif 'slot' in node_str:
            slot_id, vec_length = [int(x) for x in node_str[5:].split('-')]
            nodes.append(Vec(slot_id, iM).pureApply(vec_length))
        else:
            raise Exception('Input layer includes illegal input nodes: %s' % (node_str))
    return nodes

'''
def memoryCheck():
    info = psutil.virtual_memory()
    print u'内存使用：', psutil.Process(os.getpid()).memory_info().rss
    print u'总内存：', info.total
    print u'内存占比：', info.percent
    print u'cpu个数：', psutil.cpu_count()
'''

def extractSlotId(slot_string):
    res = RE_SLOT.match(slot_string)
    if res is None:
        raise Exception("This type of slot input: %s  isn't allowed. It should "
                        "be like slot_XX" % (slot_string))
    else:
        return int(res.group(1))

import os
import base64
import numpy as np
import csv
import sys
import zlib
import time
import mmap
import argparse

parser = argparse.ArgumentParser()

# output_dir
parser.add_argument('--downloaded_feats', default='/media/jxgu/d4t/dataset/ai_challenger/image_captioning/bu_data', help='downloaded feature directory')
parser.add_argument('--output_dir', default='/media/jxgu/d4t/dataset/ai_challenger/image_captioning/bu_data/bu', help='output feature files')

args = parser.parse_args()

csv.field_size_limit(sys.maxsize)


FIELDNAMES = ['image_id', 'image_w','image_h','num_boxes', 'boxes', 'features']
#infiles = ['karpathy_test_resnet101_faster_rcnn_genome.0']
infiles = ['chinese_feats_val.0']

#infiles = ['chinese_train_resnet101_faster_rcnn_genome.0',
#           'chinese_train_resnet101_faster_rcnn_genome.1',
#           'chinese_train_resnet101_faster_rcnn_genome.2',
#           'chinese_train_resnet101_faster_rcnn_genome.3',
#           'chinese_train_resnet101_faster_rcnn_genome.4',
#           'chinese_train_resnet101_faster_rcnn_genome.5']

#infiles = ['karpathy_val_resnet101_faster_rcnn_genome.0']
#infiles = ['trainval/karpathy_test_resnet101_faster_rcnn_genome.0', 'trainval/karpathy_val_resnet101_faster_rcnn_genome.0']
#infiles = ['karpathy_train_resnet101_faster_rcnn_genome.0', 'karpathy_train_resnet101_faster_rcnn_genome.1']

if not os.path.exists(args.output_dir+'_att'): os.makedirs(args.output_dir+'_att')
if not os.path.exists(args.output_dir+'_fc'): os.makedirs(args.output_dir+'_fc')
if not os.path.exists(args.output_dir+'_box'): os.makedirs(args.output_dir+'_box')
#if not os.path.exists(args.output_dir+'_box_cls_prob'): os.makedirs(args.output_dir+'_box_cls_prob')
#if not os.path.exists(args.output_dir+'_box_keep_boxes'): os.makedirs(args.output_dir+'_box_keep_boxes')

for infile in infiles:
    print('Reading ' + infile)
    with open(os.path.join(args.downloaded_feats, infile), "r+b") as tsv_in_file:
        reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames = FIELDNAMES)
        for item in reader:
            #item['image_id'] = int(item['image_id'])
            item['image_id'] = item['image_id']
            item['num_boxes'] = int(item['num_boxes'])
            for field in ['boxes', 'features']:
                item[field] = np.frombuffer(base64.decodestring(item[field]), dtype=np.float32).reshape((item['num_boxes'],-1))
            #for field in ['cls_prob']:
            #    item[field] = np.frombuffer(base64.decodestring(item[field]), dtype=np.float32).reshape((item['num_boxes'],-1))
            np.savez_compressed(os.path.join(args.output_dir+'_att', str(item['image_id'])), feat=item['features'])
            np.save(os.path.join(args.output_dir+'_fc', str(item['image_id'])), item['features'].mean(0))
            np.save(os.path.join(args.output_dir+'_box', str(item['image_id'])), item['boxes'])
            #np.save(os.path.join(args.output_dir + '_box_cls_prob', str(item['image_id'])), item['cls_prob'])
            #np.save(os.path.join(args.output_dir + '_box_keep_boxes', str(item['image_id'])), item['keep_boxes'])


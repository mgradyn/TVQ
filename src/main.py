import os
import time
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import sys
sys.path.append('/home/TVQ/')

import main_task_arithmetic, main_emr_merging, main_lines, main_adamerging #, main_ties_merging, main_tall_mask
from args import parse_arguments


def create_log_dir(path, filename='log.txt'):
    import logging
    if not os.path.exists(path):
        os.makedirs(path)
    logger = logging.getLogger(path)
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(path+'/'+filename)
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger

def main():
    args = parse_arguments()

    model = 'ViT-B-32'
    source_root_path = '/home/TVQ'
    pretrained_checkpoint = source_root_path+'/checkpoints/'+model+'/zeroshot.pt'
    exam_datasets = ['SUN397', 'Cars', 'RESISC45', 'EuroSAT', 'SVHN', 'GTSRB', 'MNIST', 'DTD']

    args.data_location = '/home/TVQ/data'
    args.model = model
    args.save = source_root_path+'/checkpoints/' + model
    args.logs_path = source_root_path+'/logs/' + model

    str_time_ = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
    log = create_log_dir(args.logs_path, f'log_{str_time_}_{args.method}.txt')
    
    if args.method == "task_arithmetic":
        log.info("Running Task Arithmetic...")
        main_task_arithmetic.run(args, log, exam_datasets, pretrained_checkpoint, source_root_path)
    elif args.method == "adamerging":
        log.info("Running AdaMerging...")
        main_adamerging.run(args, log, exam_datasets, pretrained_checkpoint, source_root_path)
    elif args.method == "emr_merging":
        log.info("Running EMR Merging...")
        main_emr_merging.run(args, log, exam_datasets, pretrained_checkpoint, source_root_path)
    elif args.method == "lines":
        log.info("Running LiNeS...")
        main_lines.run(args, log, exam_datasets, pretrained_checkpoint, source_root_path)
    else:
        raise ValueError(f"Unknown method: {args.method}")

if __name__ == "__main__":
    main()
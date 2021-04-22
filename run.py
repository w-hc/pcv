import os
import os.path as osp
import argparse
import random

import torch
import torch.multiprocessing as mp
import torch.distributed as dist

import panoptic
from panoptic.entry import Entry
from fabric.utils.mailer import ExceptionEmail
from fabric.utils.logging import setup_logging
from fabric.utils.git import git_version

logger = setup_logging(__file__)


def main():
    parser = argparse.ArgumentParser(description="run script")
    parser.add_argument('--command', '-c', type=str, default='train')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--dist', action='store_true')
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    git_hash = git_version(osp.dirname(panoptic.__file__))
    logger.info('git repository hash: {}'.format(git_hash))

    global fly
    # only decorate when not in interative mode; Bugs are expected there.
    if 'INTERACTIVE' not in os.environ:
        recipient = 'whc@ttic.edu'
        logger.info("decorating with warning email to {}".format(recipient))
        email_subject_headline = "{} tripped".format(
            # take the immediate dirname as email label
            osp.dirname(osp.abspath(__file__)).split('/')[-1]
        )
        fly = ExceptionEmail(
            subject=email_subject_headline, address=recipient
        )(fly)

    ngpus = torch.cuda.device_count()
    port = random.randint(10000, 20000)
    argv = (ngpus, args.command, args.debug, args.opts, port)
    if args.dist:
        mp.spawn(fly, nprocs=ngpus, args=argv)
    else:
        fly(None, *argv)


def fly(rank, ngpus, command, debug, opts, port):
    distributed = rank is not None  # and not debug
    if distributed:  # multiprocess distributed training
        dist.init_process_group(
            world_size=ngpus, rank=rank,
            backend='nccl', init_method=f'tcp://127.0.0.1:{port}',
        )
        assert command == 'train'  # for now only train uses mp distributed
        torch.cuda.set_device(rank)

    entry = Entry(
        __file__, override_opts=opts, debug=debug,
        mp_distributed=distributed, rank=rank, world_size=ngpus
    )
    if command == 'train':
        entry.train()
    elif command == 'validate': # for evaluate semantic segmentation mean iou
        entry.validate(False)
    elif command == 'evaluate':
        entry.evaluate()
    elif command == 'report':
        entry.report()
    elif command == 'test':
        entry.PQ_test(save_output=True)
    elif command == 'make_figures':
        entry.make_figures()
    else:
        raise ValueError("unrecognized command")


if __name__ == '__main__':
    main()

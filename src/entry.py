import os
import os.path as osp
from functools import partial
import json
from pprint import pprint

import numpy as np
from PIL import Image
import torch
import apex

from . import cfg
from .utils import CompoundTimer
from .pcv import get_pcv
#from .models import get_model_module, convert_inplace_sync_batchnorm, convert_naive_sync_batchnorm
from .models import *
from .datasets import get_dataset_module
from .metric import PanMetric as Metric, PQMetric
from .reporters import reporter_modules
from .vis import d2_vis

from panopticapi.utils import rgb2id, id2rgb
from fabric.utils.logging import setup_logging
logger = setup_logging(__file__)


class Entry():
    def __init__(
        self, cfg_yaml_path, override_opts=None, debug=False,
        mp_distributed=False, rank=None, world_size=None, val_split='val'
    ):
        """
        Args:
            cfg_yaml_path: path to the directory where the yaml config is stored
                the logic will first parse the dir from the path, and then look
                for 'config.yml' in this directory.
        """
        torch.manual_seed(120)
        np.random.seed(120)
        # preliminary setup
        self.debug = debug
        mp_distributed = mp_distributed
        self.mp_distributed = mp_distributed
        if mp_distributed:
            assert rank is not None and world_size is not None
        self.rank, self.world_size = rank, world_size
        cfg.init_state(cfg_yaml_path, override_opts)
        ckpt_mngr = cfg.manager.get_ckpt_writer(
            save_f=torch.save, load_f=partial(torch.load, map_location='cpu')
        )
        output_mngr = cfg.manager.get_output_writer()
        tboard_writer = cfg.manager.get_tboard_writer()
        ckpt_mngr.delete_last = True
        ckpt_mngr.keep_interval = 10
        self.ckpt_mngr, self.output_mngr, self.tboard_writer = \
            ckpt_mngr, output_mngr, tboard_writer
        self.leader_log("experiment configs:\n{}".format(cfg))

        if self.debug:
            self.leader_log("Entering debugging mode. 0 worker.")
            cfg.data.num_loading_threads = 0

        torch.backends.cudnn_benchmark = cfg.runtime.cudnn_benchmark
        torch.autograd.set_detect_anomaly(cfg.runtime.detect_anomaly)

        # assembling components
        # 1. pixel consensus voting module
        pcv = get_pcv(cfg.pcv)

        # 2. data loaders
        dset_module = get_dataset_module(cfg.data.dataset.name)
        self.train_loader = dset_module.make_loader(
            cfg.data, pcv, is_train=True,
            mp_distributed=self.mp_distributed, world_size=self.world_size,
        )
        self.val_loader = dset_module.make_loader(
            cfg.data, pcv, is_train=False,
            mp_distributed=self.mp_distributed, world_size=self.world_size,
            val_split=val_split
        )
        # if getattr(cfg, 'testing', False) and cfg.data.dataset.name.startswith('coco'):
        # no test loader for cityscapes
        if cfg.data.dataset.name.startswith('coco'):
            self.test_loader = dset_module.make_loader(
                cfg.data, pcv, is_train=False,
                mp_distributed=self.mp_distributed, world_size=self.world_size,
                val_split='test-dev'
            )
        dset_meta = self.train_loader.dataset.meta
        self.gt_prod_handle = self.train_loader.dataset.gt_prod_handle
        # logger.info("dataset meta info:\n{}".format(pformat(dset_meta)))

        # 3. model
        model = self.get_model(cfg, pcv, dset_meta)
        model.add_log_writer(tboard_writer)

        # load final coco checkpoint
        # model.net.module.load_state_dict(torch.load(
        #     '/share/data/vision-greg2/users/whc/lab/panoptic/ablations/coco/runs/co_ups_pan/checkpoint/19-11-12-15-02-23_iter12.ckpt')['model_state']
        # )

        # Load resnext.
        # model.net.module.load_state_dict(torch.load(
        #     '/home-nfs/whc/glab/panoptic/exp/co_sem/runs/co_upsx152_pan/checkpoint_correct/19-09-28-00-10-55_iter12.ckpt')['model_state']
        # )

        # load final cs checkpoint
        # model.net.module.load_state_dict(torch.load(
        #     '/share/data/vision-greg2/users/whc/lab/panoptic/ablations/backbone/runs/UPS_res50_freeze/checkpoint/19-11-13-13-32-00_iter64.ckpt')['model_state']
        # )

        model.load_latest_checkpoint_if_available(ckpt_mngr)
        # if getattr(cfg, 'testing', False) and cfg.data.dataset.name.startswith('coco'):
        #     model.net.module.load_state_dict(torch.load(
        #         '/home-nfs/whc/panoptic/ablations/coco/runs/co_ups_pan/checkpoint/19-11-12-15-02-23_iter12.ckpt')['model_state']
        #     )
        self.model = model

        # Only for ensemble
        # model2 = self.get_model(cfg, pcv, dset_meta)
        # model2.net.module.load_state_dict(torch.load('/home-nfs/whc/glab/panoptic/exp/co_sem/runs/co_upsx152_pan/checkpoint_correct/19-09-28-00-10-55_iter12.ckpt')['model_state'])
        # self.model2 = model2
        # assert list(self.model.net.module.parameters())[0].data_ptr() != list(self.model2.net.module.parameters())[0].data_ptr()

    @property
    def is_leader(self):
        if self.mp_distributed:
            return self.rank == 0
        else:
            return True

    def leader_log(self, msg, raw_print=False):
        if self.is_leader:
            if raw_print:
                print(msg)
            else:
                logger.info(msg)

    def get_model(self, cfg, pcv_module, dset_meta):
        model_module = get_model_module(cfg.model.name)
        inst = model_module(cfg, pcv_module, dset_meta)
        inst.net = inst.net.cuda()
        if not self.mp_distributed:
            inst.net = torch.nn.DataParallel(inst.net)
        else:
            if getattr(cfg.model, 'inplace_abn', False):
                inst.net = convert_inplace_sync_batchnorm(inst.net)
            elif getattr(cfg.model, 'naive_syncbn', False):  # detectron2 sync bn
                inst.net = convert_naive_sync_batchnorm(inst.net)
            elif getattr(cfg, 'apex', False):
                inst.net = convert_apex_sync_batchnorm(inst.net)
            else:
                inst.net = torch.nn.SyncBatchNorm.convert_sync_batchnorm(inst.net)
            if getattr(cfg, 'apex', False):
                inst.net, inst.optimizer = apex.amp.initialize(inst.net.cuda(), inst.optimizer, opt_level=getattr(cfg, 'opt_level', 'O0'), keep_batchnorm_fp32=getattr(cfg, 'keep_batchnorm_fp32', None), loss_scale=getattr(cfg, 'loss_scale', None))
                from . import get_scheduler; inst.scheduler = get_scheduler(cfg.scheduler)(optimizer=inst.optimizer)
                inst.net = apex.parallel.DistributedDataParallel(inst.net)
            else:
                inst.net = torch.nn.parallel.DistributedDataParallel(
                    inst.net, device_ids=[self.rank], find_unused_parameters=True
                )
        return inst

    # def train(self):
    #     loader = self.train_loader
    #     model = self.model
    #     torch.manual_seed(0)
    #     np.random.seed(0)
    #     for epoch in range(model.curr_epoch, model.total_train_epoch):
    #         print('current epoch {}'.format(epoch))
    #         for i, inputs in enumerate(loader):
    #             if i % 50 == 0:
    #                 print('iter {}'.format(i))

    def train(self):
        loader = self.train_loader
        dset = self.train_loader.dataset
        model = self.model
        self.leader_log("dataset size: {}".format(len(dset)))
        self.leader_log("num iterations per epoch: {}".format(len(loader)))
        # maybe later for training
        metric = Metric(
            model.dset_meta['num_classes'],
            model.pcv.num_votes, model.dset_meta['trainId_2_catName']
        )

        val_freq = cfg.runtime.val_freq
        break_freq = cfg.runtime.train_break
        progress = 0
        for epoch in range(model.curr_epoch, model.total_train_epoch):
            metric.init_state()
            if self.mp_distributed:
                loader.batch_sampler.sampler.set_epoch(epoch)
            timer = CompoundTimer()
            timer.data.tic()
            for i, inputs in enumerate(loader):
                global_step = epoch * len(loader) + i
                model.scheduler.step(global_step)
                if self.debug and i > 2:
                    break
                inputs = [ el.cuda() for el in inputs ]
                # torch.cuda.synchronize()
                timer.data.toc()
                timer.compute.tic()
                model.ingest_train_input(*inputs)
                model.optimize_params()
                # torch.cuda.synchronize()
                timer.compute.toc()
                # print('iter {} rank {}: {}'.format(i, self.rank, [tsr.shape for tsr in inputs]))
                if self.is_leader:
                    model.log_statistics(epoch * len(loader) + i, level=1)
                if i % 50 == 0:
                    pred = model.infer(inputs[0], take_argmax=True)
                    metric.update(*pred, *inputs[1:3])
                    self.leader_log(timer)
                    self.leader_log(
                        "rank {} epoch {}/{} iter {}/{}, eta to epoch end {}, to train end {}".format(
                            self.rank, epoch, model.total_train_epoch, i, len(loader),
                            timer.eta(i, len(loader)),
                            timer.eta(i, len(loader) * (model.total_train_epoch - epoch))
                        )
                    )
                    self.leader_log("{}".format(model.latest_loss()))
                    self.leader_log(metric, raw_print=True)
                timer.data.tic()
            if self.is_leader:
                model.write_checkpoint(self.ckpt_mngr)  # eyesore
            model.advance_to_next_epoch()  # update lr, increment inx, etc
            if val_freq > 0 and epoch % val_freq == 0:
                self.validate()
            progress += 1
            if break_freq > 0 and progress == break_freq:
                self.leader_log("training breaks")
                break
        self.leader_log("training halts")
        self.validate()

    def evaluate(self):
        # self.validate()
        # save_path = osp.join(self.output_mngr.root, 'res.pkl')
        # self.output_mngr.save_f(metric.scores, save_path)
        self.PQ_eval(
            save_output=cfg.evaluate.save_pq, oracle_mode=cfg.evaluate.oracle_mode
        )

    def validate(self, to_write=True):
        model = self.model
        loader = self.val_loader
        dset = loader.dataset

        eval_at = model.curr_epoch - 1
        self.leader_log("evaluating at epoch {}".format(eval_at))
        cudnn_old_state = torch.backends.cudnn.benchmark
        torch.backends.cudnn.benchmark = False  # input size could vary
        self.leader_log("eval dset size: {}".format(len(dset)))
        self.leader_log("eval num iterations: {}".format(len(loader)))
        metric = Metric(
            model.dset_meta['num_classes'],
            model.pcv.num_votes, model.dset_meta['trainId_2_catName']
        )
        timer = CompoundTimer()
        timer.data.tic()
        for i, inputs in enumerate(loader):
            if self.debug and (i > 1):
                break
            x = inputs[0].cuda()
            # torch.cuda.synchronize()
            timer.data.toc()
            timer.compute.tic()
            pred = model.infer(x, upsize_sem=True, take_argmax=True)
            metric.update(*pred, *inputs[1:3])
            # torch.cuda.synchronize()
            timer.compute.toc()
            if (i % 50) == 0:
                self.leader_log(timer)
                self.leader_log(
                    "iter {}, eta to val end {}".format(i, timer.eta(i, len(loader)))
                )
            timer.data.tic()
        if to_write:
            metric.save("{}_val.pkl".format(eval_at), self.output_mngr)
        self.leader_log("eval complete \n{}".format(metric))
        torch.backends.cudnn.benchmark = cudnn_old_state
        return metric

    def PQ_eval(
        self, oracle_res=4, save_output=False,
        oracle_mode=None
        # oracle=False, semantic_oracle=False, vote_oracle=False
    ):
        """
        optionally save all predicted files through output writer
        """
        _VALID_ORACLE_MODES = ('full', 'sem', 'vote')

        assert not self.mp_distributed
        model = self.model
        loader = self.val_loader
        dset = loader.dataset
        eval_at = model.curr_epoch - 1
        logger.info("Panoptic Quality eval at epoch {}".format(eval_at))

        # setup space to save pan predictions
        PRED_OUT_NAME = 'pred'
        dump_root = osp.join(self.output_mngr.root, 'pd_dump')
        pred_meta_fname = osp.join(dump_root, '{}.json'.format(PRED_OUT_NAME))
        pred_mask_dir = osp.join(dump_root, PRED_OUT_NAME)
        del dump_root

        overall_pred_meta = {
            'images': list(dset.imgs.values()),
            'categories': list(dset.meta['cats'].values()),
            'annotations': []
        }

        metric = PQMetric(dset.meta)
        timer = CompoundTimer()
        timer.data.tic()

        for i, inputs in enumerate(loader):
            x = inputs[0].cuda()
            del inputs
            if self.debug and (i > 5):
                break

            imgMeta, segments_info, _, pan_gt_mask = dset.pan_getitem(
                i, apply_trans=False)

            if dset.transforms is not None:
                _, trans_pan_gt = dset.transforms(_, pan_gt_mask)
            else:
                trans_pan_gt = pan_gt_mask.copy()

            pan_gt_mask = pan_gt_mask
            pan_gt_ann = {
                'image_id': imgMeta['id'],
                # shameful mogai; can only access image f_name here. alas...
                'file_name': imgMeta['file_name'].split('.')[0] + '.png',
                'segments_info': list(segments_info.values())
            }

            # torch.cuda.synchronize()
            timer.data.toc()
            timer.compute.tic()

            if oracle_mode != 'full':  # way too ugly
                # from fabric.utils.timer import global_timer
                # global_timer.network.tic()
                sem_pd, vote_pd = model.infer(x, softmax_normalize=True)
                # global_timer.network.toc()

            if oracle_mode is not None:
                assert oracle_mode in _VALID_ORACLE_MODES
                sem_ora, vote_ora = gt_tsr_res_reduction(
                    oracle_res, self.gt_prod_handle,
                    dset.meta, model.pcv, trans_pan_gt, segments_info
                )
                if oracle_mode == 'vote':
                    pass  # if using model sem pd, maintain stuff pred thresh
                else:  # sem or full oracle, using gt sem pd, do not filter
                    model.dset_meta['stuff_pred_thresh'] = -1

                if oracle_mode == 'sem':
                    sem_pd = sem_ora
                elif oracle_mode == 'vote':
                    vote_pd = vote_ora
                else:
                    sem_pd, vote_pd = sem_ora, vote_ora

            pan_pd_mask, pan_pd_ann = model.stitch_pan_mask(
                cfg.pcv, sem_pd, vote_pd, pan_gt_mask.size
            )

            pan_pd_ann['image_id'] = pan_gt_ann['image_id']
            pan_pd_ann['file_name'] = pan_gt_ann['file_name']
            overall_pred_meta['annotations'].append(pan_pd_ann)

            metric.update(
                pan_gt_ann, rgb2id(np.array(pan_gt_mask)), pan_pd_ann, pan_pd_mask
            )

            if save_output:
                # pan_pd_mask = Image.fromarray(id2rgb(pan_pd_mask))
                # fname = osp.join(pred_mask_dir, pan_pd_ann['file_name'])
                # os.makedirs(osp.dirname(fname), exist_ok=True)  # make region subdir
                # pan_pd_mask.save(fname)
                fname = osp.join(pred_mask_dir, pan_pd_ann['file_name'])
                os.makedirs(osp.dirname(fname), exist_ok=True)  # make region subdir
                pan_pd_mask = Image.fromarray(id2rgb(pan_pd_mask))
                pan_pd_mask.save(fname)

            # torch.cuda.synchronize()
            timer.compute.toc()

            if (i % 50) == 0:
                logger.info(timer)
                logger.info(
                    "iter {}, eta to val end {}".format(i, timer.eta(i, len(loader)))
                )
                print(metric)
            timer.data.tic()

        if save_output:
            with open(pred_meta_fname, 'w') as f:
                json.dump(overall_pred_meta, f)
            self.output_mngr.save_f(
                metric.results, osp.join(self.output_mngr.root, 'score.pkl')
            )
        logger.info("eval complete \n{}".format(metric))
        return metric

    def PQ_eval_dirty(
        self, oracle_res=4, save_output=False,
        oracle_mode=None
        # oracle=False, semantic_oracle=False, vote_oracle=False
    ):
        """
        optionally save all predicted files through output writer
        """
        _VALID_ORACLE_MODES = ('full', 'sem', 'vote')

        assert not self.mp_distributed
        model = self.model
        loader = self.val_loader
        dset = loader.dataset
        eval_at = model.curr_epoch - 1
        logger.info("Panoptic Quality eval at epoch {}".format(eval_at))

        # setup space to save pan predictions
        PRED_OUT_NAME = 'pred'
        dump_root = osp.join(self.output_mngr.root, 'pd_dump')
        pred_meta_fname = osp.join(dump_root, '{}.json'.format(PRED_OUT_NAME))
        pred_mask_dir = osp.join(dump_root, PRED_OUT_NAME)
        del dump_root

        overall_pred_meta = {
            'images': list(dset.imgs.values()),
            'categories': list(dset.meta['cats'].values()),
            'annotations': []
        }

        metric = PQMetric(dset.meta)
        timer = CompoundTimer()
        timer.data.tic()

        # sota_sseg = SOTASSeg(dset.name)

        # do nasty
        upsnet_pred_json = json.load(open('/home-nfs/whc/panout/upsnet/coco/val/pred.json', 'r'))
        upsnet_pred_ann = {_['image_id']: _ for _ in upsnet_pred_json['annotations']}
        ups_thing_cat_ids = [_['id'] for _ in upsnet_pred_json['categories'] if _['isthing']]
        ups_stuff_cat_ids = [_['id'] for _ in upsnet_pred_json['categories'] if _['isthing'] == 0]
        def load_pred_mask(filename):
            return rgb2id(np.array(Image.open(f'/home-nfs/whc/panout/upsnet/coco/val/pred/{filename}').convert('RGB')))

        for i, inputs in enumerate(loader):
            x = inputs[0].cuda()
            del inputs
            if self.debug and (i > 5):
                break

            imgMeta, segments_info, _, pan_gt_mask = dset.pan_getitem(
                i, apply_trans=False)

            # rt change
            # to get ground truth instance centroid
            # lo_pan_mask = pan_gt_mask.resize(
            #     np.array(_.size, dtype=np.int) // 4, resample=Image.NEAREST
            # )
            # tmp_handle = dset.gt_prod_handle(dset.meta, dset.pcv,
            #     lo_pan_mask, segments_info)
            # tmp_handle.generate_gt()
            # gt_ins_centroids = tmp_handle.ins_centroids

            if dset.transforms is not None:
                _, trans_pan_gt = dset.transforms(_, pan_gt_mask)
            else:
                trans_pan_gt = pan_gt_mask.copy()

            pan_gt_mask = pan_gt_mask
            pan_gt_ann = {
                'image_id': imgMeta['id'],
                # shameful mogai; can only access image f_name here. alas...
                'file_name': imgMeta['file_name'].split('.')[0] + '.png',
                'segments_info': list(segments_info.values())
            }

            # torch.cuda.synchronize()
            timer.data.toc()
            timer.compute.tic()

            if oracle_mode != 'full':  # way too ugly
                from fabric.utils.timer import global_timer
                global_timer.network.tic()
                sem_pd, vote_pd = model.infer(x, softmax_normalize=True)
                global_timer.network.toc()

            if oracle_mode is not None:
                assert oracle_mode in _VALID_ORACLE_MODES
                sem_ora, vote_ora = gt_tsr_res_reduction(
                    oracle_res, self.gt_prod_handle,
                    dset.meta, model.pcv, trans_pan_gt, segments_info
                )
                if oracle_mode == 'vote':
                    pass  # if using model sem pd, maintain stuff pred thresh
                else:  # sem or full oracle, using gt sem pd, do not filter
                    model.dset_meta['stuff_pred_thresh'] = -1

                if oracle_mode == 'sem':
                    sem_pd = sem_ora
                elif oracle_mode == 'vote':
                    vote_pd = vote_ora
                else:
                    sem_pd, vote_pd = sem_ora, vote_ora

            pan_pd_mask, pan_pd_ann = model.stitch_pan_mask(
                cfg.pcv, sem_pd, vote_pd, pan_gt_mask.size
            )

            # if oracle:
            #     sem_pd, vote_pd = gt_tsr_res_reduction(
            #         oracle_res, self.gt_prod_handle,
            #         dset.meta, model.pcv, trans_pan_gt, segments_info
            #     )
            #     model.dset_meta['stuff_pred_thresh'] = -1
            #     pan_pd_mask, pan_pd_ann = model.stitch_pan_mask(
            #         sem_pd, vote_pd, pan_gt_mask.size)
            # elif semantic_oracle:
            #     sem_pd, vote_pd = model.infer(x, softmax_normalize=True)
            #     sem_pd, _ = gt_tsr_res_reduction(
            #         oracle_res, self.gt_prod_handle,
            #         dset.meta, model.pcv, trans_pan_gt, segments_info
            #     )
            #     model.dset_meta['stuff_pred_thresh'] = -1
            #     pan_pd_mask, pan_pd_ann = model.stitch_pan_mask(
            #         sem_pd, vote_pd, pan_gt_mask.size)
            # elif vote_oracle:
            #     sem_pd, vote_pd = model.infer(x, softmax_normalize=True)
            #     _, vote_pd = gt_tsr_res_reduction(
            #         oracle_res, self.gt_prod_handle,
            #         dset.meta, model.pcv, trans_pan_gt, segments_info
            #     )
            #     pan_pd_mask, pan_pd_ann = model.stitch_pan_mask(
            #         sem_pd, vote_pd, pan_gt_mask.size)
            # else:
            #     # _sem_pd, _vote_pd = gt_tsr_res_reduction(
            #     #     oracle_res, self.gt_prod_handle,
            #     #     dset.meta, model.pcv, trans_pan_gt, segments_info
            #     # )
            #     sem_pd, vote_pd = model.infer(x, softmax_normalize=True)
            #     # if hasattr(self, 'model2'):
            #     #     _sem_pd, _vote_pd = self.model2.infer(x, softmax_normalize=True)
            #     #     sem_pd = (sem_pd + _sem_pd) / 2
            #     #     vote_pd = (vote_pd + _vote_pd) / 2
            #     # sem_pd, vote_pd = model.flip_infer(x, softmax_normalize=True)
            #     # sem_pd = sota_sseg.get(i, sem_pd.shape[-2:])
            #     # model.dset_meta['tmp_gt_ins_centroids'] = gt_ins_centroids # rt change
            #     pan_pd_mask, pan_pd_ann = model.stitch_pan_mask(
            #         sem_pd, vote_pd, pan_gt_mask.size)
            pan_pd_ann['image_id'] = pan_gt_ann['image_id']
            pan_pd_ann['file_name'] = pan_gt_ann['file_name']
            overall_pred_meta['annotations'].append(pan_pd_ann)

            metric.update(
                pan_gt_ann, rgb2id(np.array(pan_gt_mask)), pan_pd_ann, pan_pd_mask
            )

            if save_output:
                # pan_pd_mask = Image.fromarray(id2rgb(pan_pd_mask))
                # fname = osp.join(pred_mask_dir, pan_pd_ann['file_name'])
                # os.makedirs(osp.dirname(fname), exist_ok=True)  # make region subdir
                # pan_pd_mask.save(fname)
                fname = osp.join(pred_mask_dir, pan_pd_ann['file_name'])
                os.makedirs(osp.dirname(fname), exist_ok=True)  # make region subdir
                os.makedirs(osp.dirname(fname.replace(pred_mask_dir, pred_mask_dir+'_d2')), exist_ok=True)  # make region subdir
                # os.makedirs(osp.dirname(fname.replace(pred_mask_dir, pred_mask_dir+'_d2_ups')), exist_ok=True)  # make region subdir

                from .vis import d2_vis
                im = dset.pan_getitem(i, apply_trans=False)[2]
                if cfg.data.dataset.params['caffe_mode']:
                    im = np.array(im)[:, :, ::-1]
                im = np.array(im)
                Image.fromarray(
                    d2_vis(
                        dset.meta,
                        pan_pd_mask,
                        pan_pd_ann,
                        im
                    )
                ).save(fname.replace(pred_mask_dir, pred_mask_dir+'_d2'))

                from .vis import d2_vis
                im = dset.pan_getitem(i, apply_trans=False)[2]
                if cfg.data.dataset.params['caffe_mode']:
                    im = np.array(im)[:, :, ::-1]
                im = np.array(im)
                tmp_ann = upsnet_pred_ann[pan_gt_ann['image_id']]
                for seg in tmp_ann['segments_info']:
                    if seg['category_id'] in ups_thing_cat_ids:
                        seg['isthing'] = 1
                    else:
                        seg['isthing'] = 0
                Image.fromarray(
                    d2_vis(
                        dset.meta,
                        load_pred_mask(pan_gt_ann['file_name']),
                        tmp_ann,
                        im
                    )
                ).save(fname.replace(pred_mask_dir, pred_mask_dir+'_d2').replace('.png', '_ups.png'))

                pan_pd_mask = Image.fromarray(id2rgb(pan_pd_mask))
                pan_pd_mask.save(fname)

            # torch.cuda.synchronize()
            timer.compute.toc()

            if (i % 50) == 0:
                logger.info(timer)
                logger.info(
                    "iter {}, eta to val end {}".format(i, timer.eta(i, len(loader)))
                )
                print(metric)
            timer.data.tic()

        if save_output:
            with open(pred_meta_fname, 'w') as f:
                json.dump(overall_pred_meta, f)
            self.output_mngr.save_f(
                metric.results, osp.join(self.output_mngr.root, 'score.pkl')
            )
        logger.info("eval complete \n{}".format(metric))
        return metric

    def report(self):
        model = self.model
        loader = self.val_loader
        dset = loader.dataset
        report_at = model.curr_epoch - 1
        self.leader_log("reporting at epoch {}".format(report_at))
        self.leader_log("eval dset size: {}".format(len(dset)))
        self.leader_log("eval num iterations: {}".format(len(loader)))

        output_root = self.output_mngr.root
        reporter_cfgs = cfg.z_report

        reporters = [
            reporter_modules[r_cfg['name']](
                r_cfg['params'], model, dset,
                output_root=osp.join(output_root, 'report_{}'.format(i))
            )  # each report occupies output/report_i/
            for i, r_cfg in enumerate(reporter_cfgs)
        ]
        cudnn_old_state = torch.backends.cudnn.benchmark
        torch.backends.cudnn.benchmark = False  # input size could vary

        timer = CompoundTimer()
        timer.data.tic()
        for i, inputs in enumerate(loader):
            if self.debug and (i > 5):
                break
            x = inputs[0].cuda()
            timer.data.toc()
            timer.compute.tic()
            pred = model.infer(x)
            for actor in reporters:
                actor.process(i, pred, inputs, dset)
                if (i % 50) == 0:
                    logger.info(
                        "iter {}, eta to val end {}".format(i, timer.eta(i, len(loader)))
                    )
                    print(actor.metric)
            timer.compute.toc()

            if (i % 10) == 0:
                self.leader_log(timer)
                self.leader_log(
                    "iter {}, eta to val end {}".format(i, timer.eta(i, len(loader)))
                )
            timer.data.tic()
        self.leader_log("reporting complete\n")
        torch.backends.cudnn.benchmark = cudnn_old_state
        reports = [ actor.generate_report() for actor in reporters ]
        for i, r in enumerate(reports):
            print('report {} ----- '.format(i))
            pprint(reporter_cfgs[i])
            print(r)
        return reports

    def PQ_test(self, save_output=False):
        assert not self.mp_distributed
        model = self.model
        loader = self.test_loader
        dset = loader.dataset
        eval_at = model.curr_epoch - 1
        logger.info("Panoptic Quality eval at epoch {}".format(eval_at))

        # PRED_OUT_NAME = 'panoptic_{}2017_pcv_results'.format(dset.split)
        PRED_OUT_NAME = 'pred'
        dump_root = osp.join(self.output_mngr.root, 'pd_dump')
        pred_meta_fname = osp.join(dump_root, '{}.json'.format(PRED_OUT_NAME))
        pred_mask_dir = osp.join(dump_root, PRED_OUT_NAME)

        overall_pred_meta = {
            'images': list(dset.imgs.values()),
            'categories': list(dset.meta['cats'].values()),
            'annotations': []
        }

        timer = CompoundTimer()
        timer.data.tic()

        for i, inputs in enumerate(loader):
            if self.debug and (i > 5):
                break
            x = inputs[0].cuda()
            del inputs
            imgMeta, _ = dset.get_meta(i)

            timer.data.toc()
            timer.compute.tic()

            sem_pd, vote_pd = model.infer(x, softmax_normalize=True)
            # if hasattr(self, 'model2'):
            #     _sem_pd, _vote_pd = self.model2.infer(x, softmax_normalize=True)
            #     sem_pd = (sem_pd + _sem_pd) / 2
            #     vote_pd = (vote_pd + _vote_pd) / 2
            img_original_size = (imgMeta['width'], imgMeta['height'])
            pan_pd_mask, pan_pd_ann = model.stitch_pan_mask(
                cfg.pcv, sem_pd, vote_pd, img_original_size
            )
            pan_pd_ann['image_id'] = imgMeta['id']
            pan_pd_ann['file_name'] = imgMeta['file_name'].split('.')[0] + '.png'
            overall_pred_meta['annotations'].append(pan_pd_ann)

            if save_output:
                fname = osp.join(pred_mask_dir, pan_pd_ann['file_name'])
                os.makedirs(osp.dirname(fname), exist_ok=True)  # make region subdir
                os.makedirs(osp.dirname(fname.replace(pred_mask_dir, pred_mask_dir+'_d2')), exist_ok=True)  # make region subdir

                from .vis import d2_vis
                im = dset.pan_getitem(i, apply_trans=False)[2]
                if cfg.data.dataset.params['caffe_mode']:
                    im = np.array(im)[:, :, ::-1]
                im = np.array(im)
                Image.fromarray(
                    d2_vis(
                        dset.meta,
                        pan_pd_mask,
                        pan_pd_ann,
                        im
                    )
                ).save(fname.replace(pred_mask_dir, pred_mask_dir+'_d2'))

                pan_pd_mask = Image.fromarray(id2rgb(pan_pd_mask))
                pan_pd_mask.save(fname)

            # torch.cuda.synchronize()
            timer.compute.toc()

            if (i % 50) == 0:
                logger.info(timer)
                logger.info(
                    "iter {}/{}, eta to val end {}".format(
                        i, len(loader), timer.eta(i, len(loader))
                    )
                )
            timer.data.tic()

        if save_output:
            with open(pred_meta_fname, 'w') as f:
                json.dump(overall_pred_meta, f)
        logger.info("test eval complete")

    def make_figures(self):
        # fmt: off
        split =                  cfg.figures.split
        figure_scale =           cfg.figures.figure_scale
        confined_set =           cfg.figures.chosen_imgs
        # fmt: on

        assert not self.mp_distributed
        model = self.model
        loader = self.val_loader if split == 'val' else self.test_loader
        dset = loader.dataset

        # confine dset loading to the chosen subset of images
        if dset.name == 'coco':
            confined_set = [int(elem) for elem in confined_set]
        dset.confine_to_subset(confined_set)

        eval_at = model.curr_epoch - 1
        logger.info("Making figure; loaded model at epoch {}".format(eval_at))

        dump_root = osp.join(self.output_mngr.root, split)
        os.makedirs(dump_root, exist_ok=True)

        # if dset.name == 'coco' and split == 'val':
        #     upsnet_pred_json = json.load(open('/home-nfs/whc/panout/upsnet/coco/val/pred.json', 'r'))
        #     upsnet_pred_ann = {_['image_id']: _ for _ in upsnet_pred_json['annotations']}
        #     ups_thing_cat_ids = [_['id'] for _ in upsnet_pred_json['categories'] if _['isthing']]
        #     ups_stuff_cat_ids = [_['id'] for _ in upsnet_pred_json['categories'] if _['isthing'] == 0]

        #     def load_ups_mask(filename):
        #         return rgb2id(np.array(
        #                 Image.open(f'/home-nfs/whc/panout/upsnet/coco/val/pred/{filename}').convert('RGB')
        #         ))

        for i, inputs in enumerate(loader):
            if self.debug and (i > 5):
                break
            x = inputs[0].cuda()
            del inputs
            imgMeta, _ = dset.get_meta(i)
            sem_pd, vote_pd = model.infer(x, softmax_normalize=True)
            img_original_size = (imgMeta['width'], imgMeta['height'])
            pan_pd_mask, pan_pd_ann, vote_hmap = model.stitch_pan_mask(
                cfg.pcv, sem_pd, vote_pd, img_original_size, return_hmap=True
            )

            # save visualizations
            # basename is used since cs fname is city/id_left*.png;
            # don't want the city as an extra directory layer in this case
            fname = osp.basename(imgMeta['file_name']).split('.')[0]
            save_path = osp.join(dump_root, fname)
            im = dset.pan_getitem(i, apply_trans=False)[2]
            if cfg.data.dataset.params['caffe_mode']:
                im = np.array(im)[:, :, ::-1]
            im = np.array(im)
            print(im.shape)
            print(vote_hmap.shape)
            Image.fromarray(
                d2_vis(dset.meta, pan_pd_mask, pan_pd_ann, im, scale=figure_scale)
            ).save('{}_vis.png'.format(save_path))
            np.save('{}_hmap.npy'.format(save_path), vote_hmap)
            Image.fromarray(im).save('{}_img.png'.format(save_path))

            # if dset.name == 'coco' and split == 'val':
            #     tmp_ann = upsnet_pred_ann[imgMeta['id']]
            #     for seg in tmp_ann['segments_info']:
            #         if seg['category_id'] in ups_thing_cat_ids:
            #             seg['isthing'] = 1
            #         else:
            #             seg['isthing'] = 0
            #     ups_pred_mask = load_ups_mask(fname)
            #     Image.fromarray(
            #         d2_vis(dset.meta, ups_pred_mask, tmp_ann, im, scale=figure_scale)
            #     ).save(save_path.replace('.png', '_ups.png'))

    def vis(self, oracle=False, oracle_res=4):
        """
        optionally save all predicted files through output writer
        """
        assert not self.mp_distributed
        from panoptic.vis import Visualizer
        from ipywidgets import interactive, BoundedIntText, BoundedFloatText
        from IPython.display import display

        model, loader = self.model, self.val_loader
        dset, pcv = loader.dataset, model.pcv
        vis_engine = Visualizer(cfg, dset.meta, pcv)
        vis_engine.display_stdout_and_err_in_curr_cell()

        def logic(inx, hmap_thresh):
            _, segments_info, img, pan_gt_mask = dset.pan_getitem(inx)
            if oracle:
                sem_pd, vote_pd = gt_tsr_res_reduction(
                    oracle_res, self.gt_prod_handle,
                    dset.meta, pcv, pan_gt_mask, segments_info
                )
            else:
                # sem_pd, _ = gt_tsr_res_reduction(
                #     oracle_res, self.gt_prod_handle,
                #     dset.meta, pcv, pan_gt_mask, segments_info
                # )
                sem_pd, vote_pd = model.infer(
                    dset[inx][0].unsqueeze(0).cuda(), softmax_normalize=True
                )

            # img = _downsample_PIL(img)  # rtchange
            if cfg.data.dataset.params['caffe_mode']:
                img = np.array(img)[:, :, ::-1]
            pan_gt_mask = _downsample_PIL(pan_gt_mask)
            vis_engine.vis(
                img, pan_gt_mask, segments_info, sem_pd, vote_pd,
                self.gt_prod_handle, model.criteria, hmap_thresh
            )

        wdgt = interactive(
            logic,
            inx=BoundedIntText(
                min=0, max=len(dset) - 1, step=1
            ),
            hmap_thresh=BoundedFloatText(
                value=cfg.pcv.hmap_thresh, min=0, max=1000.0, step=0.2,
                description='ws_thresh'
            )
        )
        wdgt.children[-1].layout.height = '1300px'
        display(wdgt)
        return vis_engine


def _downsample_PIL(im, ratio=4):
    """round up division"""
    def ceil_div(x, div):
        return (x + div - 1) // div
    w, h = im.size
    w, h = ceil_div(w, ratio), ceil_div(h, ratio)
    im = im.resize((w, h), resample=Image.NEAREST)
    return im


def gt_tsr_res_reduction(
    resolution, gen_handle, dset_meta, pcv, pan_gt_mask, segments_info
):
    pan_gt_mask = _downsample_PIL(pan_gt_mask, resolution)
    generator = gen_handle(dset_meta, pcv, pan_gt_mask, segments_info)
    generator.generate_gt()
    sem_pb, vote_pb = generator.collect_prob_tsr()
    sem_pd = torch.as_tensor(sem_pb).float().cuda()
    vote_pd = torch.as_tensor(vote_pb).float().cuda()
    return sem_pd, vote_pd


class SOTASSeg(torch.utils.data.Dataset):
    def __init__(self, dset):
        from .test.test_ups_sseg import UPSSSeg
        self.sseg = UPSSSeg(dset)

    def __len__(self):
        return len(self.sseg)

    def get(self, index, target_size):
        '''
        Args:
            target_size: (H, W)
        '''
        from .datasets.base import tensorize_2d_spatial_assignement
        mask = self.sseg[index]
        # to (W, H)
        mask = np.array(
            Image.fromarray(mask).resize(
                target_size[::-1], resample=Image.NEAREST
            )
        )
        assert mask.shape == target_size
        tsr = tensorize_2d_spatial_assignement(mask, self.sseg.num_classes)
        tsr = torch.as_tensor(tsr).float().cuda()
        # print(tsr.shape)
        return tsr

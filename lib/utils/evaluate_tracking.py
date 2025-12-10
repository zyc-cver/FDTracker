"""
evaluate tracking h+o methods
"""

import sys, os
import time
import pickle
import torch

import numpy as np
import trimesh
from tqdm import tqdm
sys.path.append('my_code/eval_lib')
sys.path.append('my_code/eval_code')
import utils.metrics as metric
from utils.evaluate_joint import JointReconEvaluator
from core.config import cfg
class JointTrackEvaluator(JointReconEvaluator):
    def eval(self, res_dir, gt_dir, metadata, video_list=None, outfile=None):
        """

        :param res_dir:
        :param gt_dir:
        :param outfile:
        :return:
        """
        data_gt, data_pred = gt_dir, res_dir
        data_complete = self.check_data(data_gt, data_pred)
        # data_complete = True
        if data_complete:
            start = time.time()
            if video_list is None:
                error_dict, videos_error = self.compute_errors(data_gt, data_pred, metadata)
                videos_error = None
            else:
                error_dict, videos_error = self.compute_errors(data_gt, data_pred, metadata, video_list)
            end = time.time()
            self.logging(f'Evaluation done after {end-start:.4f} seconds')
        else:
            error_dict = {}
            for dname in ['behave', 'icap']:
                ed_i = {f'{k}_{dname}':np.inf for k in ["SMPL", 'obj']}
                error_dict.update(**ed_i)
        if outfile is not None:
            self.write_errors(outfile, error_dict)
        return error_dict, videos_error

    def check_data(self, dict_gt:dict, dict_pred:dict):
        if len(dict_pred.keys()) != len(dict_gt.keys()):
            self.logging("Warning: Not enough sequences are reconstructed for evaluation!")
            print("recon:", len(dict_pred.keys()), "GT:", len(dict_gt.keys()))
            # return False
            return True
        else:
            self.logging(f"Start evaluating {len(dict_pred.keys())} sequences.")
            return True

    def compute_errors(self, data_gt, data_pred, metadata, videos=None):
        """

        :param data_gt:
        :param data_pred:
        :param videos: list of video names to compute individual results for
        :return:
        """
        errors_all, videos_err = {}, {}
        # Initialize video-specific error tracking if videos list is provided
        if videos is None:
            # default videos
            videos = list(data_pred.keys())
        for vid in videos:
            videos_err[vid] = {}
        # manager = mp.Manager()
        # errors_all = manager.dict()
        jobs = []
        for k in tqdm(sorted(data_pred.keys())):
            if k not in data_gt:
                self.logging(f'sequence id {k} not found in GT data!')
                continue
            self.eval_seq(data_gt, data_pred, errors_all, k, metadata, videos_err, videos)
        #     p = mp.Process(target=self.eval_seq, args=(data_gt, data_pred, errors_all, k))
        #     p.start()
        #     jobs.append(p)
        # for job in jobs:
        #     job.join()
            # self.eval_seq(data_gt, data_pred, errors_all, k)

        errors_avg = {k: np.mean(v) for k, v in errors_all.items()}
        
        # # Calculate final averages for each specified video
        # if videos is not None:
        #     videos_err = {}
        #     for vid in videos:
        #         if vid in videos_err and videos_err[vid]:
        #             videos_err[vid] = {}
        #             for metric_name, values in videos_err[vid].items():
        #                 videos_error[vid][metric_name] = np.mean(values) if len(values) > 0 else 0.0
        #     return errors_avg, videos_error
        # else:
        return errors_avg, videos_err

    def compute_accel_error(self, th_gt, th_pr):
        """
        compute acceleration error for translation
        return: (N, )
        """
        accel_gt = th_gt[:-2] - 2 * th_gt[1:-1] + th_gt[2:]
        accel_pr = th_pr[:-2] - 2 * th_pr[1:-1] + th_pr[2:]
        err = np.sqrt(np.sum((accel_gt - accel_pr) ** 2, -1)) * self.m2mm
        return err

    def eval_seq(self, data_gt, data_pred, errors_all, k, metadata, videos_err, videos=None):
        """
        evaluate one sequence specified by k
        :param data_gt:
        :param data_pred:
        :param errors_all: the result dictionary
        :param k:
        :param videos_err: video-specific errors dictionary
        :param videos: list of videos to track individually
        :return:
        """
        start = time.time()
        # self.logging(f'start evaluating {k}...')
        if cfg.DATASET.name == 'BEHAVE':
            dname = 'behave'
        elif cfg.DATASET.name == 'InterCap':
            dname = 'icap'
        gender, obj_name = metadata[k]["gender"], metadata[k]["obj_name"]
        template_data = pickle.load(open(cfg.OBJ.template_path, 'rb'))['templates']
        temp_faces = template_data[obj_name]['faces']
        temp_verts = template_data[obj_name]['verts']

        # compute acceleration error of human and object
        th_gt, to_gt = data_gt[k]['smplh_trans'].numpy(), data_gt[k]['obj_trans'].numpy() # GT human obj translation
        th_pr, to_pr = data_pred[k]['trans'], data_pred[k]['obj_trans']
        if len(th_gt) != len(th_pr):
            self.logging(f'The number of object predictions does not match GT! {len(th_gt)}!={len(th_pr)}')
            exit(-1)
        if len(to_gt) != len(to_pr):
            self.logging(f'The number of human predictions does not match GT! {len(to_gt)}!={len(to_pr)}')
            exit(-1)
        err_h = np.mean(self.compute_accel_error(th_gt, th_pr))
        err_o = np.mean(self.compute_accel_error(to_gt, to_pr))
        
        # Store acceleration errors for overall statistics
        if f'acc-h_{dname}' not in errors_all:
            errors_all[f'acc-h_{dname}'] = []
        if f'acc-o_{dname}' not in errors_all:
            errors_all[f'acc-o_{dname}'] = []
        errors_all[f'acc-h_{dname}'].append(err_h)
        errors_all[f'acc-o_{dname}'].append(err_o)
        
        # Store acceleration errors for specific videos if requested
        if videos is not None and k in videos:
            videos_err[k][f'acc-h_{dname}'] = err_h
            videos_err[k][f'acc-o_{dname}'] = err_o

        freq = 10 # instead of evaluate all frames which is expensive, only evaluate over some key frames

        # compute Object
        rot_pr, trans_pr = data_pred[k]['obj_rot'][::freq], data_pred[k]['obj_trans'][::freq]  # (T, 3, 3) and (T, 3)
        ov_pr = np.matmul(temp_verts[None].repeat(len(rot_pr), 0), rot_pr.transpose(0, 2, 1)) + trans_pr[:, None]
        # pts_pr = np.matmul(temp_samples[None].repeat(len(rot_pr), 0), rot_pr.transpose(0, 2, 1)) + trans_pr[:, None]
        ov_gt = np.matmul(temp_verts[None].repeat(len(rot_pr), 0),
                          data_gt[k]['obj_rot'][::freq].numpy().transpose(0, 2, 1)) + \
                data_gt[k]['obj_trans'][::freq][:, None].numpy()
        # pts_gt = np.matmul(temp_samples[None].repeat(len(rot_pr), 0), data_gt[k]['obj_rot'].transpose(0, 2, 1)) + \
        #             data_gt[k]['obj_trans'][:, None]
        if len(ov_pr) != len(ov_gt):
            self.logging(f'the number of object vertices does not match! {len(ov_pr)}!={len(ov_gt)}, seq id={k}')
            exit(-1)
            # return
        # compute SMPL
        pose_pred = data_pred[k]['pose']
        if pose_pred.shape[-1] == 72:
            pose_pred = self.smpl2smplh_pose(pose_pred)
        elif pose_pred.shape[-1] == 66:
            temp = np.zeros((pose_pred.shape[0], 156))
            temp[:, :66] = pose_pred
            pose_pred = temp
        model = self.smplh_male if gender == 'male' else self.smplh_female
        sv_pr = model.update(pose_pred[::freq], data_pred[k]['betas'][::freq], data_pred[k]['trans'][::freq])[0]
        sv_gt = model.update(data_gt[k]['smplh_poses'][::freq].numpy(), data_gt[k]['smplh_betas'][::freq].numpy(),
                             data_gt[k]['smplh_trans'][::freq].numpy())[0]
        ee = time.time()
        # print('time to finish SMPL forward:', ee-start) # this is the most time consuming part, it can take up to 6s to finish 1500 frames
        if len(sv_pr) != len(sv_gt):
            self.logging(f'the number of SMPL vertices does not match! {len(sv_pr)}!={len(sv_gt)}, seq id={k}')
            exit(-1)
            # return
        # classify based on dataset
        if f'SMPL_{dname}' not in errors_all:
            errors_all[f'SMPL_{dname}'] = []
        if f'obj_{dname}' not in errors_all:
            errors_all[f'obj_{dname}'] = []
            
        # Initialize temporary lists for video-specific errors if this video is in the target list
        video_smpl_errors = []
        video_obj_errors = []

        L = len(sv_pr)
        time_window = 300//freq
        arot, atrans, ascale = None, None, None  # global alignment
        for i in range(0, L, 1): # cannot evaluate all frames since codalab server is too slow: this is not the bootleneck
            if arot is None or i % time_window == 0:
                # combine all vertices in this window and align
                bend = min(L, i + time_window)
                indices = np.arange(i, bend)
                # print(sv_gt.shape, ov_gt.shape, sv_pr.shape, ov_pr.shape)
                verts_clip_gt = np.concatenate([np.concatenate(x[indices], 0) for x in [sv_gt, ov_gt]], 0)
                verts_clip_pr = np.concatenate([np.concatenate(x[indices], 0) for x in [sv_pr, ov_pr]], 0)
                ss = time.time()
                _, arot, atrans, ascale = metric.compute_similarity_transform(verts_clip_pr, verts_clip_gt)
                ee = time.time()
                # print('time to align:',ee -ss) # around 0.137s to align

            # align
            ov_pr_i = (ascale * arot.dot(ov_pr[i].T) + atrans).T
            # ov_pr_i = (ascale*arot.dot(pts_pr[i].T) + atrans).T
            sv_pr_i = (ascale * arot.dot(sv_pr[i].T) + atrans).T
            # compute errors
            # err_obj, err_smpl = self.compute_joint_errors(ov_gt[i], ov_pr_i, sv_gt[i], sv_pr_i, temp_faces, model.faces)

            # compute v2v instead of cd: cd is too expensive to compute
            err_obj = np.mean(np.sqrt(np.sum((ov_pr_i - ov_gt[i]) ** 2, -1)))
            err_smpl = np.mean(np.sqrt(np.sum((sv_pr_i - sv_gt[i]) ** 2, -1)))

            errors_all[f'obj_{dname}'].append(err_obj * self.m2mm)
            errors_all[f'SMPL_{dname}'].append(err_smpl * self.m2mm) # this continuous access slows down multi process a lot!
            
            # Collect errors for specific videos if requested
            if videos is not None and k in videos:
                video_obj_errors.append(err_obj * self.m2mm)
                video_smpl_errors.append(err_smpl * self.m2mm)

        # Calculate and store average errors for this video
        if videos is not None and k in videos:
            videos_err[k][f'obj_{dname}'] = np.mean(video_obj_errors) if len(video_obj_errors) > 0 else 0.0
            videos_err[k][f'SMPL_{dname}'] = np.mean(video_smpl_errors) if len(video_smpl_errors) > 0 else 0.0

        end = time.time()
        # print(f'seq {k} ({len(th_gt)} frames) done after {end-start:.4f} seconds')
def evaluate_track(pred_data, gt_data, metadata, videos=None):
    # pred_data is a dict of tensors, transfer to numpy
    evaluator = JointTrackEvaluator('data/base_data/human_models/mano_v1_2/models')
    if videos is None:
        error_dict = evaluator.eval(pred_data, gt_data, metadata)
        return error_dict
    else:
        error_dict, videos_error = evaluator.eval(pred_data, gt_data, metadata, videos)
        return error_dict, videos_error

def create_renamed_data(pred_data, gt_data):
    # Create a new dictionary with the same keys as pred_data and values from gt_data
    renamed_data = {}
    for key in pred_data.keys():
        if key in gt_data:
            renamed_data[key] = {}
            renamed_data[key]['pose'] = gt_data[key]['smplh_poses'].numpy()# pred_data[key]['pose']#
            renamed_data[key]['betas'] = pred_data[key]['betas']#gt_data[key]['smplh_betas'].numpy()#
            renamed_data[key]['trans'] = pred_data[key]['trans']#gt_data[key]['smplh_trans'].numpy()
            renamed_data[key]['obj_trans'] = pred_data[key]['obj_trans'] # gt_data[key]['obj_trans'].numpy()
            renamed_data[key]['obj_rot'] = gt_data[key]['obj_rot'].numpy()# pred_data[key]['obj_rot']#
        else:
            print(f"Key {key} not found in gt_data.")
    return renamed_data

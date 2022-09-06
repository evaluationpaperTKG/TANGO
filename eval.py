import time
from utils import *
import numpy as np #added eval_paper_authors

def push_data(*args, device=None):
    out_args = []
    for arg in args:
        arg = [_arg.to(device) for _arg in arg]
        out_args.append(arg)
    return out_args


def push_data2(*args, device=None):
    out_args = []
    for arg in args:
        arg = arg.to(device)
        out_args.append(arg)
    return out_args


def predict(loader, model, params, num_e, test_adjmtx, logger, log_scores_flag=False, ts_max=0, setting='time'): #modified version from eval_paper_authors - to include the multistep prediction, add setting for logging
    print("prediction setting", setting)
    model.eval()
    p = params
    rank_group_num = 2000
    
    #modified eval_paper_authors logging
    if log_scores_flag:
        eval_paper_authors_logging_dict = {} #added eval_paper_authors
        import inspect
        import sys
        import os
        currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        # parentdir = os.path.dirname(currentdir)
        sys.path.insert(1, currentdir) 
        sys.path.insert(1, os.path.join(sys.path[0], '../../..'))        
        sys.path.insert(1, os.path.join(sys.path[0], '..'))    
        import evaluation_utils 
        dataset_dir = os.path.join(params.dataset) #modified eval_paper_authors
        num_nodes, num_rels = evaluation_utils.get_total_number(dataset_dir, 'stat.txt')
    # end modified eval_paper_authors logging

    with torch.no_grad():
        results = {}
        iter = loader

        print("Start evaluation")
        t1 = time.time()

        for step, (
        sub_in, rel_in, obj_in, lab_in, sub_tar, rel_tar, obj_tar, lab_tar, tar_ts, in_ts, edge_idlist, edge_typelist,
        indep_lab, adj_mtx, edge_jump_w, edge_jump_id, rel_jump) in enumerate(iter):
            if len(sub_tar) == 0:
                continue
            for step_idx in range(len(sub_tar)): # mod eval_paper_authors: loop across all timesteps
                # forward
                # emb = model.forward_eval(in_ts, tar_ts, edge_idlist, edge_typelist, edge_jump_id, edge_jump_w, rel_jump) 
                emb = model.forward_eval(in_ts, [tar_ts[step_idx]], edge_idlist, edge_typelist, edge_jump_id, edge_jump_w, rel_jump) # mod eval_paper_authors: and compute embdedding predition for timestep of interest tar_ts[step_idx]

                rank_count = 0
                while rank_count < sub_tar[step_idx].shape[0]:
                    l, r = rank_count, (rank_count + rank_group_num) if (rank_count + rank_group_num) <= sub_tar[step_idx].shape[
                        0] else sub_tar[step_idx].shape[0]

                    # push data onto gpu
                    [sub_tar_, rel_tar_, obj_tar_, lab_tar_, indep_lab_] = \
                        push_data2(sub_tar[step_idx][l:r], rel_tar[step_idx][l:r], obj_tar[step_idx][l:r], lab_tar[step_idx][l:r, :],
                                indep_lab[step_idx][l:r, :], device=p.device)

                    # compute scores for corresponding triples
                    score = model.score_comp(sub_tar_, rel_tar_, emb, model.odeblock.odefunc) #scores for all entities in this timestep (tar_ts) for all triples (sub and ob directions)
                    b_range = torch.arange(score.shape[0], device=p.device) 

                    # added eval_paper_authors:
                    if log_scores_flag:
                        for triplesindex in range(sub_tar_.shape[0]):
                            tar_ts_unscaled =  rescale_timesteps(tar_ts[step_idx].cpu().detach().numpy(), ts_max, params.scale)
                            test_query = [int(sub_tar_[triplesindex].cpu().detach().numpy()), int(rel_tar_[triplesindex].cpu().detach().numpy()), int(obj_tar_[triplesindex].cpu().detach().numpy()), tar_ts_unscaled ] #added eval_paper_authors
                            query_name, gt_test_query_ids = evaluation_utils.query_name_from_quadruple(test_query, num_rels) #added eval_paper_authors
                            eval_paper_authors_logging_dict[query_name] = [score[triplesindex].cpu().detach().numpy(), gt_test_query_ids]# added eval_paper_authors - l
                    # end added eval_paper_authors

                    # raw ranking
                    ranks = 1 + torch.argsort(torch.argsort(score, dim=1, descending=True), dim=1, descending=False)[
                        b_range, obj_tar_]

                    ranks = ranks.float()
                    results['count_raw'] = torch.numel(ranks) + results.get('count_raw', 0.0)
                    results['mar_raw'] = torch.sum(ranks).item() + results.get('mar_raw', 0.0)
                    results['mrr_raw'] = torch.sum(1.0 / ranks).item() + results.get('mrr_raw', 0.0)
                    for k in range(10):
                        results['hits@{}_raw'.format(k + 1)] = torch.numel(ranks[ranks <= (k + 1)]) + results.get(
                            'hits@{}_raw'.format(k + 1), 0.0)

                    # time aware filtering
                    target_score = score[b_range, obj_tar_]
                    score = torch.where(lab_tar_.byte(), -torch.ones_like(score) * 10000000, score)
                    score[b_range, obj_tar_] = target_score

                    # time aware filtered ranking
                    ranks = 1 + torch.argsort(torch.argsort(score, dim=1, descending=True), dim=1, descending=False)[
                        b_range, obj_tar_]
                    ranks = ranks.float()
                    results['count'] = torch.numel(ranks) + results.get('count', 0.0)
                    results['mar'] = torch.sum(ranks).item() + results.get('mar', 0.0)
                    results['mrr'] = torch.sum(1.0 / ranks).item() + results.get('mrr', 0.0)
                    for k in range(10):
                        results['hits@{}'.format(k + 1)] = torch.numel(ranks[ranks <= (k + 1)]) + results.get(
                            'hits@{}'.format(k + 1), 0.0)

                    # time unaware filtering
                    score = torch.where(indep_lab_.byte(), -torch.ones_like(score) * 10000000, score)
                    score[b_range, obj_tar_] = target_score

                    # time unaware filtered ranking
                    ranks = 1 + torch.argsort(torch.argsort(score, dim=1, descending=True), dim=1, descending=False)[
                        b_range, obj_tar_]
                    ranks = ranks.float()
                    results['count_ind'] = torch.numel(ranks) + results.get('count_ind', 0.0)
                    results['mar_ind'] = torch.sum(ranks).item() + results.get('mar_ind', 0.0)
                    results['mrr_ind'] = torch.sum(1.0 / ranks).item() + results.get('mrr_ind', 0.0)
                    for k in range(10):
                        results['hits@{}_ind'.format(k + 1)] = torch.numel(ranks[ranks <= (k + 1)]) + results.get(
                            'hits@{}_ind'.format(k + 1), 0.0)

                    rank_count += rank_group_num
            del sub_tar_, rel_tar_, obj_tar_, lab_tar_, indep_lab_
        
        results['mar'] = round(results['mar'] / results['count'], 5)
        results['mrr'] = round(results['mrr'] / results['count'], 5)
        results['mar_raw'] = round(results['mar_raw'] / results['count_raw'], 5)
        results['mrr_raw'] = round(results['mrr_raw'] / results['count_raw'], 5)
        results['mar_ind'] = round(results['mar_ind'] / results['count_ind'], 5)
        results['mrr_ind'] = round(results['mrr_ind'] / results['count_ind'], 5)
        for k in range(10):
            results['hits@{}'.format(k + 1)] = round(results['hits@{}'.format(k + 1)] / results['count'], 5)
            results['hits@{}_raw'.format(k + 1)] = round(results['hits@{}_raw'.format(k + 1)] / results['count_raw'], 5)
            results['hits@{}_ind'.format(k + 1)] = round(results['hits@{}_ind'.format(k + 1)] / results['count_ind'], 5)

        t2 = time.time()
        print("evaluation time: ", t2 - t1)
        logger.info("evaluation time: {}".format(t2 - t1))

        #eval_paper_authors
        if log_scores_flag:
            import pathlib
            import pickle
            dirname = os.path.join(pathlib.Path().resolve(), 'results' )
            if (len(sub_tar)) > 1:
                steps = 'multistep'
            else:
                steps = 'singlestep'
            logname = 'tango' + '-' + params.dataset + '-' + steps + '-' + setting
            eval_paper_authorsfilename = os.path.join(dirname, logname + ".pkl")
            # if not os.path.isfile(eval_paper_authorsfilename):
            with open(eval_paper_authorsfilename,'wb') as file:
                pickle.dump(eval_paper_authors_logging_dict, file, protocol=4) 
            file.close()
        #END eval_paper_authors

    return results


def rescale_timesteps(timesteps, ts_max, scale): #eval_paper_authors
    # for logging
    # utils.y setup_tkG -> unscale
    #   timestamps
        # normalize timestamps, and scale
        #ts_max = max(max(timestamp['train']), max(timestamp['test'])) # max timestamp in the dataset
        #train_timestamps = (torch.tensor(timestamp['train']) / torch.tensor(ts_max, dtype=torch.float)) * scale
    original_train_timestamps = np.round(timesteps*ts_max/scale).astype(int) #round bec otherwise problems with rescale
    return original_train_timestamps

def odl_predict2(loader, model, params, num_e, test_adjmtx, logger, log_scores_flag=False, ts_max=0, setting='time'): #original version before modification with modified aargs
    model.eval()
    p = params
    rank_group_num = 2000

    with torch.no_grad():
        results = {}
        iter = loader

        print("Start evaluation")
        t1 = time.time()

        for step, (
        sub_in, rel_in, obj_in, lab_in, sub_tar, rel_tar, obj_tar, lab_tar, tar_ts, in_ts, edge_idlist, edge_typelist,
        indep_lab, adj_mtx, edge_jump_w, edge_jump_id, rel_jump) in enumerate(iter):
            if len(sub_tar) == 0:
                continue

            # forward
            emb = model.forward_eval(in_ts, tar_ts, edge_idlist, edge_typelist, edge_jump_id, edge_jump_w, rel_jump) # comment eval_paper_authors: this only returns embeddings for 1 timestep

            rank_count = 0
            while rank_count < sub_tar[0].shape[0]: # comment eval_paper_authors: no matter how many steps ahead with tar_ts, this will always look at element [0]
                l, r = rank_count, (rank_count + rank_group_num) if (rank_count + rank_group_num) <= sub_tar[0].shape[
                    0] else sub_tar[0].shape[0]

                # push data onto gpu
                [sub_tar_, rel_tar_, obj_tar_, lab_tar_, indep_lab_] = \
                    push_data2(sub_tar[0][l:r], rel_tar[0][l:r], obj_tar[0][l:r], lab_tar[0][l:r, :],
                               indep_lab[0][l:r, :], device=p.device)

                # compute scores for corresponding triples
                score = model.score_comp(sub_tar_, rel_tar_, emb, model.odeblock.odefunc) #scores for all entities in this timestep (tar_ts) for all triples (sub and ob directions)
                b_range = torch.arange(score.shape[0], device=p.device) 

                # raw ranking
                ranks = 1 + torch.argsort(torch.argsort(score, dim=1, descending=True), dim=1, descending=False)[
                    b_range, obj_tar_]

                ranks = ranks.float()
                results['count_raw'] = torch.numel(ranks) + results.get('count_raw', 0.0)
                results['mar_raw'] = torch.sum(ranks).item() + results.get('mar_raw', 0.0)
                results['mrr_raw'] = torch.sum(1.0 / ranks).item() + results.get('mrr_raw', 0.0)
                for k in range(10):
                    results['hits@{}_raw'.format(k + 1)] = torch.numel(ranks[ranks <= (k + 1)]) + results.get(
                        'hits@{}_raw'.format(k + 1), 0.0)

                # time aware filtering
                target_score = score[b_range, obj_tar_]
                score = torch.where(lab_tar_.byte(), -torch.ones_like(score) * 10000000, score)
                score[b_range, obj_tar_] = target_score

                # time aware filtered ranking
                ranks = 1 + torch.argsort(torch.argsort(score, dim=1, descending=True), dim=1, descending=False)[
                    b_range, obj_tar_]
                ranks = ranks.float()
                results['count'] = torch.numel(ranks) + results.get('count', 0.0)
                results['mar'] = torch.sum(ranks).item() + results.get('mar', 0.0)
                results['mrr'] = torch.sum(1.0 / ranks).item() + results.get('mrr', 0.0)
                for k in range(10):
                    results['hits@{}'.format(k + 1)] = torch.numel(ranks[ranks <= (k + 1)]) + results.get(
                        'hits@{}'.format(k + 1), 0.0)

                # time unaware filtering
                score = torch.where(indep_lab_.byte(), -torch.ones_like(score) * 10000000, score)
                score[b_range, obj_tar_] = target_score

                # time unaware filtered ranking
                ranks = 1 + torch.argsort(torch.argsort(score, dim=1, descending=True), dim=1, descending=False)[
                    b_range, obj_tar_]
                ranks = ranks.float()
                results['count_ind'] = torch.numel(ranks) + results.get('count_ind', 0.0)
                results['mar_ind'] = torch.sum(ranks).item() + results.get('mar_ind', 0.0)
                results['mrr_ind'] = torch.sum(1.0 / ranks).item() + results.get('mrr_ind', 0.0)
                for k in range(10):
                    results['hits@{}_ind'.format(k + 1)] = torch.numel(ranks[ranks <= (k + 1)]) + results.get(
                        'hits@{}_ind'.format(k + 1), 0.0)

                rank_count += rank_group_num
            del sub_tar_, rel_tar_, obj_tar_, lab_tar_, indep_lab_

        results['mar'] = round(results['mar'] / results['count'], 5)
        results['mrr'] = round(results['mrr'] / results['count'], 5)
        results['mar_raw'] = round(results['mar_raw'] / results['count_raw'], 5)
        results['mrr_raw'] = round(results['mrr_raw'] / results['count_raw'], 5)
        results['mar_ind'] = round(results['mar_ind'] / results['count_ind'], 5)
        results['mrr_ind'] = round(results['mrr_ind'] / results['count_ind'], 5)
        for k in range(10):
            results['hits@{}'.format(k + 1)] = round(results['hits@{}'.format(k + 1)] / results['count'], 5)
            results['hits@{}_raw'.format(k + 1)] = round(results['hits@{}_raw'.format(k + 1)] / results['count_raw'], 5)
            results['hits@{}_ind'.format(k + 1)] = round(results['hits@{}_ind'.format(k + 1)] / results['count_ind'], 5)

        t2 = time.time()
        print("evaluation time: ", t2 - t1)
        logger.info("evaluation time: {}".format(t2 - t1))

    return results

def old_predict(loader, model, params, num_e, test_adjmtx, logger): #original version before modification
    model.eval()
    p = params
    rank_group_num = 2000

    with torch.no_grad():
        results = {}
        iter = loader

        print("Start evaluation")
        t1 = time.time()

        for step, (
        sub_in, rel_in, obj_in, lab_in, sub_tar, rel_tar, obj_tar, lab_tar, tar_ts, in_ts, edge_idlist, edge_typelist,
        indep_lab, adj_mtx, edge_jump_w, edge_jump_id, rel_jump) in enumerate(iter):
            if len(sub_tar) == 0:
                continue

            # forward
            emb = model.forward_eval(in_ts, tar_ts, edge_idlist, edge_typelist, edge_jump_id, edge_jump_w, rel_jump) # comment eval_paper_authors: this only returns embeddings for 1 timestep

            rank_count = 0
            while rank_count < sub_tar[0].shape[0]: # comment eval_paper_authors: no matter how many steps ahead with tar_ts, this will always look at element [0]
                l, r = rank_count, (rank_count + rank_group_num) if (rank_count + rank_group_num) <= sub_tar[0].shape[
                    0] else sub_tar[0].shape[0]

                # push data onto gpu
                [sub_tar_, rel_tar_, obj_tar_, lab_tar_, indep_lab_] = \
                    push_data2(sub_tar[0][l:r], rel_tar[0][l:r], obj_tar[0][l:r], lab_tar[0][l:r, :],
                               indep_lab[0][l:r, :], device=p.device)

                # compute scores for corresponding triples
                score = model.score_comp(sub_tar_, rel_tar_, emb, model.odeblock.odefunc) #scores for all entities in this timestep (tar_ts) for all triples (sub and ob directions)
                b_range = torch.arange(score.shape[0], device=p.device) 

                # raw ranking
                ranks = 1 + torch.argsort(torch.argsort(score, dim=1, descending=True), dim=1, descending=False)[
                    b_range, obj_tar_]

                ranks = ranks.float()
                results['count_raw'] = torch.numel(ranks) + results.get('count_raw', 0.0)
                results['mar_raw'] = torch.sum(ranks).item() + results.get('mar_raw', 0.0)
                results['mrr_raw'] = torch.sum(1.0 / ranks).item() + results.get('mrr_raw', 0.0)
                for k in range(10):
                    results['hits@{}_raw'.format(k + 1)] = torch.numel(ranks[ranks <= (k + 1)]) + results.get(
                        'hits@{}_raw'.format(k + 1), 0.0)

                # time aware filtering
                target_score = score[b_range, obj_tar_]
                score = torch.where(lab_tar_.byte(), -torch.ones_like(score) * 10000000, score)
                score[b_range, obj_tar_] = target_score

                # time aware filtered ranking
                ranks = 1 + torch.argsort(torch.argsort(score, dim=1, descending=True), dim=1, descending=False)[
                    b_range, obj_tar_]
                ranks = ranks.float()
                results['count'] = torch.numel(ranks) + results.get('count', 0.0)
                results['mar'] = torch.sum(ranks).item() + results.get('mar', 0.0)
                results['mrr'] = torch.sum(1.0 / ranks).item() + results.get('mrr', 0.0)
                for k in range(10):
                    results['hits@{}'.format(k + 1)] = torch.numel(ranks[ranks <= (k + 1)]) + results.get(
                        'hits@{}'.format(k + 1), 0.0)

                # time unaware filtering
                score = torch.where(indep_lab_.byte(), -torch.ones_like(score) * 10000000, score)
                score[b_range, obj_tar_] = target_score

                # time unaware filtered ranking
                ranks = 1 + torch.argsort(torch.argsort(score, dim=1, descending=True), dim=1, descending=False)[
                    b_range, obj_tar_]
                ranks = ranks.float()
                results['count_ind'] = torch.numel(ranks) + results.get('count_ind', 0.0)
                results['mar_ind'] = torch.sum(ranks).item() + results.get('mar_ind', 0.0)
                results['mrr_ind'] = torch.sum(1.0 / ranks).item() + results.get('mrr_ind', 0.0)
                for k in range(10):
                    results['hits@{}_ind'.format(k + 1)] = torch.numel(ranks[ranks <= (k + 1)]) + results.get(
                        'hits@{}_ind'.format(k + 1), 0.0)

                rank_count += rank_group_num
            del sub_tar_, rel_tar_, obj_tar_, lab_tar_, indep_lab_

        results['mar'] = round(results['mar'] / results['count'], 5)
        results['mrr'] = round(results['mrr'] / results['count'], 5)
        results['mar_raw'] = round(results['mar_raw'] / results['count_raw'], 5)
        results['mrr_raw'] = round(results['mrr_raw'] / results['count_raw'], 5)
        results['mar_ind'] = round(results['mar_ind'] / results['count_ind'], 5)
        results['mrr_ind'] = round(results['mrr_ind'] / results['count_ind'], 5)
        for k in range(10):
            results['hits@{}'.format(k + 1)] = round(results['hits@{}'.format(k + 1)] / results['count'], 5)
            results['hits@{}_raw'.format(k + 1)] = round(results['hits@{}_raw'.format(k + 1)] / results['count_raw'], 5)
            results['hits@{}_ind'.format(k + 1)] = round(results['hits@{}_ind'.format(k + 1)] / results['count_ind'], 5)

        t2 = time.time()
        print("evaluation time: ", t2 - t1)
        logger.info("evaluation time: {}".format(t2 - t1))

    return results
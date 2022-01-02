"""
@author:chenyankai
@file:exp.py
@time:2020/10/10
"""
from src.train import *


def exp_i(args, train_file, eval_file, test_file, logging, task):
    ''' do a neighbor sampling in an online manner '''
    tf.reset_default_graph()
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    # os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3"


    train_data, eval_data, test_data, kg_dict, user_item_dict, item_user_dict, n_relation, n_entity, n_triplet = \
        load_data(args, train_file, eval_file, test_file)

    n_user = max(max(train_data[:, 0]), max(eval_data[:, 0]), max(test_data[:, 0])) + 1
    n_item = max(max(train_data[:, 1]), max(eval_data[:, 1]), max(test_data[:, 1])) + 1

    model = CGKGR(args, n_user, n_entity, n_relation)

    max_eval_auc = 0.0
    best_test_auc = 0.0
    best_test_f1 = 0.0
    best_epoch1 = 0
    eval_precision_list = []
    eval_recall_list = []
    eval_ndcg_list = []

    test_precision_list = []
    test_recall_list = []
    test_ndcg_list = []

    best_eval_recall = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    best_epoch2 = [0, 0, 0, 0, 0, 0]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(1, args.n_epoch + 1):

            #   training
            adj_u2i, adj_i2u, adj_i2e, adj_relation = construct_adj(args, n_user, n_item, n_entity,
                                                                    kg_dict, user_item_dict, item_user_dict)

            ngh_sample_dict = get_ngh_sample_feed_dict(model, adj_u2i, adj_i2u, adj_i2e, adj_relation)

            start = 0
            time0 = time()
            iter = 0
            total_loss = 0.0
            np.random.shuffle(train_data)

            while start + args.batch_size <= len(train_data):
                # _, loss = model.train(sess, get_feed_dict(model, train_data, start, start + args.batch_size))
                feed_dict = get_feed_dict(model, train_data, start, start + args.batch_size)
                feed_dict.update(ngh_sample_dict)
                _, loss = model.train(sess, feed_dict)

                total_loss += loss
                start += args.batch_size
                iter += 1
            time_train = time() - time0

            if task == 'ctr':
            #  ctr evaluation
                time0 = time()
                train_auc, train_f1 = ctr_evaluate(sess, model, train_data, args.batch_size, ngh_sample_dict)
                eval_auc, eval_f1 = ctr_evaluate(sess, model, eval_data, args.batch_size, ngh_sample_dict)
                test_auc, test_f1 = ctr_evaluate(sess, model, test_data, args.batch_size, ngh_sample_dict)
                if eval_auc > max_eval_auc:
                    max_eval_auc = eval_auc
                    best_test_auc = test_auc
                    best_test_f1 = test_f1
                    best_epoch1 = epoch
                time1 = time() - time0
                logging.info('Epoch %d  Loss: %.4f  Train-Time: %.3fs  CTR-eval-Time: %.3fs '
                         % (epoch, total_loss / iter, time_train, time1))
                logging.info(
                'Train auc: %.4f  f1: %.4f      Eval auc: %.4f  f1: %.4f      Test auc: %.4f  f1: %.4f'
                % (train_auc, train_f1, eval_auc, eval_f1, test_auc, test_f1))

            elif task == 'topk':
                #  # top-K evaluation
                time0 = time()
                user_list, train_record, eval_record, item_set, k_list = topk_settings(train_data, eval_data, n_item)
                eval_precision, eval_recall, eval_ndcg = topk_evaluate(sess, model, user_list, train_record, eval_record,
                                                                       item_set, k_list, args.batch_size, ngh_sample_dict)

                user_list, train_record, test_record, item_set, k_list = topk_settings(train_data, test_data, n_item)
                test_precision, test_recall, test_ndcg = topk_evaluate(sess, model, user_list, train_record, test_record,
                                                                       item_set, k_list, args.batch_size, ngh_sample_dict)
                time2 = time() - time0

                eval_precision_list.append(eval_precision)
                eval_recall_list.append(eval_recall)
                eval_ndcg_list.append(eval_ndcg)

                test_precision_list.append(test_precision)
                test_recall_list.append(test_recall)
                test_ndcg_list.append(test_ndcg)

                for i, _ in enumerate(k_list):
                    if eval_recall[i] > best_eval_recall[i]:
                        best_eval_recall[i] = eval_recall[i]
                        best_epoch2[i] = epoch  # record the epoch number

                line1 = 'Eval P:'
                for i in eval_precision:
                    line1 = line1 + '%.4f\t' % i
                line1 = line1 + 'R:'
                for i in eval_recall:
                    line1 = line1 + '%.4f\t' % i
                line1 = line1 + 'NDCG:'
                for i in eval_ndcg:
                    line1 = line1 + '%.4f\t' % i

                line2 = 'Test P:'
                for i in test_precision:
                    line2 = line2 + '%.4f\t' % i
                line2 = line2 + 'R:'
                for i in test_recall:
                    line2 = line2 + '%.4f\t' % i
                line2 = line2 + 'NDCG:'
                for i in test_ndcg:
                    line2 = line2 + '%.4f\t' % i

                logging.info('Epoch %d  Loss: %.4f  Train-Time: %.3fs   Top@k-eval-Time: %.3fs '
                         % (epoch, total_loss / iter, time_train, time2))
                logging.info(line1)
                logging.info(line2)

            elif task == 'ALL':
                time0 = time()
                eval_auc, eval_f1 = ctr_evaluate(sess, model, eval_data, args.batch_size, ngh_sample_dict)
                test_auc, test_f1 = ctr_evaluate(sess, model, test_data, args.batch_size, ngh_sample_dict)

                if eval_auc > max_eval_auc:
                    max_eval_auc = eval_auc
                    best_test_auc = test_auc
                    best_test_f1 = test_f1
                    best_epoch1 = epoch
                time1 = time() - time0

                logging.info('Epoch %d  Loss: %.4f  Train-Time: %.3fs  CTR-eval-Time: %.3fs '
                             % (epoch, total_loss / iter, time_train, time1))
                logging.info('Eval auc: %.4f  f1: %.4f      Test auc: %.4f  f1: %.4f'
                             % (eval_auc, eval_f1, test_auc, test_f1))

                time0 = time()

                user_list, train_record, eval_record, item_set, k_list = topk_settings(train_data, eval_data, n_item)
                eval_precision, eval_recall, eval_ndcg = topk_evaluate(sess, model, user_list, train_record, eval_record,
                                                                       item_set, k_list, args.batch_size, ngh_sample_dict)

                user_list, train_record, test_record, item_set, k_list = topk_settings(train_data, test_data, n_item)
                test_precision, test_recall, test_ndcg = topk_evaluate(sess, model, user_list, train_record, test_record,
                                                                       item_set, k_list, args.batch_size, ngh_sample_dict)
                time1 = time() - time0

                eval_precision_list.append(eval_precision)
                eval_recall_list.append(eval_recall)
                eval_ndcg_list.append(eval_ndcg)

                test_precision_list.append(test_precision)
                test_recall_list.append(test_recall)
                test_ndcg_list.append(test_ndcg)

                for i, _ in enumerate(k_list):
                    if eval_recall[i] > best_eval_recall[i]:
                        best_eval_recall[i] = eval_recall[i]
                        best_epoch2[i] = epoch  # record the epoch number

                line1 = 'Eval P:'
                for i in eval_precision:
                    line1 = line1 + '%.4f\t' % i
                line1 = line1 + 'R:'
                for i in eval_recall:
                    line1 = line1 + '%.4f\t' % i
                line1 = line1 + 'NDCG:'
                for i in eval_ndcg:
                    line1 = line1 + '%.4f\t' % i

                line2 = 'Test P:'
                for i in test_precision:
                    line2 = line2 + '%.4f\t' % i
                line2 = line2 + 'R:'
                for i in test_recall:
                    line2 = line2 + '%.4f\t' % i
                line2 = line2 + 'NDCG:'
                for i in test_ndcg:
                    line2 = line2 + '%.4f\t' % i

                logging.info('Epoch %d  Loss: %.4f  Train-Time: %.3fs  CTR-eval-Time: %.3fs '
                             % (epoch, total_loss / iter, time_train, time1))

                logging.info(line1)
                logging.info(line2)

                total_num = get_total_parameters()
                logging.info('Total: %d ' % total_num)

            else:
                raise NotImplementedError

    k_list = [1, 5, 10, 20, 50, 100]
    return best_epoch1, best_epoch2, k_list, best_test_auc, best_test_f1, \
           eval_precision_list, eval_recall_list, eval_ndcg_list, \
           test_precision_list, test_recall_list, test_ndcg_list


def Exp_all(args):
    log_name = create_log_name(args.saved_dir)
    log_config(path=args.saved_dir, name=log_name, level=logging.DEBUG, console_level=logging.DEBUG, console=True)
    logging.info(args)
    logging.info('CG-KGR experiments.')

    file_name = [1, 2, 3, 4, 5]
    auc = []
    f1 = []

    recall_list = [[] for _ in range(6)]
    ndcg_list = [[] for _ in range(6)]

    k_list = []
    for i in file_name:
        train_file = 'train_' + str(i) + '.txt'
        eval_file = 'eval_' + str(i) + '.txt'
        test_file = 'test_' + str(i) + '.txt'

        logging.info(train_file)
        best_epoch1, best_epoch2, k_list, best_test_auc, best_test_f1, \
        eval_precision_list, eval_recall_list, eval_ndcg_list, \
        test_precision_list, test_recall_list, test_ndcg_list = exp_i(args, train_file, eval_file,
                                                                      test_file, logging, args.task)

        logging.info('')
        auc.append(best_test_auc)
        f1.append(best_test_f1)

        logging.info('CTR Evaluation - Best epoch: %d   corresponding Test auc: %.4f    corresponding Test F1: %.4f'
            % (best_epoch1, best_test_auc, best_test_f1))

        logging.info('Top@k Evaluation')
        for j, k in enumerate(k_list):
            idx = best_epoch2[j]
            recall_list[j].append(test_recall_list[idx - 1][j])
            ndcg_list[j].append(test_ndcg_list[idx - 1][j])
            logging.info('Top@%d:  Best epoch: %d   corresponding Test R: %.4f    corresponding Test NDCG: %.4f'
                         % (k, best_epoch2[j], test_recall_list[idx - 1][j], test_ndcg_list[idx - 1][j]))
        logging.info('--------------------------------')
        logging.info('')

    for i, k in enumerate(k_list):
        logging.info('Top@%d recommendation:   Avg-best-R %.4f | Avg-best-NDCG: %.4f ' %
                     (k, np.mean(recall_list[i]), np.mean(ndcg_list[i])))


    logging.info(' Avg-best-auc %.4f | Avg-best-F1: %.4f | Max-best-Auc: %.4f | Max-best-F1: %.4f' %
                 (np.mean(auc), np.mean(f1), max(auc), max(f1)))
    logging.info('********************************************************************************************')
    logging.info('********************************************************************************************')
    return log_name




def exp_i_save(args, train_file, eval_file, test_file, logging, task, data_split, save_task):
    ''' do a neighbor sampling in an online manner '''
    tf.reset_default_graph()
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)


    train_data, eval_data, test_data, kg_dict, user_item_dict, item_user_dict, n_relation, n_entity, n_triplet = \
        load_data(args, train_file, eval_file, test_file)

    n_user = max(max(train_data[:, 0]), max(eval_data[:, 0]), max(test_data[:, 0])) + 1
    n_item = max(max(train_data[:, 1]), max(eval_data[:, 1]), max(test_data[:, 1])) + 1

    model = CGKGR(args, n_user, n_entity, n_relation)

    max_eval_auc = 0.0
    best_test_auc = 0.0
    best_test_f1 = 0.0
    best_epoch1 = 0
    eval_precision_list = []
    eval_recall_list = []
    eval_ndcg_list = []

    test_precision_list = []
    test_recall_list = []
    test_ndcg_list = []

    best_eval_recall = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    best_epoch2 = [0, 0, 0, 0, 0, 0]

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(1, args.n_epoch + 1):

            #   training
            adj_u2i, adj_i2u, adj_i2e, adj_relation = construct_adj(args, n_user, n_item, n_entity,
                                                                    kg_dict, user_item_dict, item_user_dict)

            ngh_sample_dict = get_ngh_sample_feed_dict(model, adj_u2i, adj_i2u, adj_i2e, adj_relation)

            start = 0
            time0 = time()
            iter = 0
            total_loss = 0.0
            np.random.shuffle(train_data)

            while start + args.batch_size <= len(train_data):
                feed_dict = get_feed_dict(model, train_data, start, start + args.batch_size)
                feed_dict.update(ngh_sample_dict)
                _, loss = model.train(sess, feed_dict)

                total_loss += loss
                start += args.batch_size
                iter += 1
            time_train = time() - time0

            if task == 'ALL':
                time0 = time()
                eval_auc, eval_f1, eval_scores = ctr_special(sess, model, eval_data, args.batch_size, ngh_sample_dict)
                test_auc, test_f1, test_scores = ctr_special(sess, model, test_data, args.batch_size, ngh_sample_dict)

                if save_task == 'ctr':
                    save_file = f'./checkpoints/{args.data_name}/CTR-eval-split-{data_split}-epoch-{epoch}.npy'
                    np.save(save_file, eval_scores)

                    save_file = f'./checkpoints/{args.data_name}/CTR-test-split-{data_split}-epoch-{epoch}.npy'
                    np.save(save_file, test_scores)

                if eval_auc > max_eval_auc:
                    max_eval_auc = eval_auc
                    best_test_auc = test_auc
                    best_test_f1 = test_f1
                    best_epoch1 = epoch
                time1 = time() - time0

                logging.info('Epoch %d  Loss: %.4f  Train-Time: %.3fs  CTR-eval-Time: %.3fs '
                             % (epoch, total_loss / iter, time_train, time1))
                logging.info('Eval auc: %.4f  f1: %.4f      Test auc: %.4f  f1: %.4f'
                             % (eval_auc, eval_f1, test_auc, test_f1))

                time0 = time()

                eval_user_list, train_record, eval_record, item_set, k_list = topk_settings(train_data, eval_data, n_item)
                eval_precision, eval_recall, eval_ndcg, eval_score_map = topk_speical(sess, model, eval_user_list, train_record, eval_record,
                                                                       item_set, k_list, args.batch_size, ngh_sample_dict)

                test_user_list, train_record, test_record, item_set, k_list = topk_settings(train_data, test_data, n_item)
                test_precision, test_recall, test_ndcg, test_score_map = topk_speical(sess, model, test_user_list, train_record, test_record,
                                                                       item_set, k_list, args.batch_size, ngh_sample_dict)
                time1 = time() - time0

                if (save_task == 'topk'):
                    save_file = f'./checkpoints/{args.data_name}/TopK-eval-split-{data_split}-epoch-{epoch}.npy'
                    np.save(save_file, [eval_score_map, eval_user_list])

                    save_file = f'./checkpoints/{args.data_name}/TopK-test-split-{data_split}-epoch-{epoch}.npy'
                    np.save(save_file, [test_score_map, test_user_list])

                eval_precision_list.append(eval_precision)
                eval_recall_list.append(eval_recall)
                eval_ndcg_list.append(eval_ndcg)

                test_precision_list.append(test_precision)
                test_recall_list.append(test_recall)
                test_ndcg_list.append(test_ndcg)

                for i, _ in enumerate(k_list):
                    if eval_recall[i] > best_eval_recall[i]:
                        best_eval_recall[i] = eval_recall[i]
                        best_epoch2[i] = epoch  # record the epoch number

                line1 = 'Eval P:'
                for i in eval_precision:
                    line1 = line1 + '%.4f\t' % i
                line1 = line1 + 'R:'
                for i in eval_recall:
                    line1 = line1 + '%.4f\t' % i
                line1 = line1 + 'NDCG:'
                for i in eval_ndcg:
                    line1 = line1 + '%.4f\t' % i

                line2 = 'Test P:'
                for i in test_precision:
                    line2 = line2 + '%.4f\t' % i
                line2 = line2 + 'R:'
                for i in test_recall:
                    line2 = line2 + '%.4f\t' % i
                line2 = line2 + 'NDCG:'
                for i in test_ndcg:
                    line2 = line2 + '%.4f\t' % i

                logging.info('Epoch %d  Loss: %.4f  Train-Time: %.3fs  CTR-eval-Time: %.3fs '
                             % (epoch, total_loss / iter, time_train, time1))

                logging.info(line1)
                logging.info(line2)

                total_num = get_total_parameters()
                logging.info('Total: %d ' % total_num)

            else:
                raise NotImplementedError


    k_list = [1, 5, 10, 20, 50, 100]
    return best_epoch1, best_epoch2, k_list, best_test_auc, best_test_f1, \
           eval_precision_list, eval_recall_list, eval_ndcg_list, \
           test_precision_list, test_recall_list, test_ndcg_list


def Exp_all_save(args, type):
    log_name = create_log_name(args.saved_dir)
    log_config(path=args.saved_dir, name=log_name, level=logging.DEBUG, console_level=logging.DEBUG, console=True)
    logging.info(args)
    logging.info('CG-KGR experiments.')

    file_name = [1, 2, 3, 4, 5]

    auc = []
    f1 = []
    recall_list = [[] for _ in range(6)]
    ndcg_list = [[] for _ in range(6)]
    k_list = []

    for index, i in enumerate(file_name):
        train_file = 'train_' + str(i) + '.txt'
        eval_file = 'eval_' + str(i) + '.txt'
        test_file = 'test_' + str(i) + '.txt'


        logging.info(train_file)
        best_epoch1, best_epoch2, k_list, best_test_auc, best_test_f1, \
        eval_precision_list, eval_recall_list, eval_ndcg_list, \
        test_precision_list, test_recall_list, test_ndcg_list = \
            exp_i_save(args, train_file, eval_file, test_file, logging, args.task, i, type)

        logging.info('')
        auc.append(best_test_auc)
        f1.append(best_test_f1)

        logging.info('CTR Evaluation - Best epoch: %d   corresponding Test auc: %.4f    corresponding Test F1: %.4f'
                     % (best_epoch1, best_test_auc, best_test_f1))

        logging.info('Top@k Evaluation')
        for j, k in enumerate(k_list):
            idx = best_epoch2[j]
            recall_list[j].append(test_recall_list[idx - 1][j])
            ndcg_list[j].append(test_ndcg_list[idx - 1][j])
            logging.info('Top@%d:  Best epoch: %d   corresponding Test R: %.4f    corresponding Test NDCG: %.4f'
                         % (k, best_epoch2[j], test_recall_list[idx - 1][j], test_ndcg_list[idx - 1][j]))
        logging.info('--------------------------------')
        logging.info('')

    for i, k in enumerate(k_list):
        logging.info('Top@%d recommendation:   Avg-best-R %.4f | Avg-best-NDCG: %.4f ' %
                     (k, np.mean(recall_list[i]), np.mean(ndcg_list[i])))


    logging.info(' Avg-best-auc %.4f | Avg-best-F1: %.4f | Max-best-Auc: %.4f | Max-best-F1: %.4f' %
                 (np.mean(auc), np.mean(f1), max(auc), max(f1)))
    logging.info('********************************************************************************************')
    logging.info('********************************************************************************************')
    return log_name


def ctr_offline(data, batch_size, scores_load):
    auc_list = []
    f1_list = []
    start = 0
    batch_i = 0

    while start + batch_size <= data.shape[0]:
        labels = data[start:start + batch_size, 2]
        scores = scores_load[batch_i]
        auc = roc_auc_score(y_true=labels, y_score=scores)
        scores[scores >= 0.5] = 1
        scores[scores < 0.5] = 0
        f1 = f1_score(y_true=labels, y_pred=scores)
        auc_list.append(auc)
        f1_list.append(f1)
        start += batch_size
        batch_i += 1
    return float(np.mean(auc_list)), float(np.mean(f1_list))


def top_k_offline(train_data, test_data, score_map, user_list, n_item):
    k_list = [1, 5, 10, 20, 50, 100]
    precision_list = {k: [] for k in k_list}
    recall_list = {k: [] for k in k_list}
    ndcg_list = {k: [] for k in k_list}

    _, train_record, test_record, item_set, _ = \
        topk_settings(train_data, test_data, n_item)

    user_list = list(user_list)
    for user in user_list:
        item_score_map = score_map[user]
        item_score_pair_sorted = sorted(item_score_map.items(), key=lambda x: x[1], reverse=True)
        item_sorted = [i[0] for i in item_score_pair_sorted]

        hits = np.zeros(len(item_sorted))
        index = [i for i, x in enumerate(item_sorted) if x in test_record[user]]
        hits[index] = 1

        for k in k_list:
            hit_k = hits[:k]
            hit_num = np.sum(hit_k)
            precision_list[k].append(hit_num / k)
            recall_list[k].append(hit_num / len(test_record[user]))
            dcg = np.sum((2 ** hit_k - 1) / np.log2(np.arange(2, k + 2)))
            sorted_hits_k = np.flip(np.sort(hits))[:k]
            idcg = np.sum((2 ** sorted_hits_k - 1) / np.log2(np.arange(2, k + 2)))
            # idcg[idcg == 0] = np.inf
            ndcg_list[k].append(dcg / idcg)

    precision = [np.mean(precision_list[k]) for k in k_list]
    recall = [np.mean(recall_list[k]) for k in k_list]
    ndcg = [np.mean(ndcg_list[k]) for k in k_list]

    return precision, recall, ndcg


def reproduce_i(args, train_data, test_data, n_item, pretrain_file, task):
    ''' do a neighbor sampling in an online manner '''
    tf.reset_default_graph()
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)

    if task == 'ctr':
        # Read data from checkpoint file
        scores = np.load(pretrain_file)
        test_auc, test_f1 = ctr_offline(test_data, args.batch_size, scores)

        return test_auc, test_f1
    else:

        scores_map, user_list = np.load(pretrain_file, allow_pickle=True)
        test_precision, test_recall, test_ndcg = top_k_offline(train_data, test_data, scores_map, user_list, n_item)

        line = ''
        for i in test_precision:
            line = line + '%.4f\t' % i
        line = line + 'R:'
        for i in test_recall:
            line = line + '%.4f\t' % i
        line = line + 'NDCG:'
        for i in test_ndcg:
            line = line + '%.4f\t' % i

        return test_precision[3], test_recall[3], test_ndcg[3], line


def Reproduce(args, task):
    log_name = create_log_name(args.saved_dir)
    log_config(path=args.saved_dir, name=log_name, level=logging.DEBUG, console_level=logging.DEBUG, console=True)
    logging.info(args)
    logging.info('Reproducing CG-KGR.')

    file_name = [1, 2, 3, 4, 5]

    if task == 'ctr':
        for i in file_name:
            train_file = 'train_' + str(i) + '.txt'
            eval_file = 'eval_' + str(i) + '.txt'
            test_file = 'test_' + str(i) + '.txt'
            logging.info(f'Datasplit-[{i}]:')

            train_data, eval_data, test_data, kg_dict, user_item_dict, item_user_dict, n_relation, n_entity, n_triplet = \
                load_data(args, train_file, eval_file, test_file)
            n_item = max(max(train_data[:, 1]), max(eval_data[:, 1]), max(test_data[:, 1])) + 1

            for epoch in range(1, args.n_epoch + 1):
                eval_file_ctr = f'./checkpoints/{args.data_name}/CTR-eval-split-{i}-epoch-{epoch}.npy'
                eval_auc, eval_f1 = reproduce_i(args, train_data, eval_data, n_item, eval_file_ctr, 'ctr')

                test_file_ctr = f'./checkpoints/{args.data_name}/CTR-test-split-{i}-epoch-{epoch}.npy'
                test_auc, test_f1 = reproduce_i(args, train_data, test_data, n_item, test_file_ctr, 'ctr')

                logging.info('Data split-[%d]  CTR: Epoch %d - Eval auc: %.4f  f1: %.4f   Test auc: %.4f  f1: %.4f'
                             % (i, epoch, eval_auc, eval_f1, test_auc, test_f1))

            logging.info('--------------------------------')
            logging.info('')

    else:
        recall_list = []
        ndcg_list = []
        for i in file_name:
            train_file = 'train_' + str(i) + '.txt'
            eval_file = 'eval_' + str(i) + '.txt'
            test_file = 'test_' + str(i) + '.txt'
            logging.info(f'Data split-[{i}] Top@k: \n')

            train_data, eval_data, test_data, kg_dict, user_item_dict, item_user_dict, n_relation, n_entity, n_triplet = \
                load_data(args, train_file, eval_file, test_file)
            n_item = max(max(train_data[:, 1]), max(eval_data[:, 1]), max(test_data[:, 1])) + 1

            r_list = []
            n_list = []
            for epoch in range(1, args.n_epoch + 1):
                eval_file_topk = f'./checkpoints/{args.data_name}/TopK-eval-split-{i}-epoch-{epoch}.npy'
                eval_precision, eval_recall, eval_ndcg, line1 = reproduce_i(args, train_data, eval_data, n_item,
                                                                          eval_file_topk, 'topk')
                line1 = 'Eval P: ' +line1


                test_file_topk = f'./checkpoints/{args.data_name}/TopK-test-split-{i}-epoch-{epoch}.npy'
                test_precision, test_recall, test_ndcg, line2 = reproduce_i(args, train_data, test_data, n_item,
                                                                           test_file_topk, 'topk')
                line2 = 'Test P: ' + line2
                logging.info(line1)
                logging.info(line2)
                logging.info('--------------------------------')
                logging.info('')

                r_list.append(test_recall)
                n_list.append(test_ndcg)

            recall_list.append(max(r_list))
            ndcg_list.append(max(n_list))
        logging.info('Top-20 recommendation over 5 splits:   Avg-best-R %.4f | Avg-best-NDCG: %.4f ' % (
        np.mean(recall_list), np.mean(ndcg_list)))

    logging.info('********************************************************************************************')
    logging.info('********************************************************************************************')


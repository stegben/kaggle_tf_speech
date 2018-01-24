import os.path as osp
from tempfile import mkdtemp
from datetime import datetime
from typing import Iterable, Tuple

import numpy as np
# from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tqdm import tqdm

from ..names import (
    X_PLACE,
    Y_PLACE,
    SAMPLE_WEIGHT_PLACE,
    LR_PLACE,
    WAVELET_DROPOUT_PLACE,
    CONV_DROPOUT_PLACE,
    DENSE_DROPOUT_PLACE,
    IS_TRAINING_PLACE,
    OP_INFERENCE,
    OP_LOSS,
    OP_TRAIN,
)
from ..utils import (
    BatchGenerator,
    restore_session,
    save_session,
)


def fit_generator(
        sess,
        train_batch_gen,
        evaluate_set: Iterable[Tuple[int, np.ndarray, np.ndarray, bool]],
        max_batches: int = 100000,
        learning_rate: float = 0.001,
        learning_rate_decay_ratio: float = 1 / 3,
        learning_rate_decay_rounds: int = 10,
        early_stopping_rounds: int = 50,
        save_folder: str = './',
        save_best: bool = True,
        model_name_prefix: str = 'some_random_model',
        saver=None,
        wavelet_dropout_prob=0.0,
        conv_dropout_prob=0.0,
        dense_dropout_prob=0.0,
    ) -> tf.Session:

    # TODO: Check only one validation set
    early_stop = False
    early_stopping_waiting_rounds = 0
    learning_rate_decay_waiting_rounds = 0
    best_validation_accuracy = 0.

    for n_batch, (
            x_batch,
            y_batch,
            weight_batch,
    ) in tqdm(enumerate(train_batch_gen()), total=max_batches):
        if (n_batch > max_batches) or early_stop:
            break
        fit_batch(
            sess,
            x_batch,
            y_batch,
            weight_batch,
            learning_rate,
            wavelet_dropout_prob=wavelet_dropout_prob,
            conv_dropout_prob=conv_dropout_prob,
            dense_dropout_prob=dense_dropout_prob,
        )

        for eval_set_id, (
            batches_per_round,
            x_evaluate,
            y_evaluate,
            is_validation_set,
        ) in enumerate(evaluate_set):
            if (n_batch + 1) % batches_per_round > 0:
                continue
            prediction, loss = predict_and_evaluate(sess, x_evaluate, y_evaluate, batch_size=16)

            ans = y_evaluate.argmax(axis=1)
            pred_max = prediction.argmax(axis=1)
            accuracy = (ans == pred_max).mean()

            print('N_BATCH {} with eval id {} loss: {}, accuracy: {}'.format(
                n_batch,
                eval_set_id,
                loss,
                accuracy,
            ))
            # print(confusion_matrix(ans, pred_max))

            if is_validation_set:

                if accuracy > best_validation_accuracy:
                    print('Improved!')
                    early_stopping_waiting_rounds = 0
                    learning_rate_decay_waiting_rounds = 0
                    # save best model
                    if save_best:
                        model_name = '{}__batch_{}_at_{}__valacc_{:.4f}.mdl'.format(
                            model_name_prefix,
                            n_batch,
                            datetime.now().strftime('%Y%m%d-%H%M%S'),
                            accuracy,
                        )
                        best_variable_path = osp.join(save_folder, model_name)
                        print('Save the model to: {}'.format(best_variable_path))
                        save_session(sess, best_variable_path, saver)
                    best_validation_accuracy = accuracy
                else:
                    if early_stopping_waiting_rounds >= early_stopping_rounds:
                        print('Early Stop')
                        early_stop = True
                    else:
                        early_stopping_waiting_rounds += 1

                    if learning_rate_decay_waiting_rounds >= learning_rate_decay_rounds:
                        print('Reduce learning rate')
                        learning_rate = learning_rate * learning_rate_decay_ratio
                        print('new learning rate: {}'.format(learning_rate))
                        learning_rate_decay_waiting_rounds = 0
                    else:
                        learning_rate_decay_waiting_rounds += 1
    if save_best:
        print('Restore best model from: {}'.format(best_variable_path))
        _, best_sess = restore_session(best_variable_path)
    return best_sess


def fit_batch(sess, x_batch, y_batch, weight_batch, learning_rate, **kwargs):
    graph = sess.graph
    x_place = graph.get_tensor_by_name(X_PLACE + ':0')
    y_place = graph.get_tensor_by_name(Y_PLACE + ':0')
    sample_weight_place = graph.get_tensor_by_name(SAMPLE_WEIGHT_PLACE + ':0')
    lr_place = graph.get_tensor_by_name(LR_PLACE + ':0')
    is_training_place = graph.get_tensor_by_name(IS_TRAINING_PLACE + ':0')
    loss_tensor = graph.get_tensor_by_name(OP_LOSS + ':0')
    train_op = graph.get_operation_by_name(OP_TRAIN)

    wavelet_dropout_place = graph.get_tensor_by_name(WAVELET_DROPOUT_PLACE + ':0')
    conv_dropout_place = graph.get_tensor_by_name(CONV_DROPOUT_PLACE + ':0')
    dense_dropout_place = graph.get_tensor_by_name(DENSE_DROPOUT_PLACE + ':0')

    _, batch_loss = sess.run(
        [train_op, loss_tensor],
        feed_dict={
            x_place: x_batch,
            y_place: y_batch,
            sample_weight_place: weight_batch,
            lr_place: learning_rate,
            wavelet_dropout_place: kwargs['wavelet_dropout_prob'],
            conv_dropout_place: kwargs['conv_dropout_prob'],
            dense_dropout_place: kwargs['dense_dropout_prob'],
            is_training_place: True,
        },
    )

    return batch_loss


def evaluate(sess, x_val, y_val, batch_size=128):
    loss = []
    batch_gen = BatchGenerator(
        x=x_val,
        y=y_val,
        batch_size=batch_size,
        shuffle=False,
    )
    for x_batch, y_batch in tqdm(batch_gen()):
        loss.append(evaluate_batch(sess, x_batch, y_batch))
    loss = np.mean(loss)
    return loss


def evaluate_batch(sess, x_batch, y_batch):
    graph = sess.graph
    x_place = graph.get_tensor_by_name(X_PLACE + ':0')
    y_place = graph.get_tensor_by_name(Y_PLACE + ':0')
    sample_weight_place = graph.get_tensor_by_name(SAMPLE_WEIGHT_PLACE + ':0')
    loss_op = graph.get_operation_by_name(OP_LOSS)
    loss_tensor = graph.get_tensor_by_name(OP_LOSS + ':0')
    _, batch_loss = sess.run(
        [loss_op, loss_tensor],
        feed_dict={
            x_place: x_batch,
            y_place: y_batch,
            sample_weight_place: np.ones(x_batch.shape[0]),
        },
    )
    return batch_loss


def predict(sess, x_test, batch_size=128):
    batch_gen = BatchGenerator(
        x=x_test,
        y=np.empty_like(x_test),
        batch_size=batch_size,
        shuffle=False,
    )
    result = []

    for x_batch, _ in tqdm(batch_gen()):
        batch_result = predict_batch(sess, x_batch)
        result.append(batch_result)
    return np.concatenate(result, axis=0)


def predict_batch(sess, x_batch):
    graph = sess.graph
    inference_op = graph.get_operation_by_name(OP_INFERENCE)
    inference_tensor = graph.get_tensor_by_name(
        OP_INFERENCE + ':0')
    x_place = graph.get_tensor_by_name(X_PLACE + ':0')
    sample_weight_place = graph.get_tensor_by_name(SAMPLE_WEIGHT_PLACE + ':0')
    _, batch_result = sess.run(
        [inference_op, inference_tensor],
        feed_dict={
            x_place: x_batch,
            sample_weight_place: np.ones(x_batch.shape[0]),

        },
    )
    return batch_result


def predict_and_evaluate(sess, x, y, batch_size=128):
    predictions = []
    losses = []
    batch_gen = BatchGenerator(
        x=x,
        y=y,
        batch_size=batch_size,
        shuffle=False,
    )
    for x_batch, y_batch in tqdm(batch_gen()):
        batch_prediction, batch_loss = predict_and_evaluate_batch(sess, x_batch, y_batch)
        predictions.append(batch_prediction)
        losses.append(batch_loss)
    prediction = np.concatenate(predictions, axis=0)
    loss = np.mean(losses)
    return prediction, loss


def predict_and_evaluate_batch(sess, x_batch, y_batch):
    graph = sess.graph
    x_place = graph.get_tensor_by_name(X_PLACE + ':0')
    y_place = graph.get_tensor_by_name(Y_PLACE + ':0')
    sample_weight_place = graph.get_tensor_by_name(SAMPLE_WEIGHT_PLACE + ':0')
    inference_op = graph.get_operation_by_name(OP_INFERENCE)
    inference_tensor = graph.get_tensor_by_name(
        OP_INFERENCE + ':0')
    loss_op = graph.get_operation_by_name(OP_LOSS)
    loss_tensor = graph.get_tensor_by_name(OP_LOSS + ':0')
    _, batch_prediction, _, batch_loss = sess.run(
        [inference_op, inference_tensor, loss_op, loss_tensor],
        feed_dict={
            x_place: x_batch,
            y_place: y_batch,
            sample_weight_place: np.ones(x_batch.shape[0]),
        },
    )
    return batch_prediction, batch_loss

# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import csv
import os
import classifier_utils
import modeling
import optimization_finetuning as optimization
import tokenization
import tensorflow as tf
# from loss import bi_tempered_logistic_loss

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string("task_name", None, "The name of the task to train.")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_bool(
    "do_predict_raw", False,
    "Whether to run the model in inference mode on input string.")

flags.DEFINE_multi_string("text", None,
                    "The text data to do predictions.")

flags.DEFINE_string(
    "saved_model_dir", None,
    "The directory where the saved model to be loaded.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion", 0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_string(
    "export_dir", None,
    "The directory where the exported SavedModel will be stored.")

def _serving_input_receiver_fn():
  """Creates an input function for serving."""
  seq_len = FLAGS.max_seq_length
  features = {
    "input_ids": tf.placeholder(tf.int64, shape=[None, seq_len], name="input_ids"),
    "input_mask": tf.placeholder(tf.int64, shape=[None, seq_len], name="input_mask"),
    "segment_ids": tf.placeholder(tf.int64, shape=[None, seq_len], name="segment_ids"),
    "label_ids": tf.placeholder(tf.int32, shape=[None], name="label_ids"),
  }
  return tf.estimator.export.build_raw_serving_input_receiver_fn(features)()

def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  processors = {
      "sentence_pair": classifier_utils.SentencePairClassificationProcessor,
      "lcqmc_pair":classifier_utils.LCQMCPairClassificationProcessor,
      "lcqmc": classifier_utils.LCQMCPairClassificationProcessor,
      "spam": classifier_utils.SpamClassificationProcessor
  }

  tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                FLAGS.init_checkpoint)

  if not (FLAGS.do_train or FLAGS.do_eval or FLAGS.do_predict or
          FLAGS.do_predict_raw or FLAGS.export_dir):
    raise ValueError(
        "At least one of `do_train`, `do_eval`, `do_predict`, `do_predict_raw` or `export_dir` "
        "must be True.")

  task_name = FLAGS.task_name.lower()

  if task_name not in processors:
    raise ValueError("Task not found: %s" % (task_name))

  processor = processors[task_name]()

  label_list = processor.get_labels()

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  if FLAGS.do_predict_raw:
    print("***** Running single prediction*****")
    texts = FLAGS.text
    # workaround: Getting duplicate text data when using FLAGS before runing the tf app
    texts = texts[0:int(len(texts) / 2)]
    print("text data: ", texts)
    # estimator.export_saved_model(FLAGS.output_dir, create_serving_input_receiver_fn(FLAGS.max_seq_length))
    from tensorflow.contrib import predictor
    import time
    start = time.process_time()
    predict_fn = predictor.from_saved_model(FLAGS.saved_model_dir)
    print("it took", time.process_time() - start, "to load model")
    start = time.process_time()
    input_ids_data, input_mask_data, segment_ids_data = [], [], []
    for i, t in enumerate(texts):
      guid = "predict_{}".format(i + 1)
      label = tokenization.convert_to_unicode("1")
      text_a = tokenization.convert_to_unicode(t)
      text_b = None
      example = classifier_utils.InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label)
      feature = classifier_utils.convert_single_example(0, example, label_list, FLAGS.max_seq_length, tokenizer)
      input_ids_data.append(feature.input_ids)
      input_mask_data.append(feature.input_mask)
      segment_ids_data.append(feature.segment_ids)

    features = collections.OrderedDict()
    features["input_ids"] = input_ids_data
    features["input_mask"] = input_mask_data
    features["segment_ids"] = segment_ids_data
    results = predict_fn(features)
    print(results)
    print("it took", time.process_time() - start, "to do prediction")
    return

  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  if FLAGS.max_seq_length > bert_config.max_position_embeddings:
    raise ValueError(
        "Cannot use sequence length %d because the BERT model "
        "was only trained up to sequence length %d" %
        (FLAGS.max_seq_length, bert_config.max_position_embeddings))

  tf.gfile.MakeDirs(FLAGS.output_dir)

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  # Cloud TPU: Invalid TPU configuration, ensure ClusterResolver is passed to tpu.
  print("###tpu_cluster_resolver:",tpu_cluster_resolver)
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  train_examples = None
  num_train_steps = None
  num_warmup_steps = None
  if FLAGS.do_train:
    train_examples =processor.get_train_examples(FLAGS.data_dir) # TODO
    print("###length of total train_examples:",len(train_examples))
    num_train_steps = int(len(train_examples)/ FLAGS.train_batch_size * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)

  model_fn = classifier_utils.model_fn_builder(
      bert_config=bert_config,
      num_labels=len(label_list),
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=num_train_steps,
      num_warmup_steps=num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      predict_batch_size=FLAGS.predict_batch_size)

  if FLAGS.do_train:
    train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
    train_file_exists=os.path.exists(train_file)
    print("###train_file_exists:", train_file_exists," ;train_file:",train_file)
    if not train_file_exists: # if tf_record file not exist, convert from raw text file. # TODO
        classifier_utils.file_based_convert_examples_to_features(train_examples, label_list, FLAGS.max_seq_length, tokenizer, train_file)
    tf.logging.info("***** Running training *****")
    tf.logging.info("  Num examples = %d", len(train_examples))
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.logging.info("  Num steps = %d", num_train_steps)
    train_input_fn = classifier_utils.file_based_input_fn_builder(
        input_file=train_file,
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        drop_remainder=True)
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

  if FLAGS.do_eval:
    eval_examples = processor.get_dev_examples(FLAGS.data_dir)
    num_actual_eval_examples = len(eval_examples)
    if FLAGS.use_tpu:
      # TPU requires a fixed batch size for all batches, therefore the number
      # of examples must be a multiple of the batch size, or else examples
      # will get dropped. So we pad with fake examples which are ignored
      # later on. These do NOT count towards the metric (all tf.metrics
      # support a per-instance weight, and these get a weight of 0.0).
      while len(eval_examples) % FLAGS.eval_batch_size != 0:
        eval_examples.append(classifier_utils.PaddingInputExample())

    eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
    classifier_utils.file_based_convert_examples_to_features(
        eval_examples, label_list, FLAGS.max_seq_length, tokenizer, eval_file)

    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                    len(eval_examples), num_actual_eval_examples,
                    len(eval_examples) - num_actual_eval_examples)
    tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

    # This tells the estimator to run through the entire set.
    eval_steps = None
    # However, if running eval on the TPU, you will need to specify the
    # number of steps.
    if FLAGS.use_tpu:
      assert len(eval_examples) % FLAGS.eval_batch_size == 0
      eval_steps = int(len(eval_examples) // FLAGS.eval_batch_size)

    eval_drop_remainder = True if FLAGS.use_tpu else False
    eval_input_fn = classifier_utils.file_based_input_fn_builder(
        input_file=eval_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=eval_drop_remainder)
    
    best_trial_info_file = os.path.join(FLAGS.output_dir, "best_trial.txt")

    def _best_trial_info():
      """Returns information about which checkpoints have been evaled so far."""
      if tf.gfile.Exists(best_trial_info_file):
        with tf.gfile.GFile(best_trial_info_file, "r") as best_info:
          global_step, best_metric_global_step, metric_value = (
              best_info.read().split(":"))
          global_step = int(global_step)
          best_metric_global_step = int(best_metric_global_step)
          metric_value = float(metric_value)
      else:
        metric_value = -1
        best_metric_global_step = -1
        global_step = -1
      tf.logging.info(
          "Best trial info: Step: %s, Best Value Step: %s, "
          "Best Value: %s", global_step, best_metric_global_step, metric_value)
      return global_step, best_metric_global_step, metric_value

    def _remove_checkpoint(checkpoint_path):
      for ext in ["meta", "data-00000-of-00001", "index"]:
        src_ckpt = checkpoint_path + ".{}".format(ext)
        tf.logging.info("removing {}".format(src_ckpt))
        tf.gfile.Remove(src_ckpt)

    def _find_valid_cands(curr_step):
      filenames = tf.gfile.ListDirectory(FLAGS.output_dir)
      candidates = []
      for filename in filenames:
        if filename.endswith(".index"):
          ckpt_name = filename[:-6]
          idx = ckpt_name.split("-")[-1]
          if int(idx) > curr_step:
            candidates.append(filename)
      return candidates

    output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")

    if task_name == "sts-b":
      key_name = "pearson"
    elif task_name == "cola":
      key_name = "matthew_corr"
    else:
      key_name = "eval_accuracy"

    global_step, best_perf_global_step, best_perf = _best_trial_info()
    writer = tf.gfile.GFile(output_eval_file, "w")
    while global_step < num_train_steps:
      steps_and_files = {}
      filenames = tf.gfile.ListDirectory(FLAGS.output_dir)
      for filename in filenames:
        if filename.endswith(".index"):
          ckpt_name = filename[:-6]
          cur_filename = os.path.join(FLAGS.output_dir, ckpt_name)
          if cur_filename.split("-")[-1] == "best":
            continue
          gstep = int(cur_filename.split("-")[-1])
          if gstep not in steps_and_files:
            tf.logging.info("Add {} to eval list.".format(cur_filename))
            steps_and_files[gstep] = cur_filename
      tf.logging.info("found {} files.".format(len(steps_and_files)))
      if not steps_and_files:
        tf.logging.info("found 0 file, global step: {}. Sleeping."
                        .format(global_step))
        time.sleep(60)
      else:
        for checkpoint in sorted(steps_and_files.items()):
          step, checkpoint_path = checkpoint
          if global_step >= step:
            if (best_perf_global_step != step and
                len(_find_valid_cands(step)) > 1):
              _remove_checkpoint(checkpoint_path)
            continue
          result = estimator.evaluate(
              input_fn=eval_input_fn,
              steps=eval_steps,
              checkpoint_path=checkpoint_path)
          global_step = result["global_step"]
          tf.logging.info("***** Eval results *****")
          for key in sorted(result.keys()):
            tf.logging.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
          writer.write("best = {}\n".format(best_perf))
          if result[key_name] > best_perf:
            best_perf = result[key_name]
            best_perf_global_step = global_step
          elif len(_find_valid_cands(global_step)) > 1:
            _remove_checkpoint(checkpoint_path)
          writer.write("=" * 50 + "\n")
          writer.flush()
          with tf.gfile.GFile(best_trial_info_file, "w") as best_info:
            best_info.write("{}:{}:{}".format(
                global_step, best_perf_global_step, best_perf))
    writer.close()

    for ext in ["meta", "data-00000-of-00001", "index"]:
      src_ckpt = "model.ckpt-{}.{}".format(best_perf_global_step, ext)
      tgt_ckpt = "model.ckpt-best.{}".format(ext)
      tf.logging.info("saving {} to {}".format(src_ckpt, tgt_ckpt))
      tf.io.gfile.rename(
          os.path.join(FLAGS.output_dir, src_ckpt),
          os.path.join(FLAGS.output_dir, tgt_ckpt),
          overwrite=True)

  if FLAGS.do_predict:
    predict_examples = processor.get_test_examples(FLAGS.data_dir)
    num_actual_predict_examples = len(predict_examples)
    if FLAGS.use_tpu:
      # TPU requires a fixed batch size for all batches, therefore the number
      # of examples must be a multiple of the batch size, or else examples
      # will get dropped. So we pad with fake examples which are ignored
      # later on.
      while len(predict_examples) % FLAGS.predict_batch_size != 0:
        predict_examples.append(classifier_utils.PaddingInputExample())

    predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
    classifier_utils.file_based_convert_examples_to_features(predict_examples, label_list,
                                            FLAGS.max_seq_length, tokenizer,
                                            predict_file)

    tf.logging.info("***** Running prediction*****")
    tf.logging.info("  Num examples = %d (%d actual, %d padding)",
                    len(predict_examples), num_actual_predict_examples,
                    len(predict_examples) - num_actual_predict_examples)
    tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

    predict_drop_remainder = True if FLAGS.use_tpu else False
    predict_input_fn = classifier_utils.file_based_input_fn_builder(
        input_file=predict_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=predict_drop_remainder)

    result = estimator.predict(input_fn=predict_input_fn)

    output_predict_file = os.path.join(FLAGS.output_dir, "test_results.tsv")
    with tf.gfile.GFile(output_predict_file, "w") as writer:
      num_written_lines = 0
      tf.logging.info("***** Predict results *****")
      for (i, prediction) in enumerate(result):
        probabilities = prediction["probabilities"]
        if i >= num_actual_predict_examples:
          break
        output_line = "\t".join(
            str(class_probability)
            for class_probability in probabilities) + "\n"
        writer.write(output_line)
        num_written_lines += 1
    assert num_written_lines == num_actual_predict_examples

  if FLAGS.export_dir:
    tf.gfile.MakeDirs(FLAGS.export_dir)
    checkpoint_path = os.path.join(FLAGS.output_dir, "model.ckpt-best")
    tf.logging.info("Starting to export model.")
    subfolder = estimator.export_saved_model(
        export_dir_base=FLAGS.export_dir,
        serving_input_receiver_fn=_serving_input_receiver_fn,
        checkpoint_path=checkpoint_path)
    tf.logging.info("Model exported to %s.", subfolder)

    # convert the exported model as tflite model
    converter = tf.lite.TFLiteConverter.from_saved_model(subfolder) # path to the SavedModel directory
    tflite_model = converter.convert()

    tflite_model_file = os.path.join(FLAGS.export_dir, "model.tflite")
    with tf.gfile.GFile(tflite_model_file, "w") as writer:
      writer.write(tflite_model)
    tf.logging.info("Convert exported model to %s.", tflite_model_file)  


if __name__ == "__main__":
  flags.mark_flag_as_required("task_name")
  flags.mark_flag_as_required("vocab_file")
  if FLAGS.do_predict_raw:
    flags.mark_flag_as_required("saved_model_dir")
  else:
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
  tf.app.run()
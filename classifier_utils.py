# -*- coding: utf-8 -*-
# @Author: bo.shi
# @Date:   2019-12-01 22:28:41
# @Last Modified by:   bo.shi
# @Last Modified time: 2019-12-02 18:36:50
# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Utility functions for GLUE classification tasks."""

from __future__ import absolute_import
from __future__ import division

from __future__ import print_function

import json
import collections
import csv
import os
import six
import modeling
import optimization
import tokenization
import tensorflow as tf

class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, label=None):
    """Constructs a InputExample.
    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label


class PaddingInputExample(object):
  """Fake example so the num input examples is a multiple of the batch size.
  When running eval/predict on the TPU, we need to pad the number of examples
  to be a multiple of the batch size, because the TPU requires a fixed batch
  size. The alternative is to drop the last batch, which is bad because it means
  the entire output data won't be generated.
  We use this class instead of `None` because treating `None` as padding
  battches could cause silent errors.
  """

class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               label_id,
               is_real_example=True):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id
    self.is_real_example = is_real_example

class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_test_examples(self, data_dir):
    """Gets a collection of `InputExample`s for prediction."""
    raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  @classmethod
  def _read_tsv(cls, input_file, delimiter="\t", quotechar=None):
    """Reads a tab separated value file."""
    with tf.gfile.Open(input_file, "r") as f:
      reader = csv.reader(f, delimiter=delimiter, quotechar=quotechar)
      lines = []
      for line in reader:
        lines.append(line)
      return lines

  @classmethod
  def _read_txt(cls, input_file):
    """Reads a tab separated value file."""
    with tf.gfile.Open(input_file, "r") as f:
      reader = f.readlines()
      lines = []
      for line in reader:
        lines.append(line.strip().split("_!_"))
      return lines

  @classmethod
  def _read_json(cls, input_file):
    """Reads a tab separated value file."""
    with tf.gfile.Open(input_file, "r") as f:
      reader = f.readlines()
      lines = []
      for line in reader:
        lines.append(json.loads(line.strip()))
      return lines


class XnliProcessor(DataProcessor):
  """Processor for the XNLI data set."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_json(os.path.join(data_dir, "train.json")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_json(os.path.join(data_dir, "dev.json")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_json(os.path.join(data_dir, "test.json")), "test")

  def _create_examples(self, lines, set_type):
    """See base class."""
    examples = []
    for (i, line) in enumerate(lines):
      guid = "%s-%s" % (set_type, i)
      text_a = tokenization.convert_to_unicode(line['premise'])
      text_b = tokenization.convert_to_unicode(line['hypo'])
      label = tokenization.convert_to_unicode(line['label']) if set_type != 'test' else 'contradiction'
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples

  def get_labels(self):
    """See base class."""
    return ["contradiction", "entailment", "neutral"]


# class TnewsProcessor(DataProcessor):
#     """Processor for the MRPC data set (GLUE version)."""
#
#     def get_train_examples(self, data_dir):
#         """See base class."""
#         return self._create_examples(
#             self._read_txt(os.path.join(data_dir, "toutiao_category_train.txt")), "train")
#
#     def get_dev_examples(self, data_dir):
#         """See base class."""
#         return self._create_examples(
#             self._read_txt(os.path.join(data_dir, "toutiao_category_dev.txt")), "dev")
#
#     def get_test_examples(self, data_dir):
#         """See base class."""
#         return self._create_examples(
#             self._read_txt(os.path.join(data_dir, "toutiao_category_test.txt")), "test")
#
#     def get_labels(self):
#         """See base class."""
#         labels = []
#         for i in range(17):
#             if i == 5 or i == 11:
#                 continue
#             labels.append(str(100 + i))
#         return labels
#
#     def _create_examples(self, lines, set_type):
#         """Creates examples for the training and dev sets."""
#         examples = []
#         for (i, line) in enumerate(lines):
#             if i == 0:
#                 continue
#             guid = "%s-%s" % (set_type, i)
#             text_a = tokenization.convert_to_unicode(line[3])
#             text_b = None
#             label = tokenization.convert_to_unicode(line[1])
#             examples.append(
#                 InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
#         return examples


class TnewsProcessor(DataProcessor):
  """Processor for the MRPC data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_json(os.path.join(data_dir, "train.json")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_json(os.path.join(data_dir, "dev.json")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_json(os.path.join(data_dir, "test.json")), "test")

  def get_labels(self):
    """See base class."""
    labels = []
    for i in range(17):
      if i == 5 or i == 11:
        continue
      labels.append(str(100 + i))
    return labels

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      guid = "%s-%s" % (set_type, i)
      text_a = tokenization.convert_to_unicode(line['sentence'])
      text_b = None
      label = tokenization.convert_to_unicode(line['label']) if set_type != 'test' else "100"
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


# class iFLYTEKDataProcessor(DataProcessor):
#     """Processor for the iFLYTEKData data set (GLUE version)."""
#
#     def get_train_examples(self, data_dir):
#         """See base class."""
#         return self._create_examples(
#             self._read_txt(os.path.join(data_dir, "train.txt")), "train")
#
#     def get_dev_examples(self, data_dir):
#         """See base class."""
#         return self._create_examples(
#             self._read_txt(os.path.join(data_dir, "dev.txt")), "dev")
#
#     def get_test_examples(self, data_dir):
#         """See base class."""
#         return self._create_examples(
#             self._read_txt(os.path.join(data_dir, "test.txt")), "test")
#
#     def get_labels(self):
#         """See base class."""
#         labels = []
#         for i in range(119):
#             labels.append(str(i))
#         return labels
#
#     def _create_examples(self, lines, set_type):
#         """Creates examples for the training and dev sets."""
#         examples = []
#         for (i, line) in enumerate(lines):
#             if i == 0:
#                 continue
#             guid = "%s-%s" % (set_type, i)
#             text_a = tokenization.convert_to_unicode(line[1])
#             text_b = None
#             label = tokenization.convert_to_unicode(line[0])
#             examples.append(
#                 InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
#         return examples


class iFLYTEKDataProcessor(DataProcessor):
  """Processor for the iFLYTEKData data set (GLUE version)."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_json(os.path.join(data_dir, "train.json")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_json(os.path.join(data_dir, "dev.json")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_json(os.path.join(data_dir, "test.json")), "test")

  def get_labels(self):
    """See base class."""
    labels = []
    for i in range(119):
      labels.append(str(i))
    return labels

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      guid = "%s-%s" % (set_type, i)
      text_a = tokenization.convert_to_unicode(line['sentence'])
      text_b = None
      label = tokenization.convert_to_unicode(line['label']) if set_type != 'test' else "0"
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


class AFQMCProcessor(DataProcessor):
  """Processor for the internal data set. sentence pair classification"""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_json(os.path.join(data_dir, "train.json")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_json(os.path.join(data_dir, "dev.json")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_json(os.path.join(data_dir, "test.json")), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      guid = "%s-%s" % (set_type, i)
      text_a = tokenization.convert_to_unicode(line['sentence1'])
      text_b = tokenization.convert_to_unicode(line['sentence2'])
      label = tokenization.convert_to_unicode(line['label']) if set_type != 'test' else '0'
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


class CMNLIProcessor(DataProcessor):
  """Processor for the CMNLI data set."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples_json(os.path.join(data_dir, "train.json"), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples_json(os.path.join(data_dir, "dev.json"), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples_json(os.path.join(data_dir, "test.json"), "test")

  def get_labels(self):
    """See base class."""
    return ["contradiction", "entailment", "neutral"]

  def _create_examples_json(self, file_name, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    lines = tf.gfile.Open(file_name, "r")
    index = 0
    for line in lines:
      line_obj = json.loads(line)
      index = index + 1
      guid = "%s-%s" % (set_type, index)
      text_a = tokenization.convert_to_unicode(line_obj["sentence1"])
      text_b = tokenization.convert_to_unicode(line_obj["sentence2"])
      label = tokenization.convert_to_unicode(line_obj["label"]) if set_type != 'test' else 'neutral'

      if label != "-":
        examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))

    return examples


class CslProcessor(DataProcessor):
  """Processor for the CSL data set."""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_json(os.path.join(data_dir, "train.json")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_json(os.path.join(data_dir, "dev.json")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_json(os.path.join(data_dir, "test.json")), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      guid = "%s-%s" % (set_type, i)
      text_a = tokenization.convert_to_unicode(" ".join(line['keyword']))
      text_b = tokenization.convert_to_unicode(line['abst'])
      label = tokenization.convert_to_unicode(line['label']) if set_type != 'test' else '0'
      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
    return examples


# class InewsProcessor(DataProcessor):
#   """Processor for the MRPC data set (GLUE version)."""
#
#   def get_train_examples(self, data_dir):
#     """See base class."""
#     return self._create_examples(
#         self._read_txt(os.path.join(data_dir, "train.txt")), "train")
#
#   def get_dev_examples(self, data_dir):
#     """See base class."""
#     return self._create_examples(
#         self._read_txt(os.path.join(data_dir, "dev.txt")), "dev")
#
#   def get_test_examples(self, data_dir):
#     """See base class."""
#     return self._create_examples(
#         self._read_txt(os.path.join(data_dir, "test.txt")), "test")
#
#   def get_labels(self):
#     """See base class."""
#     labels = ["0", "1", "2"]
#     return labels
#
#   def _create_examples(self, lines, set_type):
#     """Creates examples for the training and dev sets."""
#     examples = []
#     for (i, line) in enumerate(lines):
#       if i == 0:
#         continue
#       guid = "%s-%s" % (set_type, i)
#       text_a = tokenization.convert_to_unicode(line[2])
#       text_b = tokenization.convert_to_unicode(line[3])
#       label = tokenization.convert_to_unicode(line[0]) if set_type != "test" else '0'
#       examples.append(
#           InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
#     return examples
#
#
# class THUCNewsProcessor(DataProcessor):
#   """Processor for the THUCNews data set (GLUE version)."""
#
#   def get_train_examples(self, data_dir):
#     """See base class."""
#     return self._create_examples(
#         self._read_txt(os.path.join(data_dir, "train.txt")), "train")
#
#   def get_dev_examples(self, data_dir):
#     """See base class."""
#     return self._create_examples(
#         self._read_txt(os.path.join(data_dir, "dev.txt")), "dev")
#
#   def get_test_examples(self, data_dir):
#     """See base class."""
#     return self._create_examples(
#         self._read_txt(os.path.join(data_dir, "test.txt")), "test")
#
#   def get_labels(self):
#     """See base class."""
#     labels = []
#     for i in range(14):
#       labels.append(str(i))
#     return labels
#
#   def _create_examples(self, lines, set_type):
#     """Creates examples for the training and dev sets."""
#     examples = []
#     for (i, line) in enumerate(lines):
#       if i == 0 or len(line) < 3:
#         continue
#       guid = "%s-%s" % (set_type, i)
#       text_a = tokenization.convert_to_unicode(line[3])
#       text_b = None
#       label = tokenization.convert_to_unicode(line[0])
#       examples.append(
#           InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
#     return examples
#
# class LCQMCProcessor(DataProcessor):
#   """Processor for the internal data set. sentence pair classification"""
#
#   def __init__(self):
#     self.language = "zh"
#
#   def get_train_examples(self, data_dir):
#     """See base class."""
#     return self._create_examples(
#         self._read_tsv(os.path.join(data_dir, "train.txt")), "train")
#     # dev_0827.tsv
#
#   def get_dev_examples(self, data_dir):
#     """See base class."""
#     return self._create_examples(
#         self._read_tsv(os.path.join(data_dir, "dev.txt")), "dev")
#
#   def get_test_examples(self, data_dir):
#     """See base class."""
#     return self._create_examples(
#         self._read_tsv(os.path.join(data_dir, "test.txt")), "test")
#
#   def get_labels(self):
#     """See base class."""
#     return ["0", "1"]
#     # return ["-1","0", "1"]
#
#   def _create_examples(self, lines, set_type):
#     """Creates examples for the training and dev sets."""
#     examples = []
#     print("length of lines:", len(lines))
#     for (i, line) in enumerate(lines):
#       # print('#i:',i,line)
#       if i == 0:
#         continue
#       guid = "%s-%s" % (set_type, i)
#       try:
#         label = tokenization.convert_to_unicode(line[2])
#         text_a = tokenization.convert_to_unicode(line[0])
#         text_b = tokenization.convert_to_unicode(line[1])
#         examples.append(
#             InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
#       except Exception:
#         print('###error.i:', i, line)
#     return examples
#
#
# class JDCOMMENTProcessor(DataProcessor):
#   """Processor for the internal data set. sentence pair classification"""
#
#   def __init__(self):
#     self.language = "zh"
#
#   def get_train_examples(self, data_dir):
#     """See base class."""
#     return self._create_examples(
#         self._read_tsv(os.path.join(data_dir, "jd_train.csv"), ",", "\""), "train")
#     # dev_0827.tsv
#
#   def get_dev_examples(self, data_dir):
#     """See base class."""
#     return self._create_examples(
#         self._read_tsv(os.path.join(data_dir, "jd_dev.csv"), ",", "\""), "dev")
#
#   def get_test_examples(self, data_dir):
#     """See base class."""
#     return self._create_examples(
#         self._read_tsv(os.path.join(data_dir, "jd_test.csv"), ",", "\""), "test")
#
#   def get_labels(self):
#     """See base class."""
#     return ["1", "2", "3", "4", "5"]
#     # return ["-1","0", "1"]
#
#   def _create_examples(self, lines, set_type):
#     """Creates examples for the training and dev sets."""
#     examples = []
#     print("length of lines:", len(lines))
#     for (i, line) in enumerate(lines):
#       # print('#i:',i,line)
#       if i == 0:
#         continue
#       guid = "%s-%s" % (set_type, i)
#       try:
#         label = tokenization.convert_to_unicode(line[0])
#         text_a = tokenization.convert_to_unicode(line[1])
#         text_b = tokenization.convert_to_unicode(line[2])
#         examples.append(
#             InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
#       except Exception:
#         print('###error.i:', i, line)
#     return examples
#
#
# class BQProcessor(DataProcessor):
#   """Processor for the internal data set. sentence pair classification"""
#
#   def __init__(self):
#     self.language = "zh"
#
#   def get_train_examples(self, data_dir):
#     """See base class."""
#     return self._create_examples(
#         self._read_tsv(os.path.join(data_dir, "train.txt")), "train")
#     # dev_0827.tsv
#
#   def get_dev_examples(self, data_dir):
#     """See base class."""
#     return self._create_examples(
#         self._read_tsv(os.path.join(data_dir, "dev.txt")), "dev")
#
#   def get_test_examples(self, data_dir):
#     """See base class."""
#     return self._create_examples(
#         self._read_tsv(os.path.join(data_dir, "test.txt")), "test")
#
#   def get_labels(self):
#     """See base class."""
#     return ["0", "1"]
#     # return ["-1","0", "1"]
#
#   def _create_examples(self, lines, set_type):
#     """Creates examples for the training and dev sets."""
#     examples = []
#     print("length of lines:", len(lines))
#     for (i, line) in enumerate(lines):
#       # print('#i:',i,line)
#       if i == 0:
#         continue
#       guid = "%s-%s" % (set_type, i)
#       try:
#         label = tokenization.convert_to_unicode(line[2])
#         text_a = tokenization.convert_to_unicode(line[0])
#         text_b = tokenization.convert_to_unicode(line[1])
#         examples.append(
#             InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
#       except Exception:
#         print('###error.i:', i, line)
#     return examples
#
#
# class MnliProcessor(DataProcessor):
#   """Processor for the MultiNLI data set (GLUE version)."""
#
#   def get_train_examples(self, data_dir):
#     """See base class."""
#     return self._create_examples(
#         self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")
#
#   def get_dev_examples(self, data_dir):
#     """See base class."""
#     return self._create_examples(
#         self._read_tsv(os.path.join(data_dir, "dev_matched.tsv")),
#         "dev_matched")
#
#   def get_test_examples(self, data_dir):
#     """See base class."""
#     return self._create_examples(
#         self._read_tsv(os.path.join(data_dir, "test_matched.tsv")), "test")
#
#   def get_labels(self):
#     """See base class."""
#     return ["contradiction", "entailment", "neutral"]
#
#   def _create_examples(self, lines, set_type):
#     """Creates examples for the training and dev sets."""
#     examples = []
#     for (i, line) in enumerate(lines):
#       if i == 0:
#         continue
#       guid = "%s-%s" % (set_type, tokenization.convert_to_unicode(line[0]))
#       text_a = tokenization.convert_to_unicode(line[8])
#       text_b = tokenization.convert_to_unicode(line[9])
#       if set_type == "test":
#         label = "contradiction"
#       else:
#         label = tokenization.convert_to_unicode(line[-1])
#       examples.append(
#           InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
#     return examples
#
#
# class MrpcProcessor(DataProcessor):
#   """Processor for the MRPC data set (GLUE version)."""
#
#   def get_train_examples(self, data_dir):
#     """See base class."""
#     return self._create_examples(
#         self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")
#
#   def get_dev_examples(self, data_dir):
#     """See base class."""
#     return self._create_examples(
#         self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")
#
#   def get_test_examples(self, data_dir):
#     """See base class."""
#     return self._create_examples(
#         self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")
#
#   def get_labels(self):
#     """See base class."""
#     return ["0", "1"]
#
#   def _create_examples(self, lines, set_type):
#     """Creates examples for the training and dev sets."""
#     examples = []
#     for (i, line) in enumerate(lines):
#       if i == 0:
#         continue
#       guid = "%s-%s" % (set_type, i)
#       text_a = tokenization.convert_to_unicode(line[3])
#       text_b = tokenization.convert_to_unicode(line[4])
#       if set_type == "test":
#         label = "0"
#       else:
#         label = tokenization.convert_to_unicode(line[0])
#       examples.append(
#           InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
#     return examples
#
#
# class ColaProcessor(DataProcessor):
#   """Processor for the CoLA data set (GLUE version)."""
#
#   def get_train_examples(self, data_dir):
#     """See base class."""
#     return self._create_examples(
#         self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")
#
#   def get_dev_examples(self, data_dir):
#     """See base class."""
#     return self._create_examples(
#         self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")
#
#   def get_test_examples(self, data_dir):
#     """See base class."""
#     return self._create_examples(
#         self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")
#
#   def get_labels(self):
#     """See base class."""
#     return ["0", "1"]
#
#   def _create_examples(self, lines, set_type):
#     """Creates examples for the training and dev sets."""
#     examples = []
#     for (i, line) in enumerate(lines):
#       # Only the test set has a header
#       if set_type == "test" and i == 0:
#         continue
#       guid = "%s-%s" % (set_type, i)
#       if set_type == "test":
#         text_a = tokenization.convert_to_unicode(line[1])
#         label = "0"
#       else:
#         text_a = tokenization.convert_to_unicode(line[3])
#         label = tokenization.convert_to_unicode(line[1])
#       examples.append(
#           InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
#     return examples

class WSCProcessor(DataProcessor):
  """Processor for the internal data set. sentence pair classification"""

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_json(os.path.join(data_dir, "train.json")), "train")

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_json(os.path.join(data_dir, "dev.json")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_json(os.path.join(data_dir, "test.json")), "test")

  def get_labels(self):
    """See base class."""
    return ["true", "false"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    for (i, line) in enumerate(lines):
      guid = "%s-%s" % (set_type, i)
      text_a = tokenization.convert_to_unicode(line['text'])
      text_a_list = list(text_a)
      target = line['target']
      query = target['span1_text']
      query_idx = target['span1_index']
      pronoun = target['span2_text']
      pronoun_idx = target['span2_index']

      assert text_a[pronoun_idx: (pronoun_idx + len(pronoun))
                    ] == pronoun, "pronoun: {}".format(pronoun)
      assert text_a[query_idx: (query_idx + len(query))] == query, "query: {}".format(query)

      if pronoun_idx > query_idx:
        text_a_list.insert(query_idx, "_")
        text_a_list.insert(query_idx + len(query) + 1, "_")
        text_a_list.insert(pronoun_idx + 2, "[")
        text_a_list.insert(pronoun_idx + len(pronoun) + 2 + 1, "]")
      else:
        text_a_list.insert(pronoun_idx, "[")
        text_a_list.insert(pronoun_idx + len(pronoun) + 1, "]")
        text_a_list.insert(query_idx + 2, "_")
        text_a_list.insert(query_idx + len(query) + 2 + 1, "_")

      text_a = "".join(text_a_list)

      if set_type == "test":
        label = "true"
      else:
        label = line['label']

      examples.append(
          InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
    return examples


class COPAProcessor(DataProcessor):
  """Processor for the internal data set. sentence pair classification"""

  def __init__(self):
    self.language = "zh"

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_json(os.path.join(data_dir, "train.json")), "train")
    # dev_0827.tsv

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_json(os.path.join(data_dir, "dev.json")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_json(os.path.join(data_dir, "test.json")), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  @classmethod
  def _create_examples_one(self, lines, set_type):
    examples = []
    for (i, line) in enumerate(lines):
      guid1 = "%s-%s" % (set_type, i)
#         try:
      if line['question'] == 'cause':
        text_a = tokenization.convert_to_unicode(line['premise'] + '原因是什么呢？' + line['choice0'])
        text_b = tokenization.convert_to_unicode(line['premise'] + '原因是什么呢？' + line['choice1'])
      else:
        text_a = tokenization.convert_to_unicode(line['premise'] + '造成了什么影响呢？' + line['choice0'])
        text_b = tokenization.convert_to_unicode(line['premise'] + '造成了什么影响呢？' + line['choice1'])
      label = tokenization.convert_to_unicode(str(1 if line['label'] == 0 else 0)) if set_type != 'test' else '0'
      examples.append(
          InputExample(guid=guid1, text_a=text_a, text_b=text_b, label=label))
#         except Exception as e:
#             print('###error.i:',e, i, line)
    return examples

  @classmethod
  def _create_examples(self, lines, set_type):
    examples = []
    for (i, line) in enumerate(lines):
      i = 2 * i
      guid1 = "%s-%s" % (set_type, i)
      guid2 = "%s-%s" % (set_type, i + 1)
#         try:
      premise = tokenization.convert_to_unicode(line['premise'])
      choice0 = tokenization.convert_to_unicode(line['choice0'])
      label = tokenization.convert_to_unicode(str(1 if line['label'] == 0 else 0)) if set_type != 'test' else '0'
      #text_a2 = tokenization.convert_to_unicode(line['premise'])
      choice1 = tokenization.convert_to_unicode(line['choice1'])
      label2 = tokenization.convert_to_unicode(
          str(0 if line['label'] == 0 else 1)) if set_type != 'test' else '0'
      if line['question'] == 'effect':
        text_a = premise
        text_b = choice0
        text_a2 = premise
        text_b2 = choice1
      elif line['question'] == 'cause':
        text_a = choice0
        text_b = premise
        text_a2 = choice1
        text_b2 = premise
      else:
        print('wrong format!!')
        return None
      examples.append(
          InputExample(guid=guid1, text_a=text_a, text_b=text_b, label=label))
      examples.append(
          InputExample(guid=guid2, text_a=text_a2, text_b=text_b2, label=label2))
#         except Exception as e:
#             print('###error.i:',e, i, line)
    return examples

class LCQMCPairClassificationProcessor(DataProcessor): # TODO NEED CHANGE2
  """Processor for the internal data set. sentence pair classification"""
  def __init__(self):
    self.language = "zh"

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.txt")), "train")
    # dev_0827.tsv

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.txt")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.txt")), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]
    #return ["-1","0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    print("length of lines:",len(lines))
    for (i, line) in enumerate(lines):
      #print('#i:',i,line)
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, i)
      try:
          label = tokenization.convert_to_unicode(line[2])
          text_a = tokenization.convert_to_unicode(line[0])
          text_b = tokenization.convert_to_unicode(line[1])
          examples.append(
              InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
      except Exception:
          print('###error.i:', i, line)
    return examples

class SentencePairClassificationProcessor(DataProcessor):
  """Processor for the internal data set. sentence pair classification"""
  def __init__(self):
    self.language = "zh"

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train_0827.tsv")), "train")
    # dev_0827.tsv

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev_0827.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test_0827.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]
    #return ["-1","0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    print("length of lines:",len(lines))
    for (i, line) in enumerate(lines):
      #print('#i:',i,line)
      if i == 0:
        continue
      guid = "%s-%s" % (set_type, i)
      try:
          label = tokenization.convert_to_unicode(line[0])
          text_a = tokenization.convert_to_unicode(line[1])
          text_b = tokenization.convert_to_unicode(line[2])
          examples.append(
              InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
      except Exception:
          print('###error.i:', i, line)
    return examples

class SpamClassificationProcessor(DataProcessor):
  """Processor for the internal data set. sentence pair classification"""
  def __init__(self):
    self.language = "zh"

  def get_train_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "train.tsv")), "train")
    # dev_0827.tsv

  def get_dev_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "dev.tsv")), "dev")

  def get_test_examples(self, data_dir):
    """See base class."""
    return self._create_examples(
        self._read_tsv(os.path.join(data_dir, "test.tsv")), "test")

  def get_labels(self):
    """See base class."""
    return ["0", "1"]

  def _create_examples(self, lines, set_type):
    """Creates examples for the training and dev sets."""
    examples = []
    print("length of lines:",len(lines))
    for (i, line) in enumerate(lines):
      guid = "%s-%s" % (set_type, i)
      try:
          text_a = tokenization.convert_to_unicode(line[0])
          label = tokenization.convert_to_unicode(line[1])
          examples.append(
              InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
      except Exception:
          print('###error.i:', i, line)
    return examples

def convert_single_example(ex_index, example, label_list, max_seq_length,
                           tokenizer):
  """Converts a single `InputExample` into a single `InputFeatures`."""

  if isinstance(example, PaddingInputExample):
    return InputFeatures(
        input_ids=[0] * max_seq_length,
        input_mask=[0] * max_seq_length,
        segment_ids=[0] * max_seq_length,
        label_id=0,
        is_real_example=False)

  label_map = {}
  for (i, label) in enumerate(label_list):
    label_map[label] = i

  tokens_a = tokenizer.tokenize(example.text_a)
  tokens_b = None
  if example.text_b:
    tokens_b = tokenizer.tokenize(example.text_b)

  if tokens_b:
    # Modifies `tokens_a` and `tokens_b` in place so that the total
    # length is less than the specified length.
    # Account for [CLS], [SEP], [SEP] with "- 3"
    _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
  else:
    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > max_seq_length - 2:
      tokens_a = tokens_a[0:(max_seq_length - 2)]

  # The convention in BERT is:
  # (a) For sequence pairs:
  #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
  #  type_ids: 0     0  0    0    0     0       0 0     1  1  1  1   1 1
  # (b) For single sequences:
  #  tokens:   [CLS] the dog is hairy . [SEP]
  #  type_ids: 0     0   0   0  0     0 0
  #
  # Where "type_ids" are used to indicate whether this is the first
  # sequence or the second sequence. The embedding vectors for `type=0` and
  # `type=1` were learned during pre-training and are added to the wordpiece
  # embedding vector (and position vector). This is not *strictly* necessary
  # since the [SEP] token unambiguously separates the sequences, but it makes
  # it easier for the model to learn the concept of sequences.
  #
  # For classification tasks, the first vector (corresponding to [CLS]) is
  # used as the "sentence vector". Note that this only makes sense because
  # the entire model is fine-tuned.
  tokens = []
  segment_ids = []
  tokens.append("[CLS]")
  segment_ids.append(0)
  for token in tokens_a:
    tokens.append(token)
    segment_ids.append(0)
  tokens.append("[SEP]")
  segment_ids.append(0)

  if tokens_b:
    for token in tokens_b:
      tokens.append(token)
      segment_ids.append(1)
    tokens.append("[SEP]")
    segment_ids.append(1)

  input_ids = tokenizer.convert_tokens_to_ids(tokens)

  # The mask has 1 for real tokens and 0 for padding tokens. Only real
  # tokens are attended to.
  input_mask = [1] * len(input_ids)

  # Zero-pad up to the sequence length.
  while len(input_ids) < max_seq_length:
    input_ids.append(0)
    input_mask.append(0)
    segment_ids.append(0)

  assert len(input_ids) == max_seq_length
  assert len(input_mask) == max_seq_length
  assert len(segment_ids) == max_seq_length

  label_id = label_map[example.label]
  if ex_index < 5:
    tf.logging.info("*** Example ***")
    tf.logging.info("guid: %s" % (example.guid))
    tf.logging.info("tokens: %s" % " ".join(
        [tokenization.printable_text(x) for x in tokens]))
    tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
    tf.logging.info("label: %s (id = %d)" % (example.label, label_id))

  feature = InputFeatures(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids,
      label_id=label_id,
      is_real_example=True)
  return feature


def file_based_convert_examples_to_features(
    examples, label_list, max_seq_length, tokenizer, output_file):
  """Convert a set of `InputExample`s to a TFRecord file."""

  writer = tf.python_io.TFRecordWriter(output_file)

  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(ex_index, example, label_list,
                                     max_seq_length, tokenizer)

    def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_int_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)
    features["label_ids"] = create_int_feature([feature.label_id])
    features["is_real_example"] = create_int_feature(
        [int(feature.is_real_example)])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
  writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  name_to_features = {
      "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
      "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "label_ids": tf.FixedLenFeature([], tf.int64),
      "is_real_example": tf.FixedLenFeature([], tf.int64),
  }

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.to_int32(t)
      example[name] = t

    return example

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    d = tf.data.TFRecordDataset(input_file)
    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return d

  return input_fn


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
  """Truncates a sequence pair in place to the maximum length."""

  # This is a simple heuristic which will always truncate the longer sequence
  # one token at a time. This makes more sense than truncating an equal percent
  # of tokens from each, since if one sequence is very short then each token
  # that's truncated likely contains more information than a longer sequence.
  while True:
    total_length = len(tokens_a) + len(tokens_b)
    if total_length <= max_length:
      break
    if len(tokens_a) > len(tokens_b):
      tokens_a.pop()
    else:
      tokens_b.pop()


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                 labels, num_labels, use_one_hot_embeddings):
  """Creates a classification model."""
  model = modeling.BertModel(
      config=bert_config,
      is_training=is_training,
      input_ids=input_ids,
      input_mask=input_mask,
      token_type_ids=segment_ids,
      use_one_hot_embeddings=use_one_hot_embeddings)

  # In the demo, we are doing a simple classification task on the entire
  # segment.
  #
  # If you want to use the token-level output, use model.get_sequence_output()
  # instead.
  output_layer = model.get_pooled_output()

  hidden_size = output_layer.shape[-1].value

  output_weights = tf.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.truncated_normal_initializer(stddev=0.02))

  output_bias = tf.get_variable(
      "output_bias", [num_labels], initializer=tf.zeros_initializer())

  with tf.variable_scope("loss"):
    ln_type = bert_config.ln_type
    if ln_type == 'preln': # add by brightmart, 10-06. if it is preln, we need to an additonal layer: layer normalization as suggested in paper "ON LAYER NORMALIZATION IN THE TRANSFORMER ARCHITECTURE"
        print("ln_type is preln. add LN layer.")
        output_layer=layer_norm(output_layer)
    else:
        print("ln_type is postln or other,do nothing.")

    if is_training:
      # I.e., 0.1 dropout
      output_layer = tf.nn.dropout(output_layer, keep_prob=0.9)

    logits = tf.matmul(output_layer, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    probabilities = tf.nn.softmax(logits, axis=-1)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)

    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1) # todo 08-29 try temp-loss
    ###############bi_tempered_logistic_loss############################################################################
    # print("##cross entropy loss is used...."); tf.logging.info("##cross entropy loss is used....")
    # t1=0.9 #t1=0.90
    # t2=1.05 #t2=1.05
    # per_example_loss=bi_tempered_logistic_loss(log_probs,one_hot_labels,t1,t2,label_smoothing=0.1,num_iters=5) # TODO label_smoothing=0.0
    #tf.logging.info("per_example_loss:"+str(per_example_loss.shape))
    ##############bi_tempered_logistic_loss#############################################################################

    loss = tf.reduce_mean(per_example_loss)

    return (loss, per_example_loss, logits, probabilities)

def layer_norm(input_tensor, name=None):
  """Run layer normalization on the last dimension of the tensor."""
  return tf.contrib.layers.layer_norm(
      inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)

def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    label_ids = features["label_ids"]
    is_real_example = None
    if "is_real_example" in features:
      is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
    else:
      is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    (total_loss, per_example_loss, logits, probabilities) = create_model(
        bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
        num_labels, use_one_hot_embeddings)

    tvars = tf.trainable_variables()
    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:

      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)
    elif mode == tf.estimator.ModeKeys.EVAL:

      def metric_fn(per_example_loss, label_ids, logits, is_real_example):
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        accuracy = tf.metrics.accuracy(
            labels=label_ids, predictions=predictions, weights=is_real_example)
        loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
        return {
            "eval_accuracy": accuracy,
            "eval_loss": loss,
        }

      eval_metrics = (metric_fn,
                      [per_example_loss, label_ids, logits, is_real_example])
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)
    else:
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          predictions={"probabilities": probabilities},
          scaffold_fn=scaffold_fn)
    return output_spec

  return model_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def input_fn_builder(features, seq_length, is_training, drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  all_input_ids = []
  all_input_mask = []
  all_segment_ids = []
  all_label_ids = []

  for feature in features:
    all_input_ids.append(feature.input_ids)
    all_input_mask.append(feature.input_mask)
    all_segment_ids.append(feature.segment_ids)
    all_label_ids.append(feature.label_id)

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    num_examples = len(features)

    # This is for demo purposes and does NOT scale to large data sets. We do
    # not use Dataset.from_generator() because that uses tf.py_func which is
    # not TPU compatible. The right way to load data is with TFRecordReader.
    d = tf.data.Dataset.from_tensor_slices({
        "input_ids":
            tf.constant(
                all_input_ids, shape=[num_examples, seq_length],
                dtype=tf.int32),
        "input_mask":
            tf.constant(
                all_input_mask,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "segment_ids":
            tf.constant(
                all_segment_ids,
                shape=[num_examples, seq_length],
                dtype=tf.int32),
        "label_ids":
            tf.constant(all_label_ids, shape=[num_examples], dtype=tf.int32),
    })

    if is_training:
      d = d.repeat()
      d = d.shuffle(buffer_size=100)

    d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
    return d

  return input_fn

# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer):
  """Convert a set of `InputExample`s to a list of `InputFeatures`."""

  features = []
  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

    feature = convert_single_example(ex_index, example, label_list,
                                     max_seq_length, tokenizer)

    features.append(feature)
  return features
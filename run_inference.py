import os, sys, time, json
start_time = time.time()
import tensorflow as tf
import numpy as np
import classifier_utils
import tokenization

tf.get_logger().setLevel('ERROR')

flags = tf.flags

FLAGS = flags.FLAGS

def to_int32(int_list):
  return [ np.int32(d) for d in int_list]

def np_json_convertor(obj):
  if type(obj).__module__ == np.__name__:
      if isinstance(obj, np.ndarray):
          return obj.tolist()
      else:
          return obj.item()
  raise TypeError('Unknown type:', type(obj))

def do_inference(model_path, vocab_file, data):
    tokenizer = tokenization.FullTokenizer(
      vocab_file=vocab_file, do_lower_case=True)
    interpreter = tf.lite.Interpreter(model_path=model_path)
    results = []

    for input in data:
        guid = "predict_1"
        text_a = tokenization.convert_to_unicode(input)
        example = classifier_utils.InputExample(guid=guid, text_a=text_a, text_b=None, label="1")
        feature = classifier_utils.convert_single_example(0, example, ["0", "1"], 128, tokenizer)
        
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        interpreter.set_tensor(input_details[0]['index'], [to_int32(feature.input_ids) if input_details[0]['dtype'] == np.int32 else feature.input_ids])
        interpreter.set_tensor(input_details[1]['index'], [to_int32(feature.input_mask) if input_details[1]['dtype'] == np.int32 else feature.input_ids])
        interpreter.set_tensor(input_details[3]['index'], [to_int32(feature.segment_ids) if input_details[3]['dtype'] == np.int32 else feature.input_ids])
        
        interpreter.invoke()

        output_data = interpreter.get_tensor(output_details[0]['index'])
        results.append(output_data[0])
    return results

def main(_):
    results = do_inference(FLAGS.tflite_model, FLAGS.vocab_file, FLAGS.text)
    
    results = { "status": "OK", "data": results, "taken": time.time() - start_time }
    print(json.dumps(results, default=np_json_convertor, indent=4))

if __name__ == "__main__":
    flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")
    flags.DEFINE_string("tflite_model", None,
        "The directory where the saved model to be loaded.")
    flags.DEFINE_multi_string("text", None,
                        "The text data to do predictions.")
    flags.mark_flag_as_required("tflite_model")
    flags.mark_flag_as_required("text")
    flags.mark_flag_as_required("vocab_file")
    tf.app.run()
import json
from run_inference import do_inference, np_json_convertor

def spam_classifier(request):
    status = "FAILURE"
    results = None
    request_json = request.get_json()
    if request.args and 'message' in request.args:
        pass
    elif request_json and 'data' in request_json:
        status = "OK"
        results = do_inference("models/spam_20210119_128.tflite", "albert_config/vocab.txt", request_json['data'])
    return json.dumps({
        "status": status,
        "data": results
    }, default=np_json_convertor, indent=4)
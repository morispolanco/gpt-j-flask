from flask import Flask, request
import json
from .lib.model import GPTJModelWrapper
import time
import torch

def create_app(config_name):
    app = Flask(__name__)

    min_vram = 16 * 1024 * 1024 * 1024 # 16GB
    if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory > min_vram:
        device = torch.device("cuda")
        # free up some space in gpu
        torch.cuda.empty_cache()
    else:
        device = torch.device("cpu")

    wrapper = GPTJModelWrapper(device, is_compressed=False, revision="main")
    model = wrapper.model
    tokenizer = wrapper.tokenizer

    @app.route("/", methods=['GET'])
    def handle():
        json_string = json.dumps({"success": True})
        return json_string.encode(encoding='utf_8')

    @app.route("/completion", methods=['POST'])
    def word_completion_handle():

        compute_start_time = time.time()

        text = request.json['text']
        min_length = request.json['min_length'] or "128"
        max_length = request.json['max_length'] or "128"
        temperature = request.json['temperature'] or "0.5"
        top_p = request.json['top_p'] or "0.9"

        prompt = tokenizer(text, return_tensors='pt')
        prompt = {key: value.to(device) for key, value in prompt.items()}
        out = model.generate(**prompt, min_length=int(min_length), max_length=int(
            max_length), do_sample=True, temperature=float(temperature), top_p=float(top_p), use_cache=True)

        compute_end_time = time.time()
        compute_time = compute_end_time - compute_start_time

        json_string = json.dumps(
            {"success": True, "compute_time": compute_time, "completion": tokenizer.decode(out[0])})
        return json_string.encode(encoding='utf_8')

    @app.cli.command()
    def test():
        import unittest
        import sys

        tests = unittest.TestLoader().discover("tests")
        result = unittest.TextTestRunner(verbosity=2).run(tests)
        if result.errors or result.failures:
            sys.exit(1)

    return app

import os
import torch
import time
from transformers import AutoTokenizer

class GPTJModelWrapper():

    def __init__(self, device, is_compressed, revision="float16"):
        print('initialising model on ' + device.type + '...')
        print("loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
         
        if(is_compressed):
            print("loading compressed model...")
            from .custom import GPTJForCausalLM
            self.model = GPTJForCausalLM.from_pretrained(
                "hivemind/gpt-j-6B-8bit", low_cpu_mem_usage=True)
        else:
            from transformers import AutoModelForCausalLM
            file_path = 'saved_models/gpt-j-6B-' + revision            

            if(os.path.isdir(file_path)):
                print('loading model from disk... ' + file_path)
                # load from disk
                self.model = AutoModelForCausalLM.from_pretrained(file_path)
                print('finished loading...')                
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    "EleutherAI/gpt-j-6B", revision=revision, low_cpu_mem_usage=True)
                self.model.save_pretrained(file_path)

        self.model.to(device)

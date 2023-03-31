#!/usr/bin/env python

"""
Name: main.py
Date: 2023-03-31 12:43:27
"""

import openai

OPENAI_API_KEY="sk-xx"

openai.api_key = OPENAI_API_KEY

class Template:
    def __init__(self):
        pass
    
    def get_response(self, line):
        completion = openai.ChatCompletion.create( model="gpt-3.5-turbo", messages=[{"role": "user", "content": line}])
        resp = completion.choices[0].message
        print(resp)
        return resp

    def get_response_stream(self, line):
        model_name = "gpt-3.5-turbo-0301"
        messages=[ {"role": "system", "content": "You are a helpful assistant."},]
        messages = messages + [{"role": "user", "content": line}]
        print(messages)
        response = openai.ChatCompletion.create(model=model_name, messages=messages, stream=True)
        res = ''
        print("The asnswer is:")
        for chunk in response:
            r =  chunk['choices'][0]['delta']
            if 'content' not in r: continue
            res += r['content']
            messages = messages + [{"role": "assistant", "content": res}]
        return res

    def forward(self):
        input_message = 'Hello, I get a fever'
        resp = self.get_response(input_message)
        print(resp)
        
if __name__ == '__main__':
    template = Template()
    template.forward()
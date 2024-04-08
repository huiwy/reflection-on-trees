import openai
import time
import httpx
from requests_futures.sessions import FuturesSession


def query_gpt4(messages):
    for i in range(5):
        try:
            response = openai.ChatCompletion.create(
                messages=messages, temperature=0.7, n=1, model="gpt-4-turbo")
            
            return response['choices'][0]['message']['content']
        except Exception as e:
            print(e, response)
            time.sleep(5)

    raise e

def query_chatgpt(messages):
    for i in range(5):
        try:
            response = openai.ChatCompletion.create(
                messages=messages,
                n=1,
                temperature=0.7,
                model="gpt-3.5-turbo-1106"
            )
            return response['choices'][0]['message']['content']
        except Exception as e:
            print(e, response)
            time.sleep(10)
    raise e

def query_mixtral(messages, n=1):
    if n == 1:
        return httpx.post('http://0.0.0.0:23100/v1/chat/completions', json={
            'messages': messages, 
            "model": 'mixtral', 
            'n': n, 
            'temperature': 0.7
        }, timeout=60).json()['choices'][0]['message']['content'].strip()
    else:
        return [c['message']['content'].strip() for c in httpx.post('http://0.0.0.0:23100/v1/chat/completions', json={
            'messages': messages, 'temperature': 0.7, "model": 'mixtral', 'n': n}, timeout=60).json()['choices']]

def query_multiple_mixtral(messages):
    session = FuturesSession()
    futures = []
    for m in messages:
        futures.append(session.post('http://0.0.0.0:23100/v1/chat/completions', json={'messages': m, 'temperature': 0.7, "model": 'mixtral'}, timeout=60))
    
    res = []
    for f in futures:
        res.append(f.result().json()['choices'][0]['message']['content'].strip())

    return res

def join_bargining_history(history):
    history = '\n'.join(['{}: {}'.format('Buyer' if i % 2 == 0 else 'Seller', m) for i, m in enumerate(history)])
    return history
import openai
import time

def query_gpt4(messages):
    for _ in range(5):
        try:
            response = openai.ChatCompletion.create(messages=messages, n=1, model="gpt-4-turbo-1106")
            return response['choices'][0]['message']['content']
        except Exception as e:
            print(e, response)
            time.sleep(15)
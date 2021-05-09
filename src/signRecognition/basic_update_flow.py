import time
from multiprocessing.dummy import Pool
import requests

pool = Pool()

response =None
ok = 0
timer=0

def on_success(r):
    global ok
    time.sleep(10)
    print('Post requests succes')
    ok = 1


def on_error(ex):
    global ok
    print('Post requests failed')
    ok = 0


def call_api(url, data, headers):
    global response
    response=requests.post(url=url, data=data, headers=headers)


def pool_processing_create(url, data, headers):
    pool.apply_async(call_api, args=[url, data, headers],
                     callback=on_success, error_callback=on_error)

timestamp1 = time.time()
i=0
pool_processing_create("https://127.0.0.1:8000/check", None, [])
while response is None:
    i+=1
    timer = time.time()
if response is None:
    print(timer-timestamp1)

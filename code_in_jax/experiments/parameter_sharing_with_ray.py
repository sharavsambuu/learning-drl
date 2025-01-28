# References :
#  - https://bair.berkeley.edu/blog/2018/01/09/ray/
# Notes : 
#   unicode-оор бичсэн коммент юм уу string remote доторхи 
#   тодорхойлолт байх ёсгүй одоогоор энэ ray-н алдаа юм шиг байна.
import ray

ray.init()

@ray.remote
class ParameterServer(object):
    def __init__(self, keys, values):
        self.parameters = dict(zip(keys, values))
    def get(self, keys):
        return [self.parameters[key] for key in keys]
    def update(self, keys, values):
        print('parameter updated : ', values)
        for key, value in zip(keys, values):
            self.parameters[key] = value

parameter_server = ParameterServer.remote(['hello_key'], [1])

some_global_key  = "hello_key"
@ray.remote
def worker_task(parameter_server):
    for i in range(10):
        keys       = [some_global_key]
        values     = ray.get(parameter_server.get.remote(keys))
        values[0] += 1
        parameter_server.update.remote(keys, values)

# 5-н ширхэг даалгавар ажиллуулах
for _ in range(5):
    ray.get(worker_task.remote(parameter_server))

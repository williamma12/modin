import ray


@ray.remote
class NodeActor(object):
    def run(self, func, *args):
        return func(*args)

    def receive(self, obj):
        return True


cluster_resources = ray.global_state.cluster_resources()
nodes = [
    node
    for node in cluster_resources.keys()
    if "CPU" not in node and "HEAD" not in node
]

all_actors = []
for resource in nodes:
    for _ in range(int(cluster_resources[resource])):
        all_actors.append(
            NodeActor._remote([], {}, num_cpus=0.01, resources={resource: 1})
        )


def get_available_actors(n_nodes):
    n_actors = int(n_nodes * cluster_resources[nodes[0]])
    return all_actors[:n_actors]

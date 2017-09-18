import edward as ed
import numpy as np
import tensorflow as tf

from edward.models import Bernoulli, Multinomial, Beta, Dirichlet, PointMass


def read_data(data_path):
    name2id = {}
    with open(data_path, 'r') as f:
        subs = []
        objs = []
        for line in f:
            sub, obj = line.strip().split()
            if sub not in name2id:
                name2id[sub] = len(name2id)
            if obj not in name2id:
                name2id[obj] = len(name2id)
            subs.append(name2id[sub])
            objs.append(name2id[obj])
    adj_mat = np.zeros((len(name2id), len(name2id)))
    adj_mat[subs, objs] = 1
    return name2id, adj_mat


class SBM(object):
    def __init__(self, n_cluster):
        self.n_cluster = n_cluster

    def run(self, adj_mat, n_iter=1000):
        assert adj_mat.shape[0] == adj_mat.shape[1]
        n_node = adj_mat.shape[0]

        # model
        gamma = Dirichlet(concentration=tf.ones([self.n_cluster]))
        Pi = Beta(concentration0=tf.ones([self.n_cluster, self.n_cluster]),
                  concentration1=tf.ones([self.n_cluster, self.n_cluster]))
        Z = Multinomial(total_count=1., probs=gamma, sample_shape=n_node)
        X = Bernoulli(probs=tf.matmul(Z, tf.matmul(Pi, tf.transpose(Z))))

        # inference (point estimation)
        qgamma = PointMass(params=tf.nn.softmax(tf.Variable(tf.random_normal([self.n_cluster]))))
        qPi = PointMass(params=tf.nn.sigmoid(tf.Variable(tf.random_normal([self.n_cluster, self.n_cluster]))))
        qZ = PointMass(params=tf.nn.softmax(tf.Variable(tf.random_normal([n_node, self.n_cluster]))))

        # map estimation
        inference = ed.MAP({gamma: qgamma, Pi: qPi, Z: qZ}, data={X: adj_mat})
        inference.initialize(n_iter=n_iter)

        tf.global_variables_initializer().run()

        for _ in range(inference.n_iter):
            info_dict = inference.update()
            inference.print_progress(info_dict)
        inference.finalize()
        return qZ.mean().eval().argmax(axis=1)

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--cls', type=int)
    args = p.parse_args()

    print('read data...')
    name2id, data = read_data('./digraph.dat')
    id2name = np.array([k for k, v in sorted(name2id.items(), key=lambda x: x[1])])
    n_cluster = args.cls
    m = SBM(n_cluster=n_cluster)
    print('running...')
    res = m.run(data)

    for i in range(n_cluster):
        print('class {:3d}: {}'.format(i+1, id2name[np.where(res==i)]))

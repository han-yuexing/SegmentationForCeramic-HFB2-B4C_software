import numpy
import __future__
from sklearn import cluster

# normalization
def max_min_normalization(data_value):
    data_shape = data_value.shape
    data_rows = data_shape[0]
    data_cols = data_shape[1]
    data_col_max_values = data_value.max(axis=0)
    data_col_min_values = data_value.min(axis=0)

    for i in xrange(0, data_rows, 1):
        for j in xrange(0, data_cols, 1):
            data_value[i][j] = \
                (data_value[i][j] - data_col_min_values[j]) / \
                (data_col_max_values[j] - data_col_min_values[j])


num = numpy.loadtxt('results\\regionsNum.txt')
b = numpy.loadtxt('results\\bars_feature.txt')
p = numpy.loadtxt('results\\parameters.txt');

features = []
# print(b.shape[0],b.shape[1])
for i in xrange(0, b.shape[0], 1):
    if b[i,0] == p[0]:
        features.append([b[i,1],b[i,2],b[i,3],b[i,4]])

nuFeatures = numpy.array(features)
max_min_normalization(nuFeatures)

c=nuFeatures[:,0:2]
y_pred = cluster.DBSCAN(eps=p[1], min_samples=p[2]).fit(c)   # 0.05+20;0.01+20;0.05+10;0.1+20

# output labels
txtname="results\\labels%d.txt"%(p[0])
numpy.savetxt(txtname, y_pred.labels_, fmt='%d')

# output number of class
n_clusters_ = len(set(y_pred.labels_))
s_n_clusters_ = "%d"%(n_clusters_)
txtname2="results\\nLabels%d.txt"%(p[0])
file =open(txtname2,'w')
file.write(s_n_clusters_)
file.close()

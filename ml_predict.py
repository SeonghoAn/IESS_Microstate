import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
import argparse
from sklearn.linear_model import LogisticRegression
from utils import load_data, seed_everything, conf_eval, kfold_cv_fs_mi, mibif_kfold

plt.rcParams['font.family'] = 'Times New Roman'
parser = argparse.ArgumentParser(description='vaccine_prediction')
parser.add_argument('--cate', type=str, default='eti')
parser.add_argument('--n', type=int, default=40)
parser.add_argument('--seed', type=int, default=42)
args = vars(parser.parse_args())

seed_everything(args['seed'])
map_dict = {'etiology': 'eti', 'Intractable epilepsy = 1': 'ie', 'Infantile Spasms': 'is'}
map_dict = {v: k for k, v in map_dict.items()}

X, y = load_data('./feat.csv', map_dict[args['cate']])

model = LogisticRegression()

if args['cate'] == 'is':
    labels = ['HC', 'IESS']
elif args['cate'] == 'eti':
    labels = ['Non Struc-meta', 'Struc-meta']
elif args['cate'] == 'ie':
    labels = ['Controlled epilepsy', 'Intractable epilepsy']

n = args['n']
mi_score = mibif_kfold(X, y, k=5)
y_true, y_pred, y_prob = kfold_cv_fs_mi(X, y, model, n, mi_score, k=5)
conf = confusion_matrix(y_true, y_pred)
_, sens, spec = conf_eval(np.array(conf))
auc = roc_auc_score(y_true, y_prob[:, 1])
print('AUC: {}, Sensitivity: {}, Specificity: {}'.format(auc, sens, spec))

plt.figure(facecolor='white')
disp = ConfusionMatrixDisplay(confusion_matrix=conf, display_labels=labels)
disp.plot(cmap=plt.cm.gray_r).figure_.savefig('./results/conf/{}_{}.png'.format(args['cate'], args['model']),dpi=300, bbox_inches='tight')
plt.close()

plt.figure(facecolor='white')
disp = RocCurveDisplay.from_predictions(y_true, y_prob[:, 1], plot_chance_level=True)
disp.plot(color='black', plot_chance_level=True).figure_.savefig('./results/roc/{}_{}.png'.format(args['cate'], args['model']), dpi=300)
plt.close()
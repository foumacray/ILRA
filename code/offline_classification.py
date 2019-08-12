__author__ = 'pedroamarinreyes'

import numpy as np
import sys
import random
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from utils.transformations import Ilr

def modelar(datos, pert, muestras):
    datos_model = []
    gnb = GaussianNB()
    datos = np.array(datos)
    pert = np.array(pert)
    muestras = np.array(muestras)

    loo = LeaveOneOut()
    for train_index, test_index in loo.split(datos):
        X_train, X_test = datos[train_index], datos[test_index]
        y_train, y_test = pert[train_index], pert[test_index]

        gnb = gnb.fit(X_train, y_train)
        y = gnb.predict_proba(X_test)
        datos_model.append(y[0])

    gnb = gnb.fit(datos, pert)
    muestras_model_post = gnb.predict_proba(muestras)
    ######ilr
    datos_model = Ilr.transform(np.add(datos_model, 0.000000001))
    muestras_model = Ilr.transform(np.add(muestras_model_post, 0.000000001))
    return [datos_model, muestras_model, muestras_model_post]

if __name__ == '__main__':
    in_path = sys.argv[1]
    out_path = sys.argv[2]

    in_file = open(in_path, 'r')
    out_file = open(out_path, 'w')

    relations = []
    lista=[]
    aux=''
    for line in in_file:
        if line.find(";") == -1:
           aux += line[:len(line)-1]
           continue
        [model_str, dum, dum2, name] = line.split(';')
        model = np.fromstring(aux[1:]+model_str[:len(model_str)-1], dtype=float, sep=' ')
        aux = ''
        if name.find('Empty') > -1:
            continue
        if relations.count(name) == 0:
            relations.append(name)
            lista.append([])
        lista[int(relations.index(name))].append(model)
    #for v in lista:
    #   print(v[0])
    #   print('------')
    in_file.close()

    iterations = 100
    acc_svm = 0
    acc_apost = 0
    for ite in range(0, iterations):
        lista_aTratar = []
        pert = []
        lista_samples = np.zeros(len(lista), dtype=object)
        for i in range(0, len(lista)):
            if len(lista[i]) > 500:
                lista_samples[i] = random.sample(lista[i], 500)
            else:
                lista_samples[i] = lista[i]
            for item in lista_samples[i]:
                    lista_aTratar.append(item)
                    pert.append(i + 1)
	for v in lista_samples:
           print(len(v))
        X_train, X_test, y_train, y_test = train_test_split(lista_aTratar, pert, test_size=0.4, random_state=0)
        [datos, muestras,muestras_posteriori] = modelar(X_train, y_train, X_test)
        print(muestras_posteriori[0])
        from sklearn import svm
        clf = svm.SVC()
        clf.fit(datos, y_train)
        predictions = clf.predict(muestras)
        acc_svm += accuracy_score(y_test, predictions)

        post_predictions = []
        for i in range(0, len(y_test)):
            post_predictions.append(np.argmax(muestras_posteriori[i]) + 1)
        acc_apost += accuracy_score(y_test, post_predictions)
    acc_svm = acc_svm / iterations
    acc_apost = acc_apost / iterations

    out_file.write('Classifier nClases trainSize testSize accuracy\n')
    out_file.write('A-posteriori ' + str(len(lista_samples)) + ' ' + str(len(X_train)) + ' ' + str(len(X_test)) + ' ' +str(acc_apost) + '\n')
    out_file.write('SVM '+ str(len(lista_samples)) + ' ' + str(len(X_train)) + ' ' + str(len(X_test)) + ' '  + str(acc_svm) + '\n')
    out_file.close()

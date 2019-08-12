__author__ = 'pedroamarinreyes'

import numpy as np
import sys
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import LeaveOneOut
import random
from utils.transformations import Ilr

def modelar(datos, pert, muestras):
    datos_model = []
    muestras_model = []
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
    muestras_model = gnb.predict_proba(muestras)
    ######ilr
    #transformation = Ilr.transform(np.concatenate((datos_model,muestras_model)))
    datos_model=Ilr.transform(np.add(datos_model,0.000000001))
    muestras_model=Ilr.transform(np.add(muestras_model,0.000000001))
    return [datos_model, muestras_model]
    #return transformation[0:len(datos_model)], transformation[len(datos_model):]


def procesarAtipicos():
    global trueAtipicos, tipicidad, atipicidad, i, ite, lista_samples, datos, pert, item, muestras, atipicos, tipicos
    for i in range(0, len(tipicidad)):
        tipicidad[i] = []
        atipicidad[i] = []
    for ite in range(0, iterations):
        lista_samples = np.zeros(len(lista), dtype=object)
        for i in range(0, len(lista)):
            if len(lista[i]) > 500:
                lista_samples[i] = random.sample(lista[i], 500)
            else:
                lista_samples[i] = lista[i]

        for i in range(0, len(lista_samples)):
            datos = []
            pert = []
            index_count = 0
            for j in range(0, len(lista_samples)):
                if j != i:
                    index_count += 1
                    for item in lista_samples[j]:
                        datos.append(item)
                        pert.append(index_count)
            # clasificador bayes
            [datos, muestras] = modelar(datos, pert, lista_samples[i])

            atipicos = 0
            tipicos = 0

            from sklearn import svm
            clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
            clf.fit(datos)
            predictions = clf.predict(muestras)
            for val in predictions:
                if val == -1:
                    atipicos += 1
                else:
                    tipicos += 1

            tipicidad[i].append(tipicos)
            atipicidad[i].append(atipicos)
    tipicos = 0
    atipicos = 0
    trueAtipicos = 0
    for i in range(0, tipicidad.shape[0]):
        atipicidad[i] = round(np.mean(atipicidad[i]))
        tipicidad[i] = round(np.mean(tipicidad[i]))
        if atipicidad[i] == 0 and tipicidad[i] == 0:
            continue
        prob = float(atipicidad[i]) / (tipicidad[i] + atipicidad[i])
        if prob >= umbral:
            trueAtipicos += 1

if __name__ == '__main__':
    in_path = sys.argv[1]
    out_path = sys.argv[2]
    umbral = 0.5
    iterations = 100
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

    in_file.close()

    tipicidad = np.zeros(len(lista), dtype=object)
    atipicidad = np.zeros(len(lista), dtype=object)
    procesarAtipicos()
    for i in range(0, len(tipicidad)):
        tipicidad[i] = []
        atipicidad[i] = []

    for ite in range(0, iterations):
        lista_aTratar = []
        muestras_aTratar = []
        pert = []
        lista_samples = np.zeros(len(lista), dtype=object)
        for i in range(0, len(lista)):
            if len(lista[i]) > 500:
                lista_samples[i] = random.sample(lista[i], 500)
            else:
                lista_samples[i] = lista[i]
            cut = len(lista_samples[i]) / 3.
            for k in range(0, len(lista_samples[i])):
                item = lista_samples[i][k]
                if k >= cut:
                    lista_aTratar.append(item)
                    pert.append(i + 1)
                else:
                    muestras_aTratar.append(item)

        [datos, muestras] = modelar(lista_aTratar, pert, muestras_aTratar)

        from sklearn import svm
        clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
        clf.fit(datos)
        predictions = clf.predict(muestras)

        atipicos = 0
        tipicos = 0
        ini = 0
        for i in range(0, len(lista_samples)):
            for val in predictions[ini : ini+int(len(lista_samples[i])/3)+1]:
                if val == -1:
                    atipicos += 1
                else:
                    tipicos += 1
            ini = ini + int(len(lista_samples[i])/3)
            tipicidad[i].append(tipicos)
            atipicidad[i].append(atipicos)
            tipicos = 0
            atipicos = 0

    trueTipicos=0
    for i in range(0, tipicidad.shape[0]):
        atipicidad[i] = round(np.mean(atipicidad[i]))
        tipicidad[i] = round(np.mean(tipicidad[i]))

        prob = float(tipicidad[i]) / (tipicidad[i] + atipicidad[i])
        if prob > 1 - umbral:
            trueTipicos += 1

    perAtipico = float(trueAtipicos) / len(atipicidad)
    perTipico = float(trueTipicos) / len(tipicidad)
    acc = (perTipico+perAtipico) / (perTipico+perAtipico+(1-perTipico)+(1-perAtipico))

    print('umbral numClasses numCompTipico aciertoTipico numCompAtipico aciertoAtipico accuracy')
    out_file.write('umbral numClasses numCompTipico aciertoTipico numCompAtipico aciertoAtipico accuracy\n')
    print(str(umbral)+' '+str(len(atipicidad))+' '+str(len(tipicidad))+' '+str(trueTipicos)+' '+str(len(atipicidad))+' '+str(trueAtipicos)+' '+str(acc))
    out_file.write(str(umbral)+' '+str(len(atipicidad))+' '+str(len(tipicidad))+' '+str(trueTipicos)+' '+str(len(atipicidad))+' '+str(trueAtipicos)+' '+str(acc)+'\n')

    out_file.close()

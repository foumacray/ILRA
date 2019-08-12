__author__ = 'pedroamarinreyes'

import numpy as np
import sys
import random

if __name__ == '__main__':
    in_path = sys.argv[1]
    out_path = sys.argv[2]

    in_file = open(in_path, 'r')
    out_file = open(out_path, 'w')
    umbral=0.5

    relations = []
    lista=[]
    aux = ''
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

    tipicidad = np.zeros((len(lista), len(lista)), dtype=object)
    atipicidad = np.zeros((len(lista), len(lista)), dtype=object)
    for i in range(0, tipicidad.shape[0]):
        for j in range(0, tipicidad.shape[1]):
            tipicidad[i,j] = []
            atipicidad[i,j] = []

    iterations = 100
    for ite in range(0, iterations):

        lista_samples = np.zeros(len(lista), dtype=object)
        for i in range(0, len(lista)):
            if len(lista[i]) > 500:
                lista_samples[i] = random.sample(lista[i], 500)
            else:
                lista_samples[i] = lista[i]

        for i in range(0, len(lista_samples)):
            for j in range(i, len(lista_samples)):
                atipicos = 0
                tipicos = 0

                from sklearn import svm
                clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
                clf.fit(lista_samples[i])
                predictions = clf.predict(lista_samples[j])#lista_samples[j])
                for val in predictions:
                    if val == -1:
                        atipicos += 1
                    else:
                        tipicos += 1

                tipicidad[i,j].append(tipicos)
                atipicidad[i,j].append(atipicos)
    #parte para estadisticas y mostrar pantalla
    ncomp_atipicos = 0
    tipicos = 0
    atipicos = 0
    for i in range(0, tipicidad.shape[0]):
        for j in range(i, tipicidad.shape[1]):
            atipicidad[i][j] = round(np.mean(atipicidad[i][j]))
            tipicidad[i][j] = round(np.mean(tipicidad[i][j]))
            if i != j:
                ncomp_atipicos += 1
                prob = float(atipicidad[i][j])/(tipicidad[i][j]+atipicidad[i][j])
                if prob >=umbral:
                    atipicos += 1
            else:
                prob = float(tipicidad[i][j])/(tipicidad[i][j]+atipicidad[i][j])
                if prob > 1-umbral:
                    tipicos += 1
    #--------------------------------
    perAtipico = float(atipicos)/ncomp_atipicos
    perTipico = float(tipicos)/len(tipicidad)
    #perAtipico = float("{0:.2f}".format(perAtipico))
    #--------------------------------
    acc=(perTipico+perAtipico)/(perTipico+perAtipico+(1-perTipico)+(1-perAtipico))
    print('umbral numClasses numCompTipico aciertoTipico numCompAtipico aciertoAtipico accuracy')
    out_file.write('umbral numClasses numCompTipico aciertoTipico numCompAtipico aciertoAtipico acuracy\n')
    print(str(umbral)+' '+str(len(tipicidad))+' '+str(len(tipicidad))+' '+str(tipicos)+' '+str(ncomp_atipicos)+' '+str(atipicos)+' '+str(acc))
    out_file.write(str(umbral)+' '+str(len(tipicidad))+' '+str(len(tipicidad))+' '+str(tipicos)+' '+str(ncomp_atipicos)+' '+str(atipicos)+' '+str(acc)+'\n')

    out_file.close()

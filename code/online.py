import sys
import random
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import LeaveOneOut
from utils.transformations import Ilr
from itertools import groupby

__author__ = 'pedro'

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


def procesark1_atipicidad(lista):
    global umbral, sizeShot, iterations
    tipicidad = []
    atipicidad = []
    print()
    for ite in range(0, iterations):
        lista_samples = np.zeros(len(lista), dtype=object)
        for i in range(0, len(lista)):
            if len(lista[i]) > sizeShot:
                lista_samples[i] = random.sample(lista[i], sizeShot)
            else:
                lista_samples[i] = lista[i]

        for i in range(0, len(lista_samples)):
            for j in range(i+1, len(lista_samples)):
                atipicos = 0
                tipicos = 0
                from sklearn import svm
                clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
                clf.fit(lista_samples[i])
                predictions = clf.predict(lista_samples[j])
                for val in predictions:
                    if val == -1:
                        atipicos += 1
                    else:
                        tipicos += 1

                tipicidad.append(tipicos)
                atipicidad.append(atipicos)

    atipicidad = int(np.mean(atipicidad))
    tipicidad = int(np.mean(tipicidad))

    prob_atipico = float(atipicidad) / (tipicidad+atipicidad)
    if prob_atipico >= umbral:
        atipico = True
    else:
        atipico = False
    return [atipico, prob_atipico]

    #--------------------------------
    #acc=(perTipico+perAtipico)/(perTipico+perAtipico+(1-perTipico)+(1-perAtipico))
    #print(acc)


def procesaOneVsOne():
    global data, model, data_to_proc, k
    data.append(data_to_proc)
    [atipico, prob] = procesark1_atipicidad(data)
    if atipico:
        ids_real.append(relations.index(ultimo_speaker))
        if len(data[1]) > sizeShot:
            data[1] = random.sample(data[1], sizeShot)
        ids_pred.append(1)
    else:
        data.pop(1)
        if len(data_to_proc) > sizeShot:
            data_to_proc = random.sample(data_to_proc, sizeShot)
        for model in data_to_proc:
            data[0].append(model)
        ids_real.append(relations.index(ultimo_speaker))
        ids_pred.append(0)
        k -= 1
    data_to_proc = []


def procesar_kmayor1():
    global k, model, data_to_proc
    tipicidad = []
    atipicidad = []
    #print(len(data[-1]))
    #print(len(data_to_proc))
    for ite in range(0, iterations):
        lista_aTratar = []
        muestras_aTratar = []
        pert = []
        lista_samples = np.zeros(len(data), dtype=object)
        for i in range(0, len(data)):
            if len(data[i]) > sizeShot:
                lista_samples[i] = random.sample(data[i], sizeShot)
            else:
                lista_samples[i] = data[i]
            for k in range(0, len(lista_samples[i])):
                lista_aTratar.append(lista_samples[i][k])
                pert.append(i)


        [datos, muestras] = modelar(lista_aTratar, pert, data_to_proc)
        from sklearn import svm
        clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
        clf.fit(datos)
        predictions = clf.predict(muestras)
        atipicos = 0
        tipicos = 0
        for val in predictions:
            if val == -1:
                atipicos += 1
            else:
                tipicos += 1
        tipicidad.append(tipicos)
        atipicidad.append(atipicos)
    atipicidad = int(np.mean(atipicidad))
    tipicidad = int(np.mean(tipicidad))
    prob_atipico = float(atipicidad) / (tipicidad + atipicidad)
    #print('prob. atipico: ' + str(prob_atipico))
    #print('tamanio muestras ' + str(len(data[-1][0])))
    if prob_atipico >= umbral:
        atipico = True
        ids_real.append(relations.index(ultimo_speaker))
        ids_pred.append(pert[-1] + 1)
        data.append(data_to_proc)
    else:
        cla = svm.SVC()
        cla.fit(datos, pert)
        predictions = cla.predict(muestras)
        group = groupby(predictions)
        clase_pred = max(group, key=lambda r: len(list(r[1])))[0]

        for model in data_to_proc:
            data[clase_pred].append(model)
        ids_real.append(relations.index(ultimo_speaker))
        ids_pred.append(clase_pred)
    data_to_proc = []

def preparar_conteo_F(real, pred):
    lista_equivalencias_resultados = []
    lista_equivalencias_reales = []
    numMuestrasIguales = 0
    for i in range(0, len(real)):
        for j in range(0, len(real)):
            if i < j:
                if pred[i] == pred[j]:
                    lista_equivalencias_resultados.append(True)
                else:
                    lista_equivalencias_resultados.append(False)
                if real[i] == real[j]:
                    lista_equivalencias_reales.append(True)
                    numMuestrasIguales += 1
                else:
                    lista_equivalencias_reales.append(False)
    return [lista_equivalencias_reales, lista_equivalencias_resultados, numMuestrasIguales]

def calcular_TRR(lista_equivalencias_reales, lista_equivalencias_resultados, numMuestrasIguales):
    cont = 0
    for index in range(0, len(lista_equivalencias_reales)):
        if lista_equivalencias_resultados[index] and lista_equivalencias_reales[index]:
            cont += 1
    return (float(cont)/numMuestrasIguales) * 100

def calcular_TDR(lista_equivalencias_reales, lista_equivalencias_resultados, numMuestrasIguales):
    cont = 0
    for index in range(0, len(lista_equivalencias_reales)):
        if not lista_equivalencias_resultados[index] and not lista_equivalencias_reales[index]:
            cont += 1
    return (float(cont)/(len(lista_equivalencias_reales)-numMuestrasIguales)) * 100

if __name__ == '__main__':
    in_path = sys.argv[1]
    out_path = sys.argv[2]
    umbral = 0.5
    iterations = 100
    sizeShot = 200
    in_file = open(in_path, 'r')


    relations = []
    data_to_proc = []
    plano_actual = -1
    k = 0
    ids_pred = []
    ids_real = []
    data=[]
    ultimo_speaker = ''
    aux=''
    for line in in_file:
        if line.find(";") == -1:
           aux += line[:len(line)-1]
           continue
        [model_str, plano, frame, name] = line.split(';')
        if plano_actual == -1 and name != 'Empty\r\n':
            plano_actual = int(plano)
        model = np.fromstring(aux[1:]+model_str[:len(model_str)-1], dtype=float, sep=' ')
        aux = ''
        if name.find('Empty') > -1:
            continue
        if relations.count(name) == 0:
            relations.append(name)

        print(name)

        if plano_actual != int(plano):
            print('real: '+ str(ids_real))
            print('pred: '+ str(ids_pred))
            print(ultimo_speaker)
            print(name)
            if len(data_to_proc) == 1:
                data_to_proc = []
                plano_actual = int(plano)
                continue
            k += 1
            if len(data_to_proc) > sizeShot: #aniadido nuevo
                data_to_proc = random.sample(data_to_proc, sizeShot)

            if k == 1:
                #print(len(data_to_proc))
                #sizeShot = len(data_to_proc)
                if len(data_to_proc) > sizeShot: #si el size del primer plano va quitar este if
                    data_to_proc = random.sample(data_to_proc, sizeShot)
                print(data_to_proc)
                print('ey')
                data.append(data_to_proc)
                data_to_proc = []
                ids_real.append(relations.index(ultimo_speaker))
                ids_pred.append(0)
            if k == 2:
                procesaOneVsOne()
            if k > 2:
                procesar_kmayor1()
            plano_actual = int(plano)


        data_to_proc.append(model)
        ultimo_speaker = name
    #if len(data_to_proc) > 0:
    #    procesar_kmayor1()
    out_file = open(out_path, 'w')
    out_file.write('real: ' + str(ids_real) + '\n')
    out_file.write('pred: ' + str(ids_pred) + '\n')
    out_file.write('tamanoPrimerPlano'+str(len(data[0]))+'\n')

    [lista_equivalencias_reales, lista_equivalencias_resultados, numMuestrasIguales] = preparar_conteo_F(ids_real, ids_pred)
    trr = calcular_TRR(lista_equivalencias_reales, lista_equivalencias_resultados, numMuestrasIguales)
    tdr = calcular_TDR(lista_equivalencias_reales, lista_equivalencias_resultados, numMuestrasIguales)
    out_file.write(str(trr) + ' ')
    out_file.write(str(tdr) + ' ')
    out_file.write(str(2*(trr*tdr)/(trr+tdr)) + '\n')
    #print(trr)
    #print(tdr)
    #print(2*(trr*tdr)/(trr+tdr))
    in_file.close()
    out_file.close()

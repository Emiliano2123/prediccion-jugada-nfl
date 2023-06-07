import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
import copy

df = pd.read_csv("./data/workingCopy.csv")
df = df.drop(columns=["Unnamed: 0"])

def obtenerTiempo(reloj, cuarto):
    return 3600 - (cuarto * 900) + reloj

def obtenerTiempoCuarto(reloj, cuarto):
  if(cuarto > 2):
    return obtenerTiempo(reloj, cuarto)
  else:
    return obtenerTiempo(reloj, cuarto + 2)

def conversionReloj(tiempo):
  aux = tiempo.replace("(", "")
  aux = aux.replace(")", "")
  auxL = aux.split(":")
  return int(auxL[0])*60 + int(auxL[1])

def hacerTiempos(data):
    tiempoAux = data[["gameClock", "quarter"]]
    tempList = np.zeros(len(tiempoAux))
    tempList2 = np.zeros(len(tiempoAux))
    for i, row in data.iterrows():
        minutos = conversionReloj(row["gameClock"])
        tempList[i] = obtenerTiempo(minutos, row["quarter"])
        tempList2[i] = obtenerTiempoCuarto(minutos, row["quarter"])
    data = data.drop(columns=["gameClock", "quarter"])
    data["time"] = tempList
    data["medioTiempo"] = tempList2
    return df

def hacerPuntos(data):
    puntosPosesion = [0]*len(data)
    puntosRival = [0]*len(data)
    diferencia = [0]*len(data)

    bla = data['homeScorePre'].tolist()
    blo = data['visitingScorePre'].tolist()

    i = 0
    while(i < len(bla)):
        if(bla[i] == 0 and blo[i] == 0):
            puntosPosesion[i] = 0
            puntosRival[i] = 0
        elif(bla[i] == bla[i-1] and blo[i] == blo[i-1]):
            puntosPosesion[i] = puntosPosesion[i-1] 
            puntosRival[i] = puntosRival[i-1]
        elif(bla[i] != bla[i-1]):
            puntosPosesion[i] = puntosRival[i-1]
            puntosRival[i] = bla[i]
        elif(blo[i] != blo[i-1]):
            puntosPosesion[i] = puntosRival[i-1]
            puntosRival[i] = blo[i]
        i = i + 1

    i = 0
    while (i < len(puntosRival)):
        diferencia[i] = puntosPosesion[i] - puntosRival[i]
        i = i + 1

    data = data.drop(columns=["homeScorePre", "visitingScorePre", "netYards"])
    #data["pointsOffense"] = puntosPosesion
    #data["pointsDefense"] = puntosRival
    data["diferenciaPuntos"] = diferencia
    return data

def renombrarEquipos(data):
    tempList = np.zeros(len(data))
    for i, row in data.iterrows():
        if(row["possessionTeamId"] == 610):
            tempList[i] = 1
        elif(row["possessionTeamId"] == 2700):
            tempList[i] = 2
        elif(row["possessionTeamId"] == 3200):
            tempList[i] = 3
        elif(row["possessionTeamId"] == 3430):
            tempList[i] = 4
        elif(row["possessionTeamId"] == 920):
            tempList[i] = 5
        elif(row["possessionTeamId"] == 325):
            tempList[i] = 6
        elif(row["possessionTeamId"] == 3900):
            tempList[i] = 7
        elif(row["possessionTeamId"] == 1050):
            tempList[i] = 8
        elif(row["possessionTeamId"] == 2250):
            tempList[i] = 9
        elif(row["possessionTeamId"] == 2100):
            tempList[i] = 10
        elif(row["possessionTeamId"] == 2200):
            tempList[i] = 11
        elif(row["possessionTeamId"] == 2120):
            tempList[i] = 12
        elif(row["possessionTeamId"] == 2310):
            tempList[i] = 13
        elif(row["possessionTeamId"] == 4400):
            tempList[i] = 14
        elif(row["possessionTeamId"] == 2520):
            tempList[i] = 15
        elif(row["possessionTeamId"] == 1400):
            tempList[i] = 16
        elif(row["possessionTeamId"] == 3700):
            tempList[i] = 17
        elif(row["possessionTeamId"] == 1200):
            tempList[i] = 18
        elif(row["possessionTeamId"] == 3410):
            tempList[i] = 19
        elif(row["possessionTeamId"] == 5110):
            tempList[i] = 20
        elif(row["possessionTeamId"] == 3000):
            tempList[i] = 21
        elif(row["possessionTeamId"] == 1540):
            tempList[i] = 22
        elif(row["possessionTeamId"] == 1800):
            tempList[i] = 23
        elif(row["possessionTeamId"] == 810):
            tempList[i] = 24
        elif(row["possessionTeamId"] == 4900):
            tempList[i] = 25
        elif(row["possessionTeamId"] == 750):
            tempList[i] = 26
        elif(row["possessionTeamId"] == 3300):
            tempList[i] = 27
        elif(row["possessionTeamId"] == 200):
            tempList[i] = 28
        elif(row["possessionTeamId"] == 4500):
            tempList[i] = 29
        elif(row["possessionTeamId"] == 4600):
            tempList[i] = 30
        elif(row["possessionTeamId"] == 2510):
            tempList[i] = 31
        elif(row["possessionTeamId"] == 3800):
            tempList[i] = 32
    data["possessionTeamId"] = tempList
    return data
    
    
#df = hacerTiempos(df)
df = hacerPuntos(df)
df = renombrarEquipos(df)
df = df[~df['playTypeDetailed'].str.contains('kneel|kickoff|xp|punt|penalty|field goal|two-point|spike|aborted', na=False)]
df.to_csv("procesado.csv")

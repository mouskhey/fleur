from data import *
import numpy as np 
import pdb
import pandas as pd

def kppv(k,test,data_train,train_label):
	predict = [] #liste des prédictions
	for indx in test.index:
		test_line = test.loc[indx]
		neighbors = find_kppv_neighbors(k,test_line,data_train) #on récupère les kppv
		neighbors['label'] = train_label.loc[neighbors['index']].reset_index(drop = True) #on récupère leur label

		labels_neighbors = pd.Series([0,0,0],index = [0,1,2],name = 'counter')
		labels_neighbors += neighbors['label'].value_counts()
		labels_neighbors = labels_neighbors.fillna(0).sort_values(ascending = False).reset_index(drop = False) #on récupère trié les classes représentées par les kppv

		if labels_neighbors.loc[0,'counter'] == labels_neighbors.loc[1,'counter'] : #Si il n'y a pas de majorité 
			neighbors['weight'] = 1/neighbors['l1'] #On pondère l'occurence de la classe de chaque fleur learn étudiées par l'inverse de la distance qui leur est associée
			final = neighbors[['weight','label']].groupby(['label']).sum().sort_values(by = 'weight', ascending = False) #On obtient les classes triés par la somme des 1/d
			predict.append(final.index[0]) #Il est également encore possible qu'il y ait égalité, soit prendre random soit prendre celui avec le nbr le plus faible
		else : 
			predict.append(labels_neighbors.loc[0,'index'])
		
	return predict

def metrics_kppv_k(k,return_pred = False):
	data_train = pd.DataFrame(IRIS_LEARN_DATA) #Data store in another fichier .py, if csv format use pd.read_csv, if .jpg use matplotlib.pyplot 
	data_test = pd.DataFrame(IRIS_TEST_DATA)
	label_test = pd.Series(IRIS_TEST_LABEL)
	label_train = pd.Series(IRIS_LEARN_LABEL)
	predicts = kppv(k,data_test,data_train,label_train)
	positive_nbr = (predicts-label_test).value_counts().loc[0] #on compte le nombre de bonne prédictions
	if return_pred:
		df = pd.concat([pd.Series(predicts),label_test], axis = 1)
		df.columns = ['prédiction','label']
		return df
	return positive_nbr/len(label_test)

def find_kppv_neighbors(k,l,neighbors):
	distance = (neighbors-l).abs().sum(axis = 1) #on calcule distance de manhattan pour chaque donnée de learn avec la donnée test
	distance = distance.sort_values() #on trie par ordre croissant
	distance.name = 'l1'
	distance = distance.reset_index(drop = False) #reset index pour pouvoir récupérer les index dans learn des 5 premier dans distance
	max_dist = distance.loc[k-1,'l1']
	return distance[distance['l1'] <= max_dist] #on récupère tous les voisins dont la distance qui les relis est au plus égal à celle du k-eme (on ne récupère pas nécessairement k voisins)

def metrics_kppv(n = 2,p = 10):
	evals = pd.Series(name = "Taux de succès")
	evals.index.name = 'k'
	for k in range(n,p+1):
		evals.loc[k] = metrics_kppv_k(k)
	return evals
if __name__ == '__main__':
	print("Voici les prédictions du k-ppv pour k = 3 :")
	print(metrics_kppv_k(3, return_pred = True))
	print()
	print("Celles pour k = 7 :")
	print(metrics_kppv_k(3, return_pred = True))
	print()
	print("Voici les taux de classifications :")
	print("k=3 ---> " + str(metrics_kppv_k(3)*100) + '%')
	print("k=7 ---> " + str(metrics_kppv_k(7)*100) + '%')
	print()
	print(metrics_kppv())
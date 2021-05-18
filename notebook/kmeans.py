import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import math


class KMeans:
    def __init__(self, k):
        self.k = k

    # On attribue des coordonnées aléatoires aux centroids qu'on veut
    
    def coord_aleatoires_centroids(self,X,k):
        '''On génère des coordonnées de centroïds aléatoires'''
        coords_centroids={}
        for i in range(k):
            coords_centroids[i]=(np.random.choice(X[:,0], 1,replace=False),np.random.choice(X[:,1], 1,replace=False))
        return coords_centroids
    
    # On attribue un cluster à chaque observation de X

    def calculateDistance(self,x1,y1,x2,y2):
        """fonction pour calculer la distance entre 2 points"""
        dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        return dist
    
    def prediction_y(self,X,coords_centroids):
        ''' On attribue à chaque observation de X un cluster dans un array y_pred'''
        y_pred=[]
        for row in X:
            dist_aux_centroids={}
            for key in coords_centroids: #on calcule la distance de l'observation aux 3 centroids
                dist_aux_centroids[key]=self.calculateDistance(row[0],row[1],coords_centroids[key][0],coords_centroids[key][1])

            # on stocke la prédiction du cluster de l'observation dans un numpy array
            for key,value in dist_aux_centroids.items(): # on récupère le centroid le plus proche
                if value == min(dist_aux_centroids[0],dist_aux_centroids[1],dist_aux_centroids[2]):
                    y_pred.append(key)
        return y_pred
    
    # On place les centroids au centre des clusters
    
    def centroids_deplaces_centre_clusters(self,X,y_pred,coords_centroids):
        """Déplace chaque centroid au centre de son cluster"""

        # on concatène les observations X avec leur prédiction y dans un dataframe
        concat_X_ypred=np.column_stack((X, y_pred))
        concat_X_ypred=pd.DataFrame(concat_X_ypred,columns=['X0','X1','y_pred'])

        # on convertit les y_pred (float) en int
        concat_X_ypred.y_pred=pd.to_numeric(concat_X_ypred.y_pred, downcast="integer")

        # on remplace les coordonnées précédentes des centroids par les coordonnées du centre de chaque cluster
        for key in coords_centroids:
            coords_centroids[key]=(concat_X_ypred[y_pred==key].mean().X0,concat_X_ypred[y_pred==key].mean().X1)
        return coords_centroids
            
    def fit_predict(self,X):
        coords_centroids=self.coord_aleatoires_centroids(X,self.k)
        for i in range(10):
            y_pred=self.prediction_y(X,coords_centroids)
            self.centroids_deplaces_centre_clusters(X,y_pred,coords_centroids)
        return coords_centroids,y_pred
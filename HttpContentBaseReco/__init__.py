import azure.functions as func
import logging
from sklearn.metrics.pairwise import linear_kernel, cosine_similarity
from enum import Enum
import numpy as np
import pandas as pd
from io import StringIO, BytesIO
import json
from collections import defaultdict
import surprise
import pickle

def main(req: func.HttpRequest, clickssamplecsv: bytes, articlemetadatacsv: bytes, dfpcareduitpickle: bytes, modelereco: bytes):
    data_json=req.get_json()
    dic_data = json.loads(data_json)

    id_user = int(dic_data['iduser'])
    type_reco = int(dic_data['paramreco'])
    selector_type = int(dic_data['paramref'])
    nb_reco1 = int(dic_data['nbreco1'])
    nb_reco2 = int(dic_data['nbreco2'])

    clicks_sample = pd.read_csv(StringIO(clickssamplecsv)) 
    article_metadata = pd.read_csv(StringIO(articlemetadatacsv))

    result={}

    if(type_reco in (1,3)): # Reco content type ou reco hybride
        df_pca_reduit = pd.read_pickle(BytesIO(dfpcareduitpickle))

        # Enumération des selections possibles d'articles
        class article_selector_type(Enum):
            FIRST_ARTICLE=1
            LAST_ARTICLE=2
            AVG_ARTICLE=3

        # Fonction qui renvoie le profil du premier article cliqué par un utilisateur
        def get_first_article(user_id):
            num_article=clicks_sample[clicks_sample["user_id"]==user_id].sort_values(by="click_timestamp",ascending=True).head(1)["click_article_id"]
            return (num_article,df_pca_reduit.loc[num_article,:])

        # Fonction qui renvoie le profil du dernier article cliqué par un utilisateur
        def get_last_article(user_id):
            num_article=clicks_sample[clicks_sample["user_id"]==user_id].sort_values(by="click_timestamp",ascending=False).head(1)["click_article_id"]
            return (num_article,df_pca_reduit.loc[num_article,:])

        #Fonction qui renvoie le profil moyen des articles pondérés par le nombre de clics de l'utilisateur
        def get_avg_article(user_id):
            arr_article=[]
            i=0
            for article in clicks_sample[clicks_sample["user_id"]==user_id]["click_article_id"]:
                if(i==0):
                    arr_article = [np.array(df_pca_reduit.loc[article])]
                    i=i+1
                else:
                    row = np.array(df_pca_reduit.loc[article])
                    arr_article = np.r_[arr_article,[row]]
            return (-1,arr_article.mean(axis=0))
        # fonction qui renvoie le profil d'un article pour un utilisateur
        def get_profile_article(user_id, selector_type):
            if(selector_type==article_selector_type.FIRST_ARTICLE):return get_first_article(user_id)
            elif(selector_type==article_selector_type.LAST_ARTICLE):return get_last_article(user_id)
            elif(selector_type==article_selector_type.AVG_ARTICLE):return get_avg_article(user_id)
            else:return get_last_article(user_id) # par defaut on renvoie le profil du dernier article

        #Fonction pour filtrer l'article de référence dans la liste des recommandations
        def get_filtered_articles(similar_indices, ignore_id):
            similar_indices_filtered = list(filter(lambda x: x != ignore_id, similar_indices))
            return similar_indices_filtered

        def get_articles_recommanded(user_id, selector_type, nb_article):
            num_article=get_profile_article(user_id, selector_type)[0]
            article_profile=get_profile_article(user_id, selector_type)[1]
            cosine_similarities = cosine_similarity(article_profile, df_pca_reduit)
            
            if(num_article.values[0]>=0):
                    similar_indices = cosine_similarities.argsort().flatten()[-(nb_article+1):]
                    similar_indices_filtered = get_filtered_articles(similar_indices, num_article.values[0])
            else:
                    similar_indices = cosine_similarities.argsort().flatten()[-nb_article:]
                    similar_indices_filtered = similar_indices
            #Sort the similar items by similarity
            similar_items = sorted([(article_metadata["article_id"][i], cosine_similarities[0,i]) for i in similar_indices_filtered], key=lambda x: -x[1])
            return similar_items

        result[0] = get_articles_recommanded(id_user, selector_type, nb_reco1)


    if(type_reco in (2,3)): # Reco collaborative filtering ou reco hybride

        blob_to_read = BytesIO(modelereco)
        with blob_to_read as f: 
            modele_reco = pickle.load(f)

        def get_article_list(id_user):
            liste = article_metadata['article_id'].tolist()
            for x in clicks_sample[clicks_sample['user_id']==id_user].click_article_id.values:
                liste.remove(x)
            return liste

        def get_predictions(id_user):
            liste = get_article_list(id_user)
            result=[]
            for article_id in liste:
                result.append(modele_reco.predict(id_user, article_id))
            return result

        def get_top_n(predictions, n=10):
            # First map the predictions to each user.
            top_n = defaultdict(list)
            for uid, iid, true_r, est, _ in predictions:
                top_n[uid].append((iid, est))

            # Then sort the predictions for each user and retrieve the k highest ones.
            for uid, user_ratings in top_n.items():
                user_ratings.sort(key=lambda x: x[1], reverse=True)
                top_n[uid] = user_ratings[:n]
            return top_n
        
        def get_collaborative_reco(id_user, nb_reco):
            test = get_predictions(id_user)
            return get_top_n(test, nb_reco)[id_user]
        
        if(type_reco ==2):
            result[0] = get_collaborative_reco(id_user, nb_reco2)
        else:
            result[1] = get_collaborative_reco(id_user, nb_reco2)

    class NpEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super(NpEncoder, self).default(obj)

    return func.HttpResponse(
            json.dumps(result, cls=NpEncoder),
             status_code=200
        )

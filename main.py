import json
from pymongo.mongo_client import MongoClient
from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans, KMeansModel
import matplotlib.pyplot as plt
from pyspark.sql.functions import col


def set_mongo():
    client = MongoClient("localhost:27017")
    return client["movielens"], client


def save_data_in_mongo(db):
    """Save data from .dat to mongo collection
    """

    collections = ["movies", "ratings", "users"]
    for item in collections:
        with open(f"./ml-1m/{item}.dat", "rb") as f:
            dat_data = f.read()
            dat_data = dat_data.decode("utf-8", errors="replace")
            json_data = json.dumps(dat_data).split('\\n')
            json_data.pop()
            data = []

            # Transform movies data from .dat into json object list
            if item == "movies":
                for movie_data in json_data:
                    movie = dict()
                    fields = movie_data.split("::")
                    movie["MovieID"] = fields[0]
                    movie["Title"] = fields[1]
                    movie["Genres"] = fields[2].split('|')
                    if not fields[0].isdigit():
                        movie["MovieID"] = fields[0][1]
                    movie["MovieID"] = int(movie["MovieID"])
                    data.append(movie)

            # Transform ratings data from .dat into json object list
            if item == "ratings":
                for rating_data in json_data:
                    rating = dict()
                    fields = rating_data.split("::")
                    rating["UserID"] = fields[0]
                    rating["MovieID"] = int(fields[1])
                    rating["Rating"] = int(fields[2])
                    rating["Timestamp"] = int(fields[3])
                    if not fields[0].isdigit():
                        rating["UserID"] = fields[0][1]
                    rating["UserID"] = int(rating["UserID"])
                    data.append(rating)

            # Transform users data from .dat into json object list
            if item == "users":
                for user_data in json_data:
                    user = dict()
                    fields = user_data.split("::")
                    user["UserID"] = fields[0]
                    user["Gender"] = fields[1]
                    user["Age"] = int(fields[2])
                    user["Occupation"] = int(fields[3])
                    user["ZipCode"] = fields[4]
                    if not fields[0].isdigit():
                        user["UserID"] = fields[0][1]
                    user["UserID"] = int(user["UserID"])
                    data.append(user)

            collection = db[item]
            collection.insert_many(data)
    print("Data saved !")


def recommander_films(utilisateur_id, nb_films):
    # Obtenir le cluster de l'utilisateur
    cluster_utilisateur = df_ratings_avec_clusters.filter(
        df_ratings_avec_clusters.UserID == utilisateur_id).select("cluster").collect()[0][0]

    # Filtrer les films du cluster de l'utilisateur
    films_cluster = df_ratings_avec_clusters.filter(
        df_ratings_avec_clusters.cluster == cluster_utilisateur).select("MovieID")

    # Déterminer les notes moyennes des films du cluster
    notes_moyennes_films = df_ratings.groupby("MovieID").mean(
        "Rating").join(films_cluster, on="MovieID")

    # Recommander les films avec les notes moyennes les plus élevées
    films_recommandes = notes_moyennes_films.orderBy(
        "Rating", ascending=False).limit(nb_films)

    return films_recommandes.collect()


if __name__ == "__main__":
    db, client = set_mongo()
    # save_data_in_mongo(db)
    spark = SparkSession.builder.appName("MovieLens").getOrCreate()
    sc = spark.sparkContext
    collections = ["movies", "ratings", "users"]
    df = dict()
    for name in collections:
        rdd = sc.parallelize(db[name].find({}, {"_id": 0}))

        # Créer un DataFrame à partir du RDD
        df[name] = spark.createDataFrame(rdd)

        # Afficher les cinq premières lignes du DataFrame
        df[name].show(5)

    # Supprimer les doublons
    df_ratings = df['ratings'].dropDuplicates()

    # Gérer les valeurs manquantes
    df_users = df['users'].fillna("Inconnu")
    df_movies = df['movies'].fillna("Inconnu")

    # Convertir les types de données
    df_ratings = df_ratings.withColumn(
        "Rating", df_ratings.Rating.cast("float"))

    # Déterminer le nombre d'utilisateurs, de films et de notes
    nb_users = df_users.count()
    nb_movies = df_movies.count()
    nb_ratings = df_ratings.count()

    # Afficher les statistiques des notes
    df_ratings.describe().show()

    # Déterminer les films les mieux notés
    df_movies_moyennes = df_ratings.groupby("MovieID").mean(
        "Rating").orderBy("avg(Rating)", ascending=False)

    df_movies_top10 = df_movies_moyennes.limit(10)

    # Afficher les films les mieux notés
    df_movies_top10.show()

    # Définir le nombre de clusters
    nb_clusters = 3

    # Entraîner le modèle KMeans

    feature_columns = ["MovieID", "Rating", "UserID"]

    # Initialize the VectorAssembler
    assembler = VectorAssembler(
        inputCols=feature_columns, outputCol="features")

    df_with_features = assembler.transform(df_ratings.select(*feature_columns))

    # Initialize and fit the KMeans model
    model = KMeans(k=nb_clusters).fit(df_with_features)

    # Prédire les clusters des utilisateurs

    predictions = model.transform(df_with_features)

    predicted_columns = predictions.columns
    cluster_col = col("features")

    # Ajouter la prédiction de cluster au dataframe
    df_ratings_avec_clusters = df_with_features.withColumn(
        "cluster", cluster_col)

    # Afficher les résultats
    df_ratings_avec_clusters.show()

    # Recommander des films pour un utilisateur
    utilisateur_id = 1
    nb_films = 5

    ####### OKAY ###########
    films_recommandes = recommander_films(utilisateur_id, nb_films)

    # Afficher les films recommandés
    for film in films_recommandes:
        print(f"Film ID: {film[0]} - Note moyenne: {film[1]}")

    # Visualiser la répartition des notes
    plt.hist(df_ratings.Rating, bins=50)
    plt.xlabel("Note")
    plt.ylabel("Nombre d'utilisateurs")

    spark.stop()

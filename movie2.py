#%%

#some preparation:
import streamlit as st
import pandas as pd
import numpy as np
movies = pd.read_csv(r"movies.csv")
ratings = pd.read_csv(r"ratings.csv")

av_rating = pd.DataFrame(ratings.groupby("movieId").aggregate(
    {"rating": ["mean", "median", "count", "min", "max"]}
    ))
av_rating = av_rating.droplevel(0, axis=1).reset_index() #removes "rating"

movies_rated = movies.merge(av_rating, on="movieId", how="left") #merges

user_movie_df = pd.pivot_table(data=ratings, values='rating', index='userId', columns='movieId')



#Extracting Years from Movies
movies["year"] = movies.title.str.extract("(\d{4})")
movies["year"] = movies.loc[movies.year.notnull(), "title"].str[:,-6]

missing_years = {
    "title":["Babylon", 
        "Ready Player One", 
        "Hyena Road", 
        "The Adventures of Sherlock Holmes and Doctor Watson", 
        "Nocturnal Animals", 
        "Paterson", 
        "Moonlight",
        "The OA",
        "Cosmos",
        "Maria Bamford: Old Baby",
        "Generation Iron 2",
        "Black Mirror"
        ], 
    "year":[1994, 2018, 2015, 1980, 2016, 2016, 2016, 2016, 2019, 2017, 2017, 2011]
}

missing_years_df = pd.DataFrame(missing_years)
#movies = movies.set_index("title")
#missing_years_df = missing_years_df.set_index("title")
#movies.fillna(missing_years_df)
#movies = movies.reset_index()
#%%

### Popular Movies #############################

st.header("Most popular movies")

#works for n <= 38
def get_x_popular_movies(n):
    max_count = movies_rated["count"].max()

    #keep only movies with at least half as many ratings as max_count
    popular_movies = movies_rated.loc[movies_rated["count"]>=max_count*0.5, ["title", "mean", "count"]]

    #sort the remaining movies by score
    popular_movies = popular_movies.sort_values("mean", ascending=False).reset_index().drop("index", axis=1)
    popular_movies.index = np.arange(1, len(popular_movies) + 1)
    popular_movies.columns = ["Title", "Rating", "Total Reviews"]
    popular_movies["Total Reviews"] = popular_movies["Total Reviews"].round(2)
    popular_movies["Rating"] = popular_movies.Rating.round(1)

    return popular_movies.head(n)



#slider to enter a number
input_int_pop_movies = st.slider("Pick a number", min_value=1, max_value=38, value=5, step=1, format=None, key="slider")

#Output of x_popular_movies
output_pop_movies = get_x_popular_movies(input_int_pop_movies)
st.dataframe(output_pop_movies) #.style.format("{:.2%}"))


def recommend_based_on(item_id, n=10):

    preference_ratings = user_movie_df.loc[:,item_id]

    ## get .corr()
    # we get warnings because computing the pearson correlation coefficient with NaNs, but the results are still ok
    similar_to_preference = user_movie_df.corrwith(preference_ratings)
    corr_preference = pd.DataFrame(similar_to_preference, columns=['PearsonR'])
    corr_preference.dropna(inplace=True)

    ## add some condition / infos
    # keep correlations where number of users rated > 10
    rating = pd.DataFrame(ratings.groupby('movieId')['rating'].mean())
    rating['rating_count'] = ratings.groupby('movieId')['rating'].count()

    preference_corr_summary = corr_preference.join(rating['rating_count'])
    preference_corr_summary.drop(item_id, inplace=True) # drop the preference itself

    top_n = preference_corr_summary[preference_corr_summary['rating_count']>=10].sort_values('PearsonR', ascending=False).head(n)
    top_n = top_n.merge(movies, left_index=True, right_on="movieId")
    top_n = top_n.drop(["PearsonR", "rating_count", "movieId"], axis = 1)
    top_n.index = np.arange(1, len(top_n) + 1)

    return top_n

st.header("If you like Toy Story, you might also like")
st.dataframe(recommend_based_on(item_id = 1, n=5))


######### Movie racommendations based on User ##### (morning)


## prepare the matrix (pivottable)

## we did something with a numpy array

## calculate cosine similarities







def recommended_for_user(userId, n):
    # compute the weights for one user
    weights = (
        user_similarities.query("UserId!=@user_id")[user_id] / sum(user_similarities.query("userID!=@user_id")[user_id])
            )

    # find items that have not been rated by user
    users_items.loc[user_id,:]==0

    # select items that the inputed user has not seen
    not_seen_items = users_items.loc[users_items.index!=user_id, users_items.loc[user_id,:]==0]

    # dot product between the not-seen-items and the weights
    weighted_averages = pd.DataFrame(not_seen_items.T.dot(weights), columns=["predicted_rating"])

    # find the top n recommendations
    recommendations = weighted_averages.merge(places, left_index=True, right_on="placeID")
    top_n = recommendations.sort_values("predicted_rating", ascending=False).head(n)
    return top_n



######## EVALUATION of model offline (afternoon)

#... we started with the pivottable and fill in all 0 (for NaN)

#... transfer pivottable to np.array() and extract coordinates for all non-zero values 
#... make a dataframe for the coordinates (and transpose (.T))
###ratings_pos = pd.DataFrame(
###    np.nonzero(np.array(users_items)), # find out all the positions different than 0
###).T
###ratings_pos.columns = ["row_pos", "column_pos"]

#... That gives us a dataframe that contains the coordinates of all non-zero values.

#-------- step 1 finished ------

#... next: train test split on coordinates
###from sklearn.model_selection import train_test_split
###train_pos, test_pos = train_test_split(ratings_pos, 
###                                       random_state=123, 
###                                       test_size=.1)



# ... fill the coordinates for TRAIN and TEST into new matrix-dataframes
# creating a matrix only with zeros for TRAIN
### train = np.zeros(users_items.shape)

# then fill in the values for TRAIN 
###for pos in train_pos.values: 
###    index = pos[0]
###    col = pos[1]
###    train[index, col] = users_items.iloc[index, col]

# convert to dataframe
###train = pd.DataFrame(train, 
###                     columns=users_items.columns, 
###                     index=users_items.index)


# ... Repeat for TEST
###test = np.zeros(users_items.shape)

###for pos in test_pos.values: 
###    index = pos[0]
###    col = pos[1]
###    test[index, col] = users_items.iloc[index, col]
    
###test = pd.DataFrame(test, 
###                    columns=users_items.columns, 
###                    index=users_items.index)

#------- step 2 finished -------


# build similarity matrix on TRAIN
###train_similarity = pd.DataFrame(cosine_similarity(train), 
###                                columns=train.index, 
###                                index=train.index)



#------ step 3 finished ---------

# Build a function that calculates predictions

###def recommender(index_pos, column_pos): 
###    # build a df with the ratings for one place (column_name) and
###    # the similarities to one user (index_name)
###    results = (
###      pd.DataFrame({
###          'ratings': train.iloc[:,column_pos], 
###          'similarities' : train_similarity.iloc[index_pos,:].tolist()
###      })
###    )
    
    # compute the weights
###    results = results.assign(weights = results.similarities / (sum(results.similarities) -1))
    
    # compute the weighted ratings
###    results = results.assign(weighted_ratings = results.ratings * results.weights)
    
    # return rating prediction for one user and one movie
###    return results.weighted_ratings.sum()

#------ step 4 finished ------------


#get outputs and store in list 
###recs_test = []

###for row in test_pos.iterrows():
###    recs_test.append(
###        recommender(
###            index_pos = int(row[1][0]), 
###            column_pos = int(row[1][1])
###        )
###    )

###test_pos = test_pos.assign(pred_rating = recs_test)

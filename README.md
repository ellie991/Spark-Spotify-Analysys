This project aims to analyze the effectiveness of various classification models on a dataset of 112.6MB containing 170,634 observations and 143 features. The dataset, downloaded from Kaggle, includes all the songs from the Daily Top 200 Spotify charts in 35+1 countries (global) over a period of more than three years (2017-2020). For each song, it contains aggregated data such as title, artist, country, genre, popularity, Top 50, Top 10, and more.
I evaluated the accuracy and performance of three models: Logistic Regression, Support Vector Machine (SVM), and Neural Network, using PySpark, a framework for distributed computing and large-scale data processing. The goal was to determine which model performs best in classifying whether a song is in the Top 50 charts.

RESULTS

Among the models tested, the Neural Network proved to be the most effective, achieving the highest accuracy and F1 score. It was able to correctly identify Top 50 songs with a notable precision and recall, though it required the longest training time.
Logistic Regression was a close contender, offering a good balance between performance and efficiency. It performed almost as well as the Neural Network but was significantly faster to train.
The Support Vector Machine (SVM) had similar accuracy to Logistic Regression but was less efficient in terms of precision. Additionally, it took the longest to train among the three models.
Overall, while the Neural Network had the best results, Logistic Regression provided a faster and nearly comparable alternative, and SVM lagged behind in both performance and efficiency.

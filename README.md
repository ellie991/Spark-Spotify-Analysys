# Spark-Spotify-Analysis
Big Data Analysis w/ Spark

Il presente progetto si propone di analizzare l'efficacia di diversi modelli di classificazione su un dataset di 112,6MB contenente 170634 osservazioni e 143 features. Il seguente dataset è stato scaricato dal sito Kaggle  e contiene tutte le canzoni delle classifiche Daily Top 200 di Spotify in 35+1 paesi (globali) per un periodo di oltre tre anni (2017-2020), e per ogni canzone contiene dati aggregati quali titolo, artista, paese, genere, popolarità, top50, top10 etc.
Il seguente progetto si concentra sul valutare l’accuratezza e la prestazione di tre modelli: regressione logistica, support vector machine (SVM) e rete neurale utilizzando PySpark, un framework per il calcolo distribuito e il trattamento di grandi set di dati.
L'obiettivo è valutare quale modello fornisce le migliori performance nella classificazione dei brani musicali, nello specifico la presenza di un brano nella classifica Top50 nel caso della regressione logistica e della rete neurale, mentre nel caso della SVM se la canzone contiene linguaggio esplicito.

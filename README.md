# Identification of Grey Sheep Users in UBCF: An Empirical Comparison

## Summary
This Master's Thesis consists of the identification of Grey Sheep Users. It is a comparison between the different solutions that exist to identify these users who are considered outliers. It is a project within the framework of recommender systems, using data mining tools, machine learning, statistics, and outlier detection algorithms. 

Through the different existing proposals to identify these users, the results given by each solution for the same data set will be compared in order to draw conclusions. 

Mainly, recommender system algorithms have been implemented. The widespread collaborative filtering method has been used to predict users' tastes, namely User-Based Collaborative Filtering.

In addition, both unsupervised machine learning algorithms have been used, creating clusters based on improved versions of the KMeans algorithm, and supervised ones, using algorithms such as KNN. 

On the other hand, mathematical formulas have been used, such as t-tests, probability formulas and formulas to calculate the similarity between users. 

In order to develop this project correctly, the Python programming language has been used. A large number of libraries such as Numpy and Pandas have been used to handle the data frames. Also, the use of libraries such as jobLib, which allows parallelizing processes and creating threads, has been required. In addition, Spark machine clusters have been used for certain parts of the project, which require a large number of resources. 

## Background

These days we are receiving many different kinds of recommendations from different companies, such as Netflix, Amazon. They have a huge impact in out daily life. The algorithms behind those recommendations aim to increase the quality of life and help people to make better choices. 
Recommender systems apply techniques to develop predictions about what a customer would like. These systems work thanks to the data collected from users and their goal is to make accurate recommendations to them. 
Thanks to ML techniques, RS have advanced a lot these past years. They are very high efficient. Using a recommender system technique called Collaborative Filtering, which is very extended, there can appear different outliers when talking about users. These outliers are users that have special tastes and don’t match with the rest of the users, they are called Grey Sheep Users. On the other hand, the users that can have accurate recommendations are called White Sheep users. These last type of users have high correlations between the users that are alike. However, grey sheep users don’t find high correlation with other users. This make it difficult for us to make good recommendations for them. 
There are few research papers and investigations about how to treat these kind of users. The goal of my project is to implement 3 solutions from 3 research publications and make different comparisons.



## Goal of the project

The goal of this project is the implementation of different approaches and solutions to deal with grey sheep users, using User-Based Collaborative Filtering. Research communities have always focused on improving the efficiency of collaborative filtering algorithms. On the other hand, they have not paid attention to the problem of gray sheep users, being a major challenge of CF. The project focuses on comparing the results of the different solutions that have been given in the research papers, which there are just a few of them. The project will be using the famous data set MovieLens, applying Recommender Systems technologies in combination of Outlier Detection and Clustering Techniques.









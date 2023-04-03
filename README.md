The solution of task on MTS Machine learning cup 2023.
https://ods.ai/competitions/mtsmlcup

This solution took 34th place over from ~2000 participants and ~500 rows of leaderboard.

The score on the private part of dataset = 1,7172184804.

# Task description

It was needed to predict the gender and the age of the internet users basing on the history of urls of web-pages they visited. Besides of urls the dataset contained such data like 
- date, part of a day of the visit,
- model, type,  price, os and the manufacturer of the device with which users visited the web-pages
- the city and the region where users connected to the net

The score consists of two two marks:
- gini score of gender binary classification
- f1_weighted of age prediction, age devided on 6 bins
Score = 2 * f1_weighted of age prediction on bins + gini on gender

# Solution description

I have generated features based either on history of urls, either on meta information about the visits:
- urls and users where imagined like data to recommendations system and fitted to BiVAECF model (https://cornac.readthedocs.io/en/latest/_modules/cornac/models/bivaecf/recom_bivaecf.html). The users z-score of of trained model where took as users features
- urls also have been vectorized with help of TfIDF and have took as features (only 2000 most popular urls)
- meta information was modified to statics such as max, min and mean values
- also i've parsed some information about mean income, age, genders of population in cities and regions

All features are fitted to classification model separatly (for age and gender). The model showen best score was DANet classification deep learing model (https://arxiv.org/abs/2112.02962)

The sklearn StackingClassifier showen good score, but less than DANet = 1.68

# Run

To run the model place files with with input data in Context_data folder and run

python3 main.py

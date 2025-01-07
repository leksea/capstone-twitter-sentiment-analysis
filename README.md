# Readme: Twitter Sentiment Analysis. 

# Table of Contents
- [1. Repository Structure](#1-repository-structure)
- [2. Business Problem Overview](#2-business-problem-overview)
- [3. Proposed Solution](#3-proposed-solution)
- [4. Datasets Download Links and Notebook Requirements](#4-datasets-download-links-and-notebook-requirements)
- [5. Datasets EDA and Cleaning](#5-datasets-eda-and-cleaning)
- [6. Data Transformation For Classification](#6-data-transformation-for-classification)
- [7. Modelling](#7-modelling)
- [8. Results](#8-results)
- [9. Conclusion](#9-conclusion)
- [10. Slides](#10-slides)


### Structure.

```plaintext
.
├── Data
│   ├── Covid-19 Twitter Dataset (Apr-Jun 2020).csv^^
│   ├── Covid-19 Twitter Dataset (Apr-Jun 2021).csv^^
│   ├── Covid-19 Twitter Dataset (Aug-Sep 2020).csv^^
│   ├── coordinate_cache.json
│   ├── glove.twitter.27B.200d.txt^^^
│   ├── heatmap_city.html^
│   ├── lemmatized_df.csv^
│   ├── location_cache.json
│   └── tweets_df_no_retweets.csv^
├── Images
│   ├── CM_4.6.png
│   ├── Features_4.7.png
│   ├── N-Grams_3.1.png
│   ├── Sentiment_Dist_4.1.png
│   └── TFIDF_3.1.png
├── LICENSE
├── Model
│   ├── X_test.pkl^
│   ├── X_train.pkl^
│   ├── X_val.pkl^
│   ├── trained_model.pkl
│   ├── y_test.pkl^
│   ├── y_train.pkl^
│   └── y_val.pkl^
├── Notebook.ipynb
├── README.md
├── Slides.pdf
└── requirements.txt

^ -- Optional data files saved/uploaded during the notebook execution after lengthy computations (recommended)!
^^ -- Data requires user download.
^^^ -- Data will be downloaded if not present. 
```
# Twitter Sentiment Analysis of Public Reaction to COVID-19 News

**Project Overview:**

This project aims to analyze a large dataset of COVID-19-related tweets to understand how public sentiment evolves and spreads in response to news announcements and events. By leveraging natural language processing (NLP) techniques and sentiment analysis models, we seek to gain valuable insights into online conversations surrounding the pandemic dynamics.

**Importance and Motivation:**

Understanding public sentiment during a global crisis like the COVID-19 pandemic is crucial for various stakeholders, including:

- **Public Health Officials:** To gauge public response to health policies and interventions.
- **Media Outlets:** To assess the impact of their news coverage on public perception.
- **Government Agencies:** Monitor public opinion and tailor communication strategies.
- **Researchers:** To study the spread of information and misinformation on social media.

This project contributes to this understanding by providing a comprehensive analysis of Twitter data, revealing trends and patterns in public sentiment related to COVID-19.


## Business Problem and Objectives

**Problem Statement:**

Media outlets and public health organizations need to better understand how their COVID-19-related news and announcements influence public sentiment on Twitter. This project addresses this need by analyzing a large dataset of tweets to identify and track sentiment trends in response to news events.

**Key Questions:**

- How do positive and negative sentiments spread among users following a COVID-19 news announcement?
- What are the key topics and themes associated with different sentiment trends?
- Can we identify any patterns or correlations between news events and changes in public sentiment?

**Project Objectives:**

- To develop a robust NLP pipeline for cleaning, preprocessing, and analyzing Twitter data.
- To apply sentiment analysis models to classify tweets and track sentiment trends over time.
- To visualize and interpret the sentiment analysis results to provide actionable insights.
- To potentially identify key influencers and networks driving sentiment on Twitter.


## Data Acquisition and Preparation

**Data Sources:**

1. **Covid-19 Twitter Dataset:** The primary dataset for this Twitter sentiment analysis project is the "Covid-19 Twitter Dataset" available on Kaggle. This dataset contains a large collection of tweets related to COVID-19, including tweet text, user details, location, and sentiment labels.
The dataset can be accessed and downloaded from the following Kaggle page:  [Covid-19 Twitter Dataset](https://www.kaggle.com/datasets/arunavakrchakraborty/covid19-twitter-dataset/data)
You have to download the dataset and extract it into the `Data` directory or upload it into the cloud and directly specify the `data_dir` at the beginning of the 2.5 section of the notebook. 

2. **Pre-trained GloVe embeddings** from Stanford NLP. These word embeddings capture semantic relationships between words and can improve the performance of NLP models, which can be obtained from the [Stanford NLP website](https://nlp.stanford.edu/projects/glove/). The notebook will look for the specific file `glove.twitter.27B.200d.txt` inside the `data_dir`. If the precompiled embedding file is not found, the entire embedding package will be downloaded and extracted. The embedding files are several gigabytes each and upload might take some time.
3. 
   

## Methodology

1. **Data Cleaning and Preprocessing:** 
   - Cleaned the raw tweet data by removing noise, special characters, links, mentions, and hashtags.
   - Preprocessed the text data by tokenizing, lemmatizing, and removing stop words.
   - Extracted relevant features for sentiment analysis.

2. **Sentiment Analysis:**
   - Applied sentiment analysis models to classify tweets as positive, negative, or neutral.
   - Trained and evaluated the performance of different sentiment analysis models.

3. **Visualization and Interpretation:**
   - Visualized the sentiment analysis results to understand trends and patterns.
   - Interpreted the findings to provide actionable insights.


## Conclusion

xx~~Summarize the key findings and insights obtained from the sentiment analysis. Discuss the implications of the results for understanding public reaction to COVID-19 news on Twitter.


## Future Work

xx~~Outline potential directions for future research and development, such as:

- Exploring more advanced NLP techniques for sentiment analysis.
- Investigating the impact of specific news events on sentiment.
- Identifying key influencers and networks driving sentiment on Twitter.





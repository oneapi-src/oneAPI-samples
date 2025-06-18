# %% [markdown]
# # Job recommendation system
# 
# The code sample contains the following parts:
# 
# 1.   Data exploration and visualization
# 2.   Data cleaning/pre-processing
# 3.   Fake job postings identification and removal
# 4.   Job recommendation by showing the most similar job postings
# 
# The scenario is that someone wants to find the best posting for themselves. They have collected the data, but he is not sure if all the data is real. Therefore, based on a trained model, as in this sample, they identify with a high degree of accuracy which postings are real, and it is among them that they choose the best ad for themselves.
# 
# For simplicity, only one dataset will be used within this code, but the process using one dataset is not significantly different from the one described earlier.
# 

# %% [markdown]
# ## Data exploration and visualization
# 
# For the purpose of this code sample we will use Real or Fake: Fake Job Postings dataset available over HuggingFace API. In this first part we will focus on data exploration and visualization. In standard end-to-end workload it is the first step. Engineer needs to first know the data to be able to work on it and prepare solution that will utilize dataset the best.
# 
# Lest start with loading the dataset. We are using datasets library to do that.

# %%
from datasets import load_dataset

dataset = load_dataset("victor/real-or-fake-fake-jobposting-prediction")
dataset = dataset['train']

# %% [markdown]
# To better analyze and understand the data we are transferring it to pandas DataFrame, so we are able to take benefit from all pandas data transformations. Pandas library provides multiple useful functions for data manipulation so it is usual choice at this stage of machine learning or deep learning project.
# 

# %%
import pandas as pd
df = dataset.to_pandas()

# %% [markdown]
# Let's see 5 first and 5 last rows in the dataset we are working on.

# %%
df.head()

# %%
df.tail()

# %% [markdown]
# Now, lets print a concise summary of the dataset. This way we will see all the column names, know the number of rows and types in every of the column. It is a great overview on the features of the dataset.

# %%
df.info()

# %% [markdown]
# At this point it is a good idea to make sure our dataset doen't contain any data duplication that could impact the results of our future system. To do that we firs need to remove `job_id` column. It contains unique number for each job posting so even if the rest of the data is the same between 2 postings it makes it different.

# %%
# Drop the 'job_id' column
df = df.drop(columns=['job_id'])
df.head()

# %% [markdown]
# And now, the actual duplicates removal. We first pring the number of duplicates that are in our dataset, than using `drop_duplicated` method we are removing them and after this operation printing the number of the duplicates. If everything works as expected after duplicates removal we should print `0` as current number of duplicates in the dataset.

# %%
# let's make sure that there are no duplicated jobs

print(df.duplicated().sum())
df = df.drop_duplicates()
print(df.duplicated().sum())

# %% [markdown]
# Now we can visualize the data from the dataset. First let's visualize data as it is all real, and later, for the purposes of the fake data detection, we will also visualize it spreading fake and real data.
# 
# When working with text data it can be challenging to visualize it. Thankfully, there is a `wordcloud` library that shows common words in the analyzed texts. The bigger word is, more often the word is in the text. Wordclouds allow us to quickly identify the most important topic and themes in a large text dataset and also explore patterns and trends in textural data.
# 
# In our example, we will create wordcloud for job titles, to have high-level overview of job postings we are working with.

# %%
from wordcloud import WordCloud # module to print word cloud
from matplotlib import pyplot as plt
import seaborn as sns

# On the basis of Job Titles form word cloud
job_titles_text = ' '.join(df['title'])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(job_titles_text)

# Plotting Word Cloud
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.title('Job Titles')
plt.axis('off')
plt.tight_layout()
plt.show()

# %% [markdown]
# Different possibility to get some information from this type of dataset is by showing top-n most common values in given column or distribution of the values int his column.
# Let's show top 10 most common job titles and compare this result with previously showed wordcould.

# %%
# Get Count of job title
job_title_counts = df['title'].value_counts()

# Plotting a bar chart for the top 10 most common job titles
top_job_titles = job_title_counts.head(10)
plt.figure(figsize=(10, 6))
top_job_titles.sort_values().plot(kind='barh')
plt.title('Top 10 Most Common Job Titles')
plt.xlabel('Frequency')
plt.ylabel('Job Titles')
plt.show()

# %% [markdown]
# Now we can do the same for different columns, as `employment_type`, `required_experience`, `telecommuting`, `has_company_logo` and `has_questions`. These should give us reale good overview of different parts of our dataset.

# %%
# Count the occurrences of each work type
work_type_counts = df['employment_type'].value_counts()

# Plotting the distribution of work types
plt.figure(figsize=(8, 6))
work_type_counts.sort_values().plot(kind='barh')
plt.title('Distribution of Work Types Offered by Jobs')
plt.xlabel('Frequency')
plt.ylabel('Work Types')
plt.show()

# %%
# Count the occurrences of required experience types
work_type_counts = df['required_experience'].value_counts()

# Plotting the distribution of work types
plt.figure(figsize=(8, 6))
work_type_counts.sort_values().plot(kind='barh')
plt.title('Distribution of Required Experience by Jobs')
plt.xlabel('Frequency')
plt.ylabel('Required Experience')
plt.show()

# %% [markdown]
# For employment_type and required_experience we also created matrix to see if there is any corelation between those two. To visualize it we created heatmap. If you think that some of the parameters can be related, creating similar heatmap can be a good idea.

# %%
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd

plt.subplots(figsize=(8, 8))
df_2dhist = pd.DataFrame({
    x_label: grp['required_experience'].value_counts()
    for x_label, grp in df.groupby('employment_type')
})
sns.heatmap(df_2dhist, cmap='viridis')
plt.xlabel('employment_type')
_ = plt.ylabel('required_experience')

# %%
# Count the occurrences of unique values in the 'telecommuting' column
telecommuting_counts = df['telecommuting'].value_counts()

plt.figure(figsize=(8, 6))
telecommuting_counts.sort_values().plot(kind='barh')
plt.title('Counts of telecommuting vs Non-telecommuting')
plt.xlabel('count')
plt.ylabel('telecommuting')
plt.show()

# %%
has_company_logo_counts = df['has_company_logo'].value_counts()

plt.figure(figsize=(8, 6))
has_company_logo_counts.sort_values().plot(kind='barh')
plt.ylabel('has_company_logo')
plt.xlabel('Count')
plt.title('Counts of With_Logo vs Without_Logo')
plt.show()

# %%
has_questions_counts = df['has_questions'].value_counts()

# Plot the counts
plt.figure(figsize=(8, 6))
has_questions_counts.sort_values().plot(kind='barh')
plt.ylabel('has_questions')
plt.xlabel('Count')
plt.title('Counts Questions vs NO_Questions')
plt.show()

# %% [markdown]
# From the job recommendations point of view the salary and location can be really important parameters to take into consideration. In given dataset we have salary ranges available so there is no need for additional data processing rather than removal of empty ranges but if the dataset you're working on has specific values, consider organizing it into appropriate ranges and only then displaying the result.

# %%
# Splitting benefits by comma and creating a list of benefits
benefits_list = df['salary_range'].str.split(',').explode()
benefits_list = benefits_list[benefits_list != 'None']
benefits_list = benefits_list[benefits_list != '0-0']


# Counting the occurrences of each skill
benefits_count = benefits_list.str.strip().value_counts()

# Plotting the top 10 most common benefits
top_benefits = benefits_count.head(10)
plt.figure(figsize=(10, 6))
top_benefits.sort_values().plot(kind='barh')
plt.title('Top 10 Salaries Range Offered by Companies')
plt.xlabel('Frequency')
plt.ylabel('Salary Range')
plt.show()

# %% [markdown]
# For the location we have both county, state and city specified, so we need to split it into individual columns, and then show top 10 counties and cities.

# %%
# Split the 'location' column into separate columns for country, state, and city
location_split = df['location'].str.split(', ', expand=True)
df['Country'] = location_split[0]
df['State'] = location_split[1]
df['City'] = location_split[2]

# %%
# Count the occurrences of unique values in the 'Country' column
Country_counts = df['Country'].value_counts()

# Select the top 10 most frequent occurrences
top_10_Country = Country_counts.head(10)

# Plot the top 10 most frequent occurrences as horizontal bar plot with rotated labels
plt.figure(figsize=(14, 10))
sns.barplot(y=top_10_Country.index, x=top_10_Country.values)
plt.ylabel('Country')
plt.xlabel('Count')
plt.title('Top 10 Most Frequent Country')
plt.show()

# %%
# Count the occurrences of unique values in the 'City' column
City_counts = df['City'].value_counts()

# Select the top 10 most frequent occurrences
top_10_City = City_counts.head(10)

# Plot the top 10 most frequent occurrences as horizontal bar plot with rotated labels
plt.figure(figsize=(14, 10))
sns.barplot(y=top_10_City.index, x=top_10_City.values)
plt.ylabel('City')
plt.xlabel('Count')
plt.title('Top 10 Most Frequent City')
plt.show()

# %% [markdown]
# ### Fake job postings data visualization 
# 
# What about fraudulent class? Let see how many of the jobs in the dataset are fake. Whether there are equally true and false offers, or whether there is a significant disproportion between the two. 

# %%
## fake job visualization
# Count the occurrences of unique values in the 'fraudulent' column
fraudulent_counts = df['fraudulent'].value_counts()

# Plot the counts using a rainbow color palette
plt.figure(figsize=(8, 6))
sns.barplot(x=fraudulent_counts.index, y=fraudulent_counts.values)
plt.xlabel('Fraudulent')
plt.ylabel('Count')
plt.title('Counts of Fraudulent vs Non-Fraudulent')
plt.show()

# %%
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='employment_type', hue='fraudulent')
plt.title('Count of Fraudulent Cases by Employment Type')
plt.xlabel('Employment Type')
plt.ylabel('Count')
plt.legend(title='Fraudulent')
plt.show()

# %%
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='required_experience', hue='fraudulent')
plt.title('Count of Fraudulent Cases by Required Experience')
plt.xlabel('Required Experience')
plt.ylabel('Count')
plt.legend(title='Fraudulent')
plt.show()

# %%
plt.figure(figsize=(30, 18))
sns.countplot(data=df, x='required_education', hue='fraudulent')
plt.title('Count of Fraudulent Cases by Required Education')
plt.xlabel('Required Education')
plt.ylabel('Count')
plt.legend(title='Fraudulent')
plt.show()

# %% [markdown]
# We can see that there is no connection between those parameters and fake job postings. This way in the future processing we can remove them.

# %% [markdown]
# ## Data cleaning/pre-processing
# 
# One of the really important step related to any type of data processing is data cleaning. For texts it usually includes removal of stop words, special characters, numbers or any additional noise like hyperlinks. 
# 
# In our case, to prepare data for Fake Job Postings recognition we will first, combine all relevant columns into single new record and then clean the data to work on it.

# %%
# List of columns to concatenate
columns_to_concat = ['title', 'location', 'department', 'salary_range', 'company_profile',
                     'description', 'requirements', 'benefits', 'employment_type',
                     'required_experience', 'required_education', 'industry', 'function']

# Concatenate the values of specified columns into a new column 'job_posting'
df['job_posting'] = df[columns_to_concat].apply(lambda x: ' '.join(x.dropna().astype(str)), axis=1)

# Create a new DataFrame with columns 'job_posting' and 'fraudulent'
new_df = df[['job_posting', 'fraudulent']].copy()

# %%
new_df.head()

# %%
# import spacy
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

def preprocess_text(text):
    # Remove newlines, carriage returns, and tabs
    text = re.sub('\n','', text)
    text = re.sub('\r','', text)
    text = re.sub('\t','', text)
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)

    # Remove special characters
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)

    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)

    # Remove digits
    text = re.sub(r'\d', '', text)

    # Convert to lowercase
    text = text.lower()

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words = [word for word in text.split() if word.lower() not in stop_words]
    text = ' '.join(words)

    return text



# %%
new_df['job_posting'] = new_df['job_posting'].apply(preprocess_text)

new_df.head()

# %% [markdown]
# The next step in the pre-processing is lemmatization. It is a process to reduce a word to its root form, called a lemma. For example the verb 'planning' would be changed to 'plan' world.

# %%
# Lemmatization
import en_core_web_sm

nlp = en_core_web_sm.load()

def lemmatize_text(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc])

# %%
new_df['job_posting'] = new_df['job_posting'].apply(lemmatize_text)

new_df.head()

# %% [markdown]
# At this stage we can also visualize the data with wordcloud by having special text column. We can show it for both fake and real dataset.

# %%
from wordcloud import WordCloud

non_fraudulent_text = ' '.join(text for text in new_df[new_df['fraudulent'] == 0]['job_posting'])
fraudulent_text = ' '.join(text for text in new_df[new_df['fraudulent'] == 1]['job_posting'])

wordcloud_non_fraudulent = WordCloud(width=800, height=400, background_color='white').generate(non_fraudulent_text)

wordcloud_fraudulent = WordCloud(width=800, height=400, background_color='white').generate(fraudulent_text)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

ax1.imshow(wordcloud_non_fraudulent, interpolation='bilinear')
ax1.axis('off')
ax1.set_title('Non-Fraudulent Job Postings')

ax2.imshow(wordcloud_fraudulent, interpolation='bilinear')
ax2.axis('off')
ax2.set_title('Fraudulent Job Postings')

plt.show()

# %% [markdown]
# ## Fake job postings identification and removal
# 
# Nowadays, it is unfortunate that not all the job offers that are posted on papular portals are genuine. Some of them are created only to collect personal data. Therefore, just detecting fake job postings can be very essential. 
# 
# We will create bidirectional LSTM model with one hot encoding. Let's start with all necessary imports.

# %%
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dropout

# %% [markdown]
# Make sure, you're using Tensorflow version 2.15.0

# %%
import tensorflow as tf
tf.__version__

# %% [markdown]
# Now, let us import Intel Extension for TensorFlow*. We are using Python API `itex.experimental_ops_override()`. It automatically replace some TensorFlow operators by Custom Operators under `itex.ops` namespace, as well as to be compatible with existing trained parameters.

# %%
import intel_extension_for_tensorflow as itex

itex.experimental_ops_override()

# %% [markdown]
# We need to prepare data for the model we will create. First let's assign job_postings to X and fraudulent values to y (expected value).

# %%
X = new_df['job_posting']
y = new_df['fraudulent']

# %% [markdown]
# One hot encoding is a technique to represent categorical variables as numerical values. 

# %%
voc_size = 5000
onehot_repr = [one_hot(words, voc_size) for words in X]

# %%
sent_length = 40
embedded_docs = pad_sequences(onehot_repr, padding='pre', maxlen=sent_length)
print(embedded_docs)

# %% [markdown]
# ### Creating model
# 
# We are creating Deep Neural Network using Bidirectional LSTM. The architecture is as followed:
# 
# * Embedding layer
# * Bidirectiona LSTM Layer
# * Dropout layer
# * Dense layer with sigmod function
# 
# We are using Adam optimizer with binary crossentropy. We are optimism accuracy.
# 
# If Intel® Extension for TensorFlow* backend is XPU, `tf.keras.layers.LSTM` will be replaced by `itex.ops.ItexLSTM`. 

# %%
embedding_vector_features = 50
model_itex = Sequential()
model_itex.add(Embedding(voc_size, embedding_vector_features, input_length=sent_length))
model_itex.add(Bidirectional(itex.ops.ItexLSTM(100)))
model_itex.add(Dropout(0.3))
model_itex.add(Dense(1, activation='sigmoid'))
model_itex.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model_itex.summary())

# %%
import numpy as np
X_final = np.array(embedded_docs)
y_final = np.array(y)

# %% [markdown]
# 

# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_final, y_final, test_size=0.25, random_state=320)

# %% [markdown]
# Now, let's train the model. We are using standard `model.fit()` method providing training and testing dataset. You can easily modify number of epochs in this training process but  keep in mind that the model can become overtrained, so that it will have very good results on training data, but poor results on test data.

# %%
model_itex.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=1, batch_size=64)

# %% [markdown]
# The values returned by the model are in the range [0,1] Need to map them to integer values of 0 or 1.

# %%
y_pred = (model_itex.predict(X_test) > 0.5).astype("int32")

# %% [markdown]
# To demonstrate the effectiveness of our models we presented the confusion matrix and classification report available within the `scikit-learn` library.

# %%
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix, classification_report

conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion matrix:")
print(conf_matrix)

ConfusionMatrixDisplay.from_predictions(y_test, y_pred)

class_report = classification_report(y_test, y_pred)
print("Classification report:")
print(class_report)

# %% [markdown]
# ## Job recommendation by showing the most similar ones

# %% [markdown]
# Now, as we are sure that the data we are processing is real, we can get back to the original columns and create our recommendation system.
# 
# Also use much more simple solution for recommendations. Even, as before we used Deep Learning to check if posting is fake, we can use classical machine learning algorithms to show similar job postings.
# 
# First, let's filter fake job postings.

# %%
real = df[df['fraudulent'] == 0]
real.head()

# %% [markdown]
# After that, we create a common column containing those text parameters that we want to be compared between theses and are relevant to us when making recommendations.

# %%
cols = ['title', 'description', 'requirements', 'required_experience',  'required_education', 'industry']
real = real[cols]
real.head()

# %%
real = real.fillna(value='')
real['text'] = real['description'] + real['requirements'] + real['required_experience'] + real['required_education'] + real['industry']
real.head()

# %% [markdown]
# Let's see the mechanism that we will use to prepare recommendations - we will use sentence similarity based on prepared `text` column in our dataset. 

# %%
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

# %% [markdown]
# Let's prepare a few example sentences that cover 4 topics. On these sentences it will be easier to show how the similarities between the texts work than on the whole large dataset we have.

# %%
messages = [
    # Smartphones
    "I like my phone",
    "My phone is not good.",
    "Your cellphone looks great.",

    # Weather
    "Will it snow tomorrow?",
    "Recently a lot of hurricanes have hit the US",
    "Global warming is real",

    # Food and health
    "An apple a day, keeps the doctors away",
    "Eating strawberries is healthy",
    "Is paleo better than keto?",

    # Asking about age
    "How old are you?",
    "what is your age?",
]

# %% [markdown]
# Now, we are preparing functions to show similarities between given sentences in the for of heat map. 

# %%
import seaborn as sns

def plot_similarity(labels, features, rotation):
  corr = np.inner(features, features)
  sns.set(font_scale=1.2)
  g = sns.heatmap(
      corr,
      xticklabels=labels,
      yticklabels=labels,
      vmin=0,
      vmax=1,
      cmap="YlOrRd")
  g.set_xticklabels(labels, rotation=rotation)
  g.set_title("Semantic Textual Similarity")

def run_and_plot(messages_):
  message_embeddings_ = model.encode(messages_)
  plot_similarity(messages_, message_embeddings_, 90)

# %%
run_and_plot(messages)

# %% [markdown]
# Now, let's move back to our job postings dataset. First, we are using sentence encoding model to be able to calculate similarities.

# %%
encodings = []
for text in real['text']:
    encodings.append(model.encode(text))

real['encodings'] = encodings

# %% [markdown]
# Then, we can chose job posting we wan to calculate similarities to. In our case it is first job posting in the dataset, but you can easily change it to any other job posting, by changing value in the `index` variable.

# %%
index = 0
corr = np.inner(encodings[index], encodings)
real['corr_to_first'] = corr

# %% [markdown]
# And based on the calculated similarities, we can show top most similar job postings, by sorting them according to calculated correlation value.

# %%
real.sort_values(by=['corr_to_first'], ascending=False).head()

# %% [markdown]
# In this code sample we created job recommendation system. First, we explored and analyzed the dataset, then we pre-process the data and create fake job postings detection model. At the end we used sentence similarities to show top 5 recommendations - the most similar job descriptions to the chosen one. 

# %%
print("[CODE_SAMPLE_COMPLETED_SUCCESSFULLY]")



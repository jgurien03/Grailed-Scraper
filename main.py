import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import time
import csv
import string
import sys
import random
import requests
import datefinder
import pickle
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver import Chrome
from selenium.webdriver.chrome.options import Options
from streamlit.components.v1 import html
from selenium.webdriver.chrome.service import Service
import seaborn as sns
import matplotlib.colors as mcolors
import io
from datetime import datetime, timedelta, date
from selenium.webdriver.support.ui import Select
from matplotlib.cm import ScalarMappable
from matplotlib import colors
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
import base64
import cv2
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from selenium.common.exceptions import NoSuchElementException
from keras.losses import CategoricalCrossentropy
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.layers import Dropout, BatchNormalization
from gensim.models import Word2Vec
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import nltk
import bs4
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch.optim as optim
from sklearn.neighbors import KNeighborsClassifier
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service as ChromeService
import os
import warnings
import tempfile
import urllib.request
from urllib.request import urlopen
import json
import path
from sklearn.cluster import KMeans
from nltk.corpus import wordnet
from PIL import Image, ImageOps
from PIL import UnidentifiedImageError
from io import BytesIO
import whisper
from selenium.webdriver.common.desired_capabilities import DesiredCapabilities
from playsound import playsound, PlaysoundException
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from keras.layers import Dense, Flatten
import locale
from keras.models import Sequential, load_model
import keras.utils as image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from keras.preprocessing.image import ImageDataGenerator
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from sklearn.model_selection import train_test_split
from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input
from keras.applications.xception import decode_predictions
from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from keras.applications.resnet import ResNet50, preprocess_input, decode_predictions
from keras.models import Model
from keras.layers import GlobalAveragePooling2D, Dense, Input
from keras.optimizers import Adam
from sklearn.preprocessing import LabelEncoder
from dateutil.relativedelta import relativedelta
import matplotlib.cm as cm
from gensim.models import LdaModel
from gensim.corpora import Dictionary
import statsmodels.api as sm
import streamlit as st

fashion_mnist = keras.datasets.fashion_mnist


warnings.filterwarnings("ignore")
model = whisper.load_model("base.en")
nltk.download('wordnet')

if __name__ == '__main__':
    pass


# Set a "base" URL to append onto
base_url = "https://www.grailed.com/designers/20471120"
COOKIES_PATH = "cookies.pkl"
time1 = random.randint(2, 6)
MP3_PATH = r'{}'.format(os.getcwd())

def login_to_grailed(username, password, brand, response):
    # Instantiate the WebDriver (e.g., Chrome driver)
    service = Service(executable_path=r'/usr/bin/chromedriver')
    options = webdriver.ChromeOptions()
    # options.add_argument('--headless')
    driver = webdriver.Chrome(service=service, options=options)
    login_url = "https://www.grailed.com/users/sign_up"
    driver.get(login_url)
    wait = WebDriverWait(driver, 20)

    try:
        element = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, '//a[@href="/users/sign_up"]'))
        )
        time.sleep(2)
        element.click()
    except:
        try:
            element = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.LINK_TEXT, "Log in"))
            )
            time.sleep(2)
            element.click()
        except:
            driver.quit()

    try:
        time.sleep(2)
        driver.find_element(By.CSS_SELECTOR, "button[data-cy='login-with-email']").click()
    except:
        driver.quit()

    wait.until(EC.element_to_be_clickable((By.ID, "email"))).send_keys(username)
    time.sleep(2)
    wait.until(EC.element_to_be_clickable((By.ID, "password"))).send_keys(password)

    try:
        element = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "button[data-cy='auth-login-submit']"))
        )
        time.sleep(2)
        element.click()
    except:
        driver.quit()

    try:
        submit_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "button[type='submit']"))
        )
        time.sleep(2)
        submit_button.click()
    except:
        if driver.current_url == 'https://www.grailed.com/':
            st.write("Logged into Grailed!")
        else:
            try:
                time.sleep(2)
                click_button(driver)
                time.sleep(2)
                driver.find_element(By.CLASS_NAME, "rc-audiochallenge-play-button").click()
                time.sleep(5)
                play_final(driver)
                time.sleep(5)
            except:
                driver.quit()

    navigate_to_brand(driver, brand, response)
    return 0

def navigate_to_brand(driver, brand, response):
    search_bar = driver.find_element(By.CSS_SELECTOR, "input#header_search-input")
    search_bar.clear()
    search_bar.send_keys(brand)
    search_bar.send_keys(Keys.RETURN)
    time.sleep(time1)
    if response.lower() == 'sold':
        try:
            WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, "//button/span[contains(text(), 'Filter')]"))).click()
        except TimeoutException:
            window_width = 800
            window_height = driver.get_window_size()['height']
            driver.set_window_size(window_width, window_height)
            WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, "//button/span[contains(text(), 'Filter')]"))).click()
        button = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, "//button[contains(@class, '-attribute-item') and span[contains(@class, '-attribute-header') and text()='Show Only']]"))).click()        
        checkbox = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.CSS_SELECTOR, "input.-toggle[name='sold']"))).click()
        time.sleep(1)
        auto_scroll(driver)
    else:
        auto_scroll(driver)
    return 0


def auto_scroll(driver):
    results = driver.find_elements(By.XPATH, '//div[@class="FiltersInstantSearch"]//div[@class="feed-item"]')
    len(results)
    last_height = driver.execute_script("return document.body.scrollHeight")
    while True:
        driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
        time.sleep(2)
        new_height = driver.execute_script("return document.body.scrollHeight")
        if new_height == last_height:
            break
        last_height = new_height
    results = driver.find_elements(By.XPATH, '//div[@class="FiltersInstantSearch"]//div[@class="feed-item"]')
    sort_results(driver, results)


def click_button(driver):
    driver.switch_to.default_content()
    driver.switch_to.frame(driver.find_element(By.XPATH, ".//iframe[@title='recaptcha challenge expires in two minutes']"))
    driver.find_element(By.ID, "recaptcha-audio-button").click()


def transcribe1(url, save_path):
    response = requests.get(url)
    if response.status_code == 200:
        filename = os.path.join(save_path, 'file.mp3')
        with open(filename, 'wb') as f:
            f.write(requests.get(url).content)
        print('Download complete.')
        assert os.path.isfile(filename)
        with open(filename, "r") as f:
            pass
        result = model.transcribe('file.mp3')
        return result["text"].strip()
    else:
        print('Failed to download the file.')
        return None


def play_final(driver):
    text = transcribe1(driver.find_element(By.ID, "audio-source").get_attribute('src'), MP3_PATH)
    driver.find_element(By.ID, "audio-response").send_keys(text)
    driver.find_element(By.ID, "recaptcha-verify-button").click()


def sort_results(driver, results):
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    titles1 = soup.find_all('p', {'data-cy': 'listing-title', 'class': 'ListingMetadata-module__title___Rsj55'})
    for title in titles1:
        titles.append(title.text)
    for WebElement in results:
        elementHTML = WebElement.get_attribute('outerHTML')
        element_soup = BeautifulSoup(elementHTML,'html.parser')
        price_element = element_soup.find(class_='ListingPriceAndHeart-module__listingPriceAndHeart___MEGdE')   
        if price_element:
            current_price_element = price_element.find(class_='Money-module__root___jRyq5', attrs={'data-testid': 'Current'})
            original_price_element = price_element.find(class_='Money-module__root___jRyq5 Price-module__original___I3r3D', attrs={'data-testid': 'Original'})      
            if current_price_element and original_price_element:
                item = current_price_element.text.strip()   
                old_item = original_price_element.text.strip()      
            elif current_price_element:
                item = current_price_element.text.strip()
                old_item = item
            prices.append(item)   
            old_prices.append(old_item) 
    for WebElement in results:
        elementHTML = WebElement.get_attribute('outerHTML')
        element_soup = BeautifulSoup(elementHTML,'html.parser')
        age_element = element_soup.find(class_='ListingAge-module__listingAge___EoWHC')
        if age_element:
            date_ago_element = age_element.find(class_='ListingAge-module__dateAgo___xmM8y')
            strike_through_element = age_element.find(class_='ListingAge-module__strikeThrough___LoORR')      
            if date_ago_element and strike_through_element:
                new_age = date_ago_element.text.strip()
                original_age = strike_through_element.text.strip()[1:-1]
                old_item = original_age         
            elif date_ago_element:
                new_age = date_ago_element.text.strip() 
                old_item = new_age  
            if '(' in new_age:
                index = new_age.index('(')
                new_age = new_age[:index]
            dates.append(new_age)
            old_dates.append(old_item)
    for WebElement in results:
        elementHTML = WebElement.get_attribute('outerHTML')
        element_soup = BeautifulSoup(elementHTML,'html.parser')
        image = element_soup.find('img')
        if image:
            image_url = image['src']
        listing_images.append(image_url)
    for link in soup.find_all('a', class_='listing-item-link'):
        href = link['href']
        listing_links.append(href)
    for WebElement in results:
        elementHTML = WebElement.get_attribute('outerHTML')
        element_soup = BeautifulSoup(elementHTML,'html.parser')
        size_element = element_soup.find('p', class_='ListingMetadata-module__size___e9naE')
        if size_element:
            size_text = size_element.get_text()
            sizes.append(size_text)


def predict_categories(model, image_paths):
    class_names = ['Accessories', 'Bottoms', 'Dresses', 'Outerwear', 'Shoes', 'Skirts', 'Tops']
    try:
        response = requests.get(image_paths)
        img = image.load_img(BytesIO(response.content), target_size=(150, 150))
        img = image.img_to_array(img)
        img = tf.expand_dims(img, axis=0)
        img = tf.keras.applications.xception.preprocess_input(img)
        predictions = model.predict(img)
        predicted_category = tf.argmax(predictions, axis=1)[0]
        return class_names[predicted_category]
    except UnidentifiedImageError:
        return "Unknown"


def clean_up_categories(cell):
    cell_lower = cell.lower()
    for category, keywords in categories.items():
        for keyword in keywords:
            if keyword in cell_lower:
                return category
            synonyms = wordnet.synsets(keyword)
            for synonym in synonyms:
                if synonym.lemmas()[0].name().lower() in cell_lower:
                    return category
    return 'Other'


def visualize_category_distribution(df):
    category_counts = df['Category'].value_counts()
    fig, ax = plt.subplots()
    ax.bar(category_counts.index, category_counts.values)
    ax.set_xlabel('Category')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Categories')
    ax.set_xticklabels(category_counts.index, rotation=45)
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format='png')
    img_buffer.seek(0)
    img_data = img_buffer.getvalue()
    st.pyplot(fig)
    return fig


def plot_category_prices(df, category_col):
    categories = df[category_col].unique()
    bar_positions = np.arange(len(categories))
    bar_width = 0.4
    category_prices = []
    category_volumes = []
    for category in categories:
        category_prices.append(df[df[category_col] == category]['Current Price'].mean())
        category_volumes.append(df[df[category_col] == category]['Category'].count())
    norm = colors.Normalize(vmin=min(category_volumes), vmax=max(category_volumes))
    category_volumes_norm = norm(category_volumes)
    cmap = plt.cm.get_cmap('viridis')  # Set the colormap to 'viridis'
    fig, ax = plt.subplots()
    ax.bar(bar_positions, category_prices, width=bar_width, color=cmap(category_volumes_norm))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Volume')
    ax.set_xticks(bar_positions)
    ax.set_xticklabels(categories, rotation=45)
    ax.set_ylabel('Price')
    ax.set_title('Average Price Comparison ($)')
    plt.tight_layout()
    st.pyplot(fig)
    return fig

def get_relative_date(time_string):
    current_datetime = datetime.now()
    if "Sold" in time_string:
        time_string = time_string.replace("Sold", "").strip()
    if "almost" in time_string:
        time_string = time_string.replace("almost", "").strip()
    if "over" in time_string:
        time_string = time_string.replace("over", "").strip()
    if "about" in time_string:
        time_string = time_string.replace("about", "").strip()
    if "minute" in time_string:
        minutes = int(time_string.split()[0])
        relative_date = current_datetime - timedelta(minutes=minutes)
    elif "hour" in time_string:
        hours = int(time_string.split()[0])
        relative_date = current_datetime - timedelta(hours=hours)
    elif "day" in time_string:
        days = int(time_string.split()[0])
        relative_date = current_datetime - timedelta(days=days)
    elif "month" in time_string:
        months = int(time_string.split()[0])
        relative_date = current_datetime - timedelta(days=months * 30)
    elif "year" in time_string:
        years = int(time_string.split()[0])
        relative_date = current_datetime - timedelta(days=years * 365)
    else:
        relative_date = current_datetime
    return relative_date


def plot_price_vs_upload_date(df, date):
    if (date == 'days' and any(df['Original Date'].str.contains('day'))) or ((not any(df['Original Date'].str.contains('month')) and not any(df['Original Date'].str.contains('year'))) and any(df['Original Date'].str.contains('days'))):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=29)
        df['Day'] = df['Relative Date'].dt.day
        avg_prices = []
        volumes = []
        for day in range(1, 31):
            mask = (df['Day'] == day) & (df['Relative Date'] >= start_date) & (df['Relative Date'] <= end_date)
            day_data = df[mask]
            if not day_data.empty:
                avg_price = day_data['Original Price'].mean()
                volume = day_data['Title'].count()
                avg_prices.append(avg_price)
                volumes.append(volume)
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax1.set_xlabel("Day")
        ax1.set_ylabel("Average Price")
        ax1.plot(range(1, len(avg_prices) + 1), avg_prices, color='blue', label='Average Price')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.set_xticks(range(1, len(avg_prices) + 1))
        ax1.set_xticklabels([f"Day {day}" for day in range(1, len(avg_prices) + 1)], rotation=45)
        ax2 = ax1.twinx()
        ax2.set_ylabel("Volume", color='red')
        ax2.plot(range(1, len(volumes) + 1), volumes, color='red', label='Volume')
        ax2.tick_params(axis='y', labelcolor='red')
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines + lines2, labels + labels2, loc='upper left')
        plt.tight_layout()
        st.pyplot(fig)
    elif date == 'months' and any(df['Original Date'].str.contains('month')):
        now = datetime.now()
        current_month = now.month
        current_year = now.year
        start_date = now - relativedelta(months=11)
        start_date = start_date.replace(day=1)
        end_date = now
        filtered_df = df[(df['Relative Date'] >= start_date) & (df['Relative Date'] <= end_date)]
        df_monthly_avg = filtered_df.groupby([filtered_df["Relative Date"].dt.year, filtered_df["Relative Date"].dt.month])["Original Price"].mean()
        df_monthly_avg = df_monthly_avg.sort_index(ascending=True)
        df_monthly_volume = filtered_df.groupby([filtered_df["Relative Date"].dt.year, filtered_df["Relative Date"].dt.month])["Title"].count()
        min_volume = df_monthly_volume.min()
        max_volume = df_monthly_volume.max()
        normalized_volume = (df_monthly_volume - min_volume) / (max_volume - min_volume)
        x_labels = [f"{month}-{year}" for year, month in df_monthly_avg.index]
        x_ticks = np.arange(len(df_monthly_avg))
        cmap = plt.cm.get_cmap('viridis')
        colors = cmap(normalized_volume)
        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(x_ticks, df_monthly_avg.values, color=colors)
        sm = ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=min_volume, vmax=max_volume))
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label('Volume')
        ax.set_xlabel("Month-Year")
        ax.set_ylabel("Average Price")
        ax.set_title("Monthly Average Price")
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
    elif date == 'years' and (any(df['Original Date'].str.contains('year')) or any(df['Original Date'].str.contains('years'))):
        df_yearly_avg = df.groupby(df["Relative Date"].dt.year)["Original Price"].mean()
        df_yearly_avg = df_yearly_avg.sort_index(ascending=True)
        df_yearly_volume = df.groupby(df["Relative Date"].dt.year)["Title"].count()
        df_yearly_volume = df_yearly_volume.sort_index(ascending=True)
        min_volume = df_yearly_volume.min()
        max_volume = df_yearly_volume.max()
        normalized_volume = (df_yearly_volume - min_volume) / (max_volume - min_volume)
        x_labels = df_yearly_avg.index
        x_ticks = np.arange(len(df_yearly_avg))
        cmap = plt.cm.get_cmap('viridis')
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_xlabel("Year")
        ax.set_ylabel("Average Price")
        for i, (price, volume) in enumerate(zip(df_yearly_avg.values, normalized_volume)):
            color = cmap(volume)
            ax.bar(x_ticks[i], price, color=color, alpha=0.7, edgecolor='black')
        colorbar = plt.cm.ScalarMappable(cmap='viridis')
        colorbar.set_array(normalized_volume)
        plt.colorbar(colorbar, ax=ax, label='Volume')
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
    else:
        fig, ax = plt.subplots()
        ax.axis('off')
        ax.text(0.5, 0.5, 'No data available', horizontalalignment='center', verticalalignment='center', fontsize=12)
        st.pyplot(fig)
    return fig
    
    
def filter_rows_by_keyword(df, keyword):
    df['Cleaned Title'] = df['Title'].str.lower().apply(lambda x: re.sub(r'[^\w\s]', '', x))
    cols = df.columns.tolist()
    cols.remove('Cleaned Title')
    cols.insert(1, 'Cleaned Title')
    df = df[cols]
    mask = df['Cleaned Title'].str.contains(keyword, case=False)
    filtered_df = df[mask]
    return filtered_df

def filter_dataframe_by_category(df, category_col, category):
    filtered_df = df[df[category_col] == category].copy()
    return filtered_df

def plot_price_by_size(df, size_col, price_col):
    grouped_df = df.groupby(size_col).agg({price_col: 'mean', size_col: 'count'})
    grouped_df.rename(columns={size_col: 'Volume'}, inplace=True)
    sorted_df = grouped_df.sort_values(price_col)
    cmap = plt.cm.get_cmap('viridis')
    normalized_volume = (sorted_df['Volume'] - sorted_df['Volume'].min()) / (sorted_df['Volume'].max() - sorted_df['Volume'].min())
    colors = cmap(normalized_volume)
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(sorted_df.index, sorted_df[price_col], color=colors)
    plt.xlabel('Size Category')
    plt.ylabel('Price')
    plt.title('Average Price by Size Category')
    sm = plt.cm.ScalarMappable(cmap=cmap)
    sm.set_array(sorted_df['Volume'])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('Volume')
    plt.xticks(rotation=45)
    plt.tight_layout()
    st.pyplot(fig)
    return fig


def new_dataset(df):
    df1 = df.copy()
    df_monthly_avg = df1.groupby([df1['Relative Date'].dt.year.rename('Year'), df1['Relative Date'].dt.month.rename('Month')])['Original Price'].mean().reset_index()
    df_monthly_volume = df1.groupby([df1['Relative Date'].dt.year.rename('Year'), df1['Relative Date'].dt.month.rename('Month')])['Title'].count().reset_index()
    df_monthly_volume = df_monthly_volume.rename(columns={'Title': 'Volume'})
    merged_df = pd.merge(df_monthly_avg, df_monthly_volume, on=['Year', 'Month'])
    merged_df['Date'] = pd.to_datetime(merged_df[['Year', 'Month']].assign(day=1))
    current_year = pd.Timestamp.today().year
    current_month = pd.Timestamp.today().month
    start_month = current_month - 14
    start_year = current_year
    if start_month <= 0:
        start_month += 12
        start_year -= 1
    end_month = current_month
    end_year = current_year
    start_date = pd.to_datetime(f'{start_year}-{start_month:02d}-01')
    end_date = pd.to_datetime(f'{end_year}-{end_month:02d}-01')
    date_range = pd.date_range(start=start_date, end=end_date, freq='MS')
    missing_dates = []
    new_rows = []
    for date in date_range:
        if not any((merged_df['Date'].dt.year == date.year) & (merged_df['Date'].dt.month == date.month)):
            missing_dates.append(date)
            year = date.year
            month = date.month
            lowest_price = merged_df['Original Price'].min()
            lowest_volume = merged_df['Volume'].min()
            # Define the range for fluctuations (e.g., -10% to +10%)
            fluctuation_range = 0.1  # 10%
            lower_limit = 1 - .5
            upper_limit = 1 + fluctuation_range
            # Generate random fluctuations around the lowest values
            random_price = np.random.uniform(lower_limit, upper_limit) * lowest_price
            random_volume = np.random.uniform(lower_limit, upper_limit) * lowest_volume
            # Use the randomly fluctuated values
            average_price = random_price
            average_volume = random_volume
            average_volume = random_volume
            new_row = {'Year': year, 'Month': month, 'Original Price': average_price, 'Volume': average_volume, 'Date': pd.to_datetime(date)}
            new_rows.append(new_row)

    merged_df = pd.concat([merged_df, pd.DataFrame(new_rows)], ignore_index=True)
    merged_df.sort_values(by='Date', inplace=True)
    return merged_df

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        _, (hidden_state, _) = self.lstm(input)
        output = self.fc(hidden_state[-1])
        return output

def train_lstm_model(dataset, look_back=12, hidden_size=50, epochs=100, batch_size=32, learning_rate=0.01):
    # Extract features (Year and Month)
    features = dataset[['Year', 'Month']].values

    # Normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalized_data = scaler.fit_transform(dataset[['Original Price', 'Volume']])

    # Create input-output pairs for LSTM
    X, y = [], []
    for i in range(len(normalized_data) - look_back):
        X.append(normalized_data[i:i+look_back])
        y.append(normalized_data[i+look_back][0])  # Price is at index 0
    X = np.array(X)
    y = np.array(y)

    # Adjust train-test split ratio
    split_index = int(len(X) * 0.8)
    X_train = X[:split_index]
    y_train = y[:split_index]
    X_test = X[split_index:]
    y_test = y[split_index:]

    # Convert data to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test = torch.tensor(X_test, dtype=torch.float32)
    y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

    # Define the model
    model = LSTMModel(X_train.shape[2], hidden_size, 1)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs.squeeze(), y_train)
        loss.backward()
        optimizer.step()

    # Evaluate the model
    model.eval()
    train_loss = criterion(model(X_train).squeeze(), y_train).item()
    test_loss = criterion(model(X_test).squeeze(), y_test).item()
    print("Training loss:", train_loss)
    print("Testing loss:", test_loss)

    return model, scaler

def add_months(start_date, num_months):
    month_offset = (start_date.month - 1 + num_months) % 12
    year_offset = (start_date.month - 1 + num_months) // 12
    new_month = month_offset + 1
    new_year = start_date.year + year_offset
    new_date = start_date.replace(year=new_year, month=new_month, day=1)
    return new_date


def generate_future_data(model, scaler, num_months, current_date, look_back=12, fluctuations=None):
    # Normalize the last available data
    current_sequence = scaler.transform(current_date[['Original Price', 'Volume']])
    # Extract the current year and month
    current_year = current_date['Year'].values[-1]
    current_month = current_date['Month'].values[-1]
    temp_year = current_year
    temp_month = current_month

    # Initialize the input sequence with the current data
    future_data = [current_sequence]
    model.eval()
    with torch.no_grad():
        torch.manual_seed(0)
        for _ in range(num_months):
            # Increment the month
            if current_month == 12:
                current_month = 1
                current_year += 1
            else:
                current_month += 1

            # Create the input tensor for prediction
            input_tensor = torch.tensor(future_data[-1], dtype=torch.float32).unsqueeze(0)
            input_tensor[:, -1, 0] = scaler.transform([[current_year, current_month]])[0][0]  # Set next month's year

            # Reshape and predict the next month's price
            predicted_price = model(input_tensor).squeeze(-1).tolist()[0]
            predicted_price = np.array([[predicted_price, 0]])  # Create a 2D array

            # Add random fluctuations to the predicted price
            if fluctuations is not None and _ < len(fluctuations):
                predicted_price = predicted_price * fluctuations[_]

            # Create the next input sequence by appending the predicted price
            next_sequence = np.concatenate((future_data[-1][1:, :], predicted_price), axis=0)
            future_data.append(next_sequence)

    # Inverse transform the generated future data
    future_data = np.array(future_data).reshape(-1, 2)
    future_data = scaler.inverse_transform(future_data)

    # Extract only the predicted prices and corresponding months for the requested number of months
    future_prices = future_data[-num_months:, 0].astype(int)
    start_date = datetime(temp_year, temp_month + 1, 1)
    future_months = [(add_months(start_date, i)).strftime('%B %Y') for i in range(num_months)]
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    future_prices_with_sign = [locale.currency(price, grouping=True) for price in future_prices]
    
    return future_months, future_prices_with_sign



titles = []
prices = []
old_prices = []
dates = []
old_dates = []
listing_images = []
listing_links = []
sizes = []

st.title("Grailed Data Visualizer")

if "page" not in st.session_state:
    st.session_state.page = 1

if st.session_state.page == 1:
    st.header("Page 1: Brand and Current/Sold Option")
    if "brand" not in st.session_state:
        st.session_state.brand = ""
    brand = st.text_input("What brand would you like to search?")
    if "response" not in st.session_state:
        st.session_state.response = ""
    response = st.selectbox("Would you like to look at current or sold listings?", ("current", "sold"))
    st.header("Page 2: Login")
    if "loading" not in st.session_state:
        st.session_state.loading = False
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        st.session_state.brand = brand
        st.session_state.response = response
        st.session_state.username = username
        st.session_state.password = password
        st.session_state.page = 2

if st.session_state.page == 2:
    st.header("Page 3: Await Results")
    if "scraping_done" not in st.session_state:
        with st.spinner("Collecting Data..."):
            login_to_grailed(st.session_state.username, st.session_state.password, st.session_state.brand, st.session_state.response)
            data = {
                'Title': titles,
                'Current Price': prices,
                'Original Price': old_prices,
                'Size': sizes,
                'Current Date': dates,
                'Original Date': old_dates,
                'Image Link': listing_images,
                'Listing Link': listing_links
            }

            categories = {
                    'Tops': ['shirt', 'blouse', 't-shirt', 'tee', 'long-sleeve', 'longsleeve', 'long sleeve', 'short sleeve', 'sweater', 'tank', 'tank top', 'top', 'button-up', 'button-down', 'vest', 'polo', 'crop-top', 'box logo', 'sweatshirt'],
                    'Bottoms': ['pants', 'jeans', 'flare', 'baggy', 'pant', 'cargo', 'talon-zip', 'dickies', 'painted denim', 'sweatpants', 'shorts', 'pleats please', 'joggers'],
                    'Skirts': ['maxi', 'skirt', 'mini-skirt', 'pleated skirt', 'mini skirt', 'midi', 'midi skirt'],
                    'Dresses': ['dress', 'gown'],
                    'Shoes': ['loafer', 'shoes', 'sneakers', 'boots', 'jordan', 'air force one', 'chuck 70', 'guidi', 'rick owens ramones', 'dunk', 'gucci slides'],
                    'Outerwear': ['Fur leather', 'Half zip', 'Quarter zip', 'Suit', 'Outerwear', 'Jacket', 'Puffer', 'jacket', 'coat', 'blazer', 'bomber', 'trenchcoat', 'trucker jacket', 'hoodie', 'zip-up', 'pullover', 'windbreaker', 'cardigan', 'Denim Trucker Jacket'],
                    'Accessories': ['Sunglasses', 'Apron', 'Necklace', 'Watch', 'Socks', 'Tie', 'Bow tie', 'Purse', 'Ring', 'Gloves', 'belt', 
                                'Scarf', 'Umbrella', 'Boots', 'Mittens', 'Stockings', 'Earmuffs', 'Hair band', 'Safety pin', 'Watch', 'Hat', 'Beanie', 'Cap', 'Beret', 'card holder', 'Straw hat', 'Derby hat', 'Helmet', 'Top hat', 'Mortar board']
            }

            df = pd.DataFrame(data)
            df.insert(1,"Category", " ")
            model = load_model('model.h5')
            df['Category'] = df['Title'].apply(clean_up_categories)
            df = filter_rows_by_keyword(df, brand)
            for index, row in df.iterrows():
                size = row['Size']
                if isinstance(size, str) and size.lower() == 'os':
                    df.at[index, 'Category'] = 'Accessories'
                elif isinstance(size, (int, float)) and 22 <= size <= 50:
                    df.at[index, 'Category'] = 'Bottoms'
                elif isinstance(size, (int, float)) and 4 <= size <= 15:
                        df.at[index, 'Category'] = 'Shoes'
            for index, item in df.iterrows():
                if df.at[index, 'Category'] == 'Other':
                    df.at[index, 'Category'] = predict_categories(model, item['Image Link'])
                    print(df.at[index, 'Category'])
                    print('grailed.com' + df.at[index, 'Listing Link'])
            df['Current Price'] = df['Current Price'].str.replace('[^\d.]', '', regex=True)
            df['Current Price'] = pd.to_numeric(df['Current Price'])
            df['Original Price'] = df['Original Price'].str.replace('[^\d.]', '', regex=True)
            df['Original Price'] = pd.to_numeric(df['Original Price'])
            df['Relative Date'] = df['Original Date'].apply(get_relative_date)
            st.session_state.df = df
            st.session_state.scraping_done = True
            st.write('Done!')
        if st.button("View Category Distribution"):
            st.session_state.page = 3
    else:
        st.session_state.page = 3

if st.session_state.page == 3:
    st.header("Page 4: Category Distribution")
    fig = visualize_category_distribution(st.session_state.df)
    canvas = FigureCanvas(fig)
    buffer = io.BytesIO()
    canvas.print_png(buffer)
    img_data = base64.b64encode(buffer.getvalue()).decode()
    st.markdown(
        f'<a href="data:image/png;base64,{img_data}" download="category_distribution.png">Download Category Distribution</a>',
        unsafe_allow_html=True
    )
    if st.button("View Prices by Category"):
        st.session_state.page = 4
if st.session_state.page == 4:
    st.header("Page 5: Prices by Category")
    fig = plot_category_prices(st.session_state.df, 'Category')
    
    img_path = "prices_by_category.png"
    fig.savefig(img_path, bbox_inches="tight")
        
    with open(img_path, "rb") as f:
        img_data = f.read()
        img_base64 = base64.b64encode(img_data).decode()
        st.markdown(
            f'<a href="data:image/png;base64,{img_base64}" download="prices_by_category.png">Download Prices by Category</a>',
            unsafe_allow_html=True
        )
    os.remove(img_path)
    if st.button("View Prices over Time"):
        st.session_state.page = 5
if st.session_state.page == 5:
    st.header("Page 6: Prices over Time")
    date = st.selectbox("Would you like to search by days, months, or years?", ("days", "months", "years"))
    if date.lower() in ['days', 'months', 'years']:
        response4 = st.selectbox("Would you like to only visualize a certain category of brand?", ("Yes", "No"))
        if response4.lower() == 'yes':
            response5 = st.selectbox("Select a category:", ("Tops", "Bottoms", "Skirts", "Dresses", "Shoes", "Outerwear", "Accessories"))
            df2 = filter_dataframe_by_category(st.session_state.df, 'Category', response5)
            fig = plot_price_vs_upload_date(df2, date)
            img_path = "prices_over_time.png"
            fig.savefig(img_path, bbox_inches="tight")
            with open(img_path, "rb") as f:
                img_data = f.read()
                img_base64 = base64.b64encode(img_data).decode()
                st.markdown(
                    f'<a href="data:image/png;base64,{img_base64}" download="prices_over_time.png">Download Prices over Time</a>',
                    unsafe_allow_html=True
                )
            os.remove(img_path)
        else:
            fig = plot_price_vs_upload_date(st.session_state.df, date)
            img_path = "prices_over_time.png"
            fig.savefig(img_path, bbox_inches="tight")
            with open(img_path, "rb") as f:
                img_data = f.read()
                img_base64 = base64.b64encode(img_data).decode()
                st.markdown(
                    f'<a href="data:image/png;base64,{img_base64}" download="prices_over_time.png">Download Prices Over Time</a>',
                    unsafe_allow_html=True
                )
            os.remove(img_path)
        if st.button("View Prices by Size"):
            st.session_state.page = 6
if st.session_state.page == 6:
    st.header("Page 7: Price by Size")
    response7 = st.selectbox("Would you like to only visualize a certain category of brand?", ("Yes", "No"), key="response7")
    if response7.lower() == 'yes':
        response8 = st.selectbox("Select a category:", ("Tops", "Bottoms", "Skirts", "Dresses", "Shoes", "Outerwear", "Accessories"), key="response8")
        df3 = filter_dataframe_by_category(st.session_state.df, 'Category', response8)
        fig = plot_price_by_size(df3, 'Size', 'Original Price')
    else:
        fig = plot_price_by_size(st.session_state.df, 'Size', 'Original Price')
    img_path = "price_by_size.png"
    fig.savefig(img_path, bbox_inches="tight")
    with open(img_path, "rb") as f:
        img_data = f.read()
        img_base64 = base64.b64encode(img_data).decode()
        st.markdown(
            f'<a href="data:image/png;base64,{img_base64}" download="price_by_size.png">Download Prices by Size</a>',
            unsafe_allow_html=True
        )
    os.remove(img_path)
    if st.button("Predict Future Prices"):
            st.session_state.page = 7
if st.session_state.page == 7:
    df4 = st.session_state.df
    st.header("Page 8: Predict Future Price for Brand")
    num_months = st.number_input("Enter the number of future months to see prices (1-12)", min_value=1, max_value=12, value=1, step=1)
    fluctuation_range = 0.1
    if st.button("Predict"):
        dataset = new_dataset(df4)
        if 'fluctuations' not in st.session_state:
            fluctuations = np.random.uniform(1 - fluctuation_range, 1 + fluctuation_range, size=num_months)
            st.session_state['fluctuations'] = fluctuations
        else:
            fluctuations = st.session_state['fluctuations']
        look_back = 12
        model_filename = "model.pt"
        if os.path.isfile(model_filename):
            model, scaler = torch.load(model_filename)
        else:
            model, scaler = train_lstm_model(dataset, look_back=look_back)
            torch.save((model, scaler), model_filename)
        last_available_data = dataset[-look_back:]
        predicted_months, predicted_prices = generate_future_data(model, scaler, num_months, last_available_data, look_back=look_back, fluctuations=fluctuations)
        combined_data = list(zip(predicted_months, predicted_prices))
        final = pd.DataFrame(combined_data, columns=['Month', 'Price'])
        col1, col2 = st.columns([1, 1.5])
        # Display the DataFrame in the first column
        with col1:
            st.write(final)
            def get_dataframe_download_link():
                csv = final.to_csv(index=False)
                csv = csv.encode()
                return csv
            dataframe_link = get_dataframe_download_link()
            st.download_button(label='Download DataFrame', data=dataframe_link, file_name='dataframe.csv', mime='text/csv')

        # Display the bar graph in the second column
        with col2:
            final['Month'] = pd.to_datetime(final['Month'])

            # Set the figure size
            fig, ax = plt.subplots(figsize=(8, 6))  # Adjust the width and height as desired
            ax.plot(final['Month'], final['Price'].str.replace('$', '').astype(float), marker='o')
            ax.set_xlabel('Month')
            ax.set_ylabel('Price ($)')
            ax.set_title('Future Monthly Prices', fontsize=16)
            plt.xticks(rotation=45)

            # Set the x-axis limits
            ax.set_xlim(final['Month'].min(), final['Month'].max())

            plt.tight_layout()
            st.pyplot(fig)
            def get_graph_download_link():
                buf = io.BytesIO()
                plt.savefig(buf, format='png')
                buf.seek(0)
                return buf
            download_link = get_graph_download_link()
            st.download_button(label='Download Graph', data=download_link, file_name='graph.png', mime='image/png')

    if st.button("View Raw DataFrame of Brand"):
            st.session_state.page = 8
if st.session_state.page == 8:
    st.header("Page 9: View DataFrame")
    model_filename = "model.pt"
    if os.path.isfile(model_filename):
        os.remove(model_filename)
    st.write(st.session_state.df)
    def get_dataframe_download_link():
        csv = st.session_state.df.to_csv(index=False)
        csv = csv.encode()
        return csv
    dataframe_link = get_dataframe_download_link()
    st.download_button(label='Download DataFrame', data=dataframe_link, file_name='raw_dataframe.csv', mime='text/csv')






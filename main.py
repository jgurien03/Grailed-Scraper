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
from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver import Chrome
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
import seaborn as sns
import matplotlib.colors as mcolors
from datetime import datetime, timedelta, date
from selenium.webdriver.support.ui import Select
from matplotlib.cm import ScalarMappable
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
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
from keras.models import Sequential, load_model
import keras.utils as image
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from keras.preprocessing.image import ImageDataGenerator
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
import matplotlib.cm as cm
from gensim.models import LdaModel
from gensim.corpora import Dictionary

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

def login_to_grailed(username, password):
    # Instantiate the WebDriver (e.g., Chrome driver)
    service = Service(executable_path=r'/usr/bin/chromedriver')
    options = webdriver.ChromeOptions()
    #options.add_argument('--headless')
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
        print("success in clicking Login in button")
    except:
        print("Bot could not click on login button.")
        driver.quit()
    try:
        time.sleep(2)
        driver.find_element(By.CSS_SELECTOR, "button[data-cy='login-with-email']").click()
        print("success in clicking Login in button")
    except:
        print("Bot could not click on login button.")
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
        print("success!")
    except:
        driver.quit()
        print("failed!")
    try:
        submit_button = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "button[type='submit']"))
        )
        time.sleep(2)
        submit_button.click()
    except:
        if driver.current_url == 'https://www.grailed.com/':
            print("success!")
        else:
            try:
                time.sleep(2)
                click_button(driver)
                print('success!!!!')
                time.sleep(2)
                driver.find_element(By.CLASS_NAME, "rc-audiochallenge-play-button").click()
                time.sleep(5)
                play_final(driver)
                time.sleep(5)
            except:
                print('failed!')
                driver.quit()
    navigate_to_brand(driver, brand)
    return 0


def navigate_to_brand(driver, brand):
    search_bar = driver.find_element(By.CSS_SELECTOR, "input#header_search-input")
    search_bar.clear()
    search_bar.send_keys(brand)
    search_bar.send_keys(Keys.RETURN)
    time.sleep(time1)
    if response.lower() == 'sold':
        time.sleep(200)
        button = WebDriverWait(driver, 10).until(EC.element_to_be_clickable((By.XPATH, "//button/span[contains(text(), 'Filter')]"))).click()
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
        time.sleep(1.5)
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
    response = requests.get(image_paths)
    img = image.load_img(BytesIO(response.content), target_size=(150, 150))
    img = image.img_to_array(img)
    img = tf.expand_dims(img, axis=0)
    img = tf.keras.applications.xception.preprocess_input(img)
    predictions = model.predict(img)
    predicted_category = tf.argmax(predictions, axis=1)[0]
    return class_names[predicted_category]


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
    plt.bar(category_counts.index, category_counts.values)
    plt.xlabel('Category')
    plt.ylabel('Count')
    plt.title('Distribution of Categories')
    plt.xticks(rotation=45)
    plt.show()


def plot_category_prices(df, category_col):
    categories = df[category_col].unique()
    bar_positions = np.arange(len(categories))
    bar_width = 0.4
    category_prices = []
    for category in categories:
        category_prices.append(df[df[category_col] == category]['Current Price'].mean())
    plt.bar(bar_positions, category_prices, width=bar_width)
    plt.xticks(bar_positions, categories)
    plt.ylabel('Price')
    plt.title('Average Price Comparison ($)')
    plt.show()

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
        plt.show()
    elif date == 'months' and any(df['Original Date'].str.contains('month')):
        df_monthly_avg = df.groupby([df["Relative Date"].dt.year, df["Relative Date"].dt.month])["Original Price"].mean()
        df_monthly_avg = df_monthly_avg.sort_index(ascending=True)
        df_monthly_volume = df.groupby([df["Relative Date"].dt.year, df["Relative Date"].dt.month])["Title"].count()
        df_monthly_volume = df_monthly_volume.sort_index(ascending=True)
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
        plt.show()
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
        plt.colorbar(colorbar, label='Volume')
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels, rotation=45)
        plt.tight_layout()
        plt.show()
    else:
        print('Not enough data to show')
    
    
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
    plt.show()


titles = []
prices = []
old_prices = []
dates = []
old_dates = []
listing_images = []
listing_links = []
sizes = []
username = "dazzlesdaddy@gmail.com"
password = "Jakey050603#"
brand = input("What brand would you like to search? ")
response = input("Would you like to look at current or sold listings? (current/sold) ")
login_to_grailed(username, password)

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
        'Shoes': ['shoes', 'sneakers', 'boots', 'jordan', 'air force one', 'chuck 70', 'guidi', 'rick owens ramones', 'dunk', 'gucci slides'],
        'Outerwear': ['Outerwear', 'Jacket', 'Puffer', 'jacket', 'coat', 'blazer', 'bomber', 'trenchcoat', 'trucker jacket', 'hoodie', 'zip-up', 'pullover', 'windbreaker', 'cardigan', 'Denim Trucker Jacket'],
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
response1 = input("Would you like to see each category and their feed appearance count? (yes/no): ")
if response1.lower() == 'yes':
    visualize_category_distribution(df)
else:
    print("Graph display skipped.")
response2 = input("Would you like to see each category and their prices compared against each other? (yes/no): ")
if response2.lower() == 'yes':
    plot_category_prices(df, 'Category')
else:
    print("Graph display skipped.")
response3 = input("Would you like to see the brand's change in price over time? (yes/no): ")
if response3.lower() == 'yes':
    df2 = df
    date = input("Would you like to search by days, months, or years? (days/months/years) ")
    if date.lower() == 'days' or date.lower() =='months' or date.lower() == 'years':
        response4 = input("Would you like to only visualize a certain category of brand? (yes/no): ")
        if response4.lower() == 'yes':
            response5 = input("Select a category: (Tops, Bottoms, Skirts, Dresses, Shoes, Outerwear, Accessories) ")
            df2 = filter_dataframe_by_category(df, 'Category', response5)
            plot_price_vs_upload_date(df2, date)
        else:
            plot_price_vs_upload_date(df2, date)
    else:
        print('Must specify days, months, or years')
else:
    print("Graph display skipped.") 
response6 = input("Would you like to see each size and their average price? (yes/no): ")
if response6.lower() == 'yes':
    df3 = df
    response7 = input("Would you like to only visualize a certain category of brand? (yes/no): ")
    if response7.lower() == 'yes':
        response8 = input("Select a category: (Tops, Bottoms, Skirts, Dresses, Shoes, Outerwear, Accessories) ")
        df3 = filter_dataframe_by_category(df, 'Category', response8)
        plot_price_by_size(df3, 'Size', 'Original Price')
    else:
        plot_price_by_size(df3, 'Size', 'Original Price')
else:
    print("Graph display skipped.")






#!/usr/bin/env python3

import os
import chardet
import pandas as pd
import numpy as np
from openai import OpenAI

client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get("OPENAI_API_KEY"),
)

def get_embedding(text, model="text-embedding-3-small"):
    text = text.replace("\n", " ")
    response = client.embeddings.create(
        input=[text],
        model=model
    )
    return response.data[0].embedding[0]

def normalize_distance(df, columns, max_values):
    normalized_df = df.copy()
    
    for column in columns:
        normalized_df[column] = normalized_df[column] / max_values[column]
        
    return normalized_df

def get_inverse_number(number):
    if number == 0:
        return 1
    else:
        return 1 / number
    
def find_similar_food(df, normalized_df, protein, energy, fat, carbohydrate, max_values, top_n=10):
    # 입력된 값을 정규화
    normalized_protein = protein / max_values['단백질(g)']
    normalized_energy = energy / max_values['에너지(㎉)']
    normalized_fat = fat / max_values['지방(g)']
    normalized_carbohydrate = carbohydrate / max_values['탄수화물(g)']

    # 정규화된 값을 기준으로 유클리디안 거리 계산
    normalized_df['distance'] = np.sqrt((normalized_df['단백질(g)'] - normalized_protein) ** 2 +
                                        (normalized_df['에너지(㎉)'] - normalized_energy) ** 2 +
                                        (normalized_df['지방(g)'] - normalized_fat) ** 2 +
                                        (normalized_df['탄수화물(g)'] - normalized_carbohydrate) ** 2)
    
    # 거리가 가장 작은 상위 top_n개의 데이터 추출
    similar_indices = normalized_df.nsmallest(top_n, 'distance').index
    similar_data = df.loc[similar_indices].copy()
    similar_data['distance'] = normalized_df.loc[similar_indices, 'distance']
    similar_data['sim_score'] = similar_data['distance'].apply(get_inverse_number)

    return similar_data

def read_data(file_path):
    with open(file_path, 'rb') as file:
        content = file.read()
        result = chardet.detect(content)
    encoding = result['encoding']
    
    try:
        data = pd.read_csv(file_path, encoding=encoding, low_memory=False)
    except UnicodeDecodeError:
        data = pd.read_csv(file_path, encoding='cp949', low_memory=False)
    
    return data

def make_combined_data():

    # List of file names
    file_names = ['가공식품1.csv', '가공식품2.csv', '가공식품3.csv', '가공식품4.csv', '가공식품5.csv',
                '가공식품6.csv', '가공식품7.csv', '가공식품8.csv', '농축.csv', '음식.csv']

    # Directory path where the files are located
    directory = './ingredient/'

    # Initialize an empty list to store the DataFrames
    dataframes = []

    # Iterate over the file names and read each file into a DataFrame
    for file_name in file_names:
        file_path = os.path.join(directory, file_name)
        print(f"Reading file: {file_name}")
        try:
            df = read_data(file_path)
            df = df[['식품코드', 'DB군', '상용제품', '식품명', '식품대분류', '총 포화 지방산(g)',
                     '식품상세분류', '1회제공량', '에너지(㎉)', '나트륨(㎎)', '콜레스테롤(㎎)',
                     '단백질(g)', '지방(g)', '탄수화물(g)', '총당류(g)']]
            dataframes.append(df)
        except Exception as e:
            print(f"Error reading file: {file_name}")
            print(f"Error message: {str(e)}")
            print("Skipping this file.")

    # Concatenate all the successfully read DataFrames vertically
    combined_data = pd.concat(dataframes, ignore_index=True)

    # Print the first few rows of the combined DataFrame
    print(combined_data.head())

    # Optionally, you can save the combined DataFrame to a new CSV file
    combined_data.to_csv('./ingredient/combined_data.csv', index=False)

    return 

def read_combined_data():
    combined_data = pd.read_csv('./ingredient/combined_data.csv')

    print(combined_data.head())

    return combined_data

def convert_to_numeric(df, columns):
    for column in columns:
        df[column] = pd.to_numeric(df[column], errors='coerce')
    return df

# make_combined_data()

df = read_combined_data()

columns_to_convert = ['단백질(g)', '에너지(㎉)', '지방(g)', '탄수화물(g)']

# 데이터프레임의 해당 열을 숫자 타입으로 변환
df = convert_to_numeric(df, columns_to_convert)

# 변환 후 결측값 처리 (선택 사항)
df = df.dropna(subset=columns_to_convert)

# 각 열의 최댓값 설정
max_values = {
    '단백질(g)': 55,
    '에너지(㎉)': 2600,
    '지방(g)': 55,
    '탄수화물(g)': 324
}

# 각 열의 값을 최댓값으로 나누어 정규화한 데이터프레임 생성
normalized_df = normalize_distance(df, columns_to_convert, max_values)

# 예시 값 (단백질, 에너지, 지방, 탄수화물)
protein = 13
energy = 600
fat = 32
carbohydrate = 123

# 유사한 데이터 추출
similar_data = find_similar_food(df, normalized_df, protein, energy, fat, carbohydrate, max_values, top_n=2)

#나트륨, 콜레스테롤, 포화지방산 데이터 전처리
similar_data['나트륨(㎎)'] = similar_data['나트륨(㎎)'].replace('-', 0)
similar_data['콜레스테롤(㎎)'] = similar_data['콜레스테롤(㎎)'].replace('-', 0)
similar_data['총 포화 지방산(g)'] = similar_data['총 포화 지방산(g)'].replace('-', 0)

similar_data['나트륨(㎎)'] = similar_data['나트륨(㎎)'].str.replace(',', '').astype(float)
similar_data['콜레스테롤(㎎)'] = similar_data['콜레스테롤(㎎)'].str.replace(',', '').astype(float)
similar_data['총 포화 지방산(g)'] = similar_data['총 포화 지방산(g)'].str.replace(',', '').astype(float)

#유해성분(나트륨, 콜레스테롤, 포화지방산)이 낮은 순으로 점수 매기기(높을수록 좋음)
similar_data['nutrient_score'] = similar_data['sim_score'] - similar_data['나트륨(㎎)']/10 - similar_data['콜레스테롤(㎎)']/100 - similar_data['총 포화 지방산(g)']/10


# nutrient_score가 높은 순으로 정렬
similar_foods = similar_data.sort_values(by='nutrient_score', ascending=False)
similar_foods['food_embedding'] = similar_foods['식품명'].apply(lambda x: get_embedding(x, model='text-embedding-3-large'))
#print(df['식품명'])
#print(get_embedding(df['식품명'], model='text-embedding-3-large'))
print(similar_foods[['식품명', 'food_embedding', 'sim_score', 'nutrient_score', '나트륨(㎎)', '콜레스테롤(㎎)', '총 포화 지방산(g)']])
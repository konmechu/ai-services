from PIL import Image
from io import BytesIO
import torch
import torchvision.transforms as transforms
from torchvision import models
from torch import nn
import requests
from flask import Flask, make_response, jsonify, request
import mysql.connector
from collections import Counter
import requests
from scipy.spatial import distance
import pandas as pd
from datetime import date
import datetime

# Initialize Flask app
app = Flask(__name__)

# # Initialize the model just like in your inference.py
model = models.resnet50()
model.fc = nn.Linear(model.fc.in_features, 4)
model.load_state_dict(torch.load('./AI/resnet_params.pth', map_location=torch.device('cpu')))
model.eval()


# # Define the transformation
transform = transforms.Compose([
    transforms.Resize((1500, 1000)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# # Define class labels
class_labels_dict = {0: '베이글', 1: '보쌈', 2: '복숭아', 3: '부침개'}

# image_path = './image/8.jpg'  # replace 'your_image.jpg' with the name of your image file
# image = Image.open(image_path)
# input_tensor = transform(image)
# input_batch = input_tensor.unsqueeze(0)  # create a mini-batch as expected by the model
@app.route('/api/recommend', methods=['GET'])

def recommend_food():
    conn = mysql.connector.connect(
    host='konmecu.cgb6blfactua.ap-northeast-2.rds.amazonaws.com',  # Host name
    user='konmecu',  # Username
    password='konmecu123',  # Password
    database='konmecu'  # Database name
)
    
    # Create a cursor to execute queries
    cursor = conn.cursor()

    # query = """
    #         SELECT menu_id
    #         FROM nutrition;
    #     """

    # # Execute the query
    # cursor.execute(query)

    # # Fetch all results from the cursor
    # food = cursor.fetchall()

    # print(food)

    # query = """
    #         SELECT menu_id
    #         FROM menu WHERE DATE(date) = CURDATE();
    #     """

    # # Execute the query
    # cursor.execute(query)

    # # Fetch all results from the cursor
    # food = cursor.fetchall()

    # print(food)
    #m.food, n.calories, n.protein, n.fat, n.carbs, n.fiber
    # SQL query to join the tables and fetch nutrition for today's food
    query = """
            SELECT m.food, n.protein, n.fat, n.carbs
            FROM menu m
            JOIN nutrition n ON m.menu_id = n.menu_id
            WHERE DATE(m.date) = '2023-11-17';
        """

    # Execute the query
    cursor.execute(query)

    # Fetch all results from the cursor
    todays_nutrition = cursor.fetchall()

    print('todays_nutrition')
    print(todays_nutrition)

    food_words = []
    eaten = [0, 0, 0]
    for item in todays_nutrition:
        food, protein, fat, carbs = item
        print(f"Food: {food}, Protein: {protein}g, Fat: {fat}g, Carbs: {carbs}g")
        eaten[0] += carbs
        eaten[1] += protein
        eaten[2] += fat
        # Split the food string into words and add them to the food_words list
        words = food.split()  # Splits the string into words
        food_words.extend(words)  # Adds the words to the food_words list

    word_count = Counter(food_words)
    # most_common_words = word_count.most_common(1)[0][0]
    most_common_words = '미역'
    print(f'most_common_words: {most_common_words}')
    
    
    base_url = "http://openapi.foodsafetykorea.go.kr/api/"
    key_id = "8949399a8f1247669991"  # Replace with your API key
    service_id = "I2790"  # Replace with the service ID if different
    data_type = "json"
    start_idx = "1"
    end_idx = "10"

    

    # Construct the URL
    url = f"{base_url}{key_id}/{service_id}/{data_type}/{start_idx}/{end_idx}/DESC_KOR={most_common_words}"
    # Send a GET request to the API
    response = requests.get(url)

    # Check for a successful response
    if response.status_code == 200:
        print("Request was successful!")
    else:
        print(f"Failed to retrieve data: {response.status_code}")

    # Parse the JSON response
    data = response.json()
    # Assuming the data is stored in a variable called data
    foods = data['I2790']['row']

    # Assuming 'foods' is a list of dictionaries containing the food data
    # Create an empty list to store the DataFrame rows
    df_rows = []

    def to_float(value):
        return float(value) if value else 0.0

    # Iterate over each food item to create a DataFrame row
    for food in foods:
        # Extract the name of the food
        food_name = food['DESC_KOR']

        # Convert the essential nutrients to floats and create an array
        essential_nutrients = [
            4*to_float(food['NUTR_CONT2']), 
            4*to_float(food['NUTR_CONT3']), 
            4*to_float(food['NUTR_CONT4'])
        ]
        # Create a row with the food name, essential nutrients, and the remaining nutrients
        # Convert nutrient values to float, using zero where the string is empty
        row = {
            'Food': food_name,
            'Essential Nutrients': essential_nutrients,
            'Calories': to_float(food['NUTR_CONT1']),
            'Sugars': to_float(food['NUTR_CONT5']),
            'Sodium': to_float(food['NUTR_CONT6']),
            'Cholesterol': to_float(food['NUTR_CONT7']),
            'Saturated Fatty Acids': to_float(food['NUTR_CONT8']),
            'Trans Fats': to_float(food['NUTR_CONT9'])
        }

        # Append the row to the list of rows
        df_rows.append(row)

    df = pd.DataFrame(df_rows)

    print(eaten)
    daily_nutrients = [130, 70, 51]
    
    required_nutrients = []

    for nutrient1,nutrient2  in zip(daily_nutrients, eaten):
        required_nutrients.append(nutrient1 - nutrient2)
    
    # df['Distance'] = df['Essential Nutrients'].apply(lambda x: distance.euclidean(x, required_nutrients))

    # closest_food = df.loc[df['Distance'].idxmin()]

    # differences = [required - found for required, found in zip(required_nutrients, closest_food['Essential Nutrients'])]

    # positive_differences = [diff if diff >= 0 else float('inf') for diff in differences]

    # min_diff_index = positive_differences.index(min(positive_differences))

    nutrient_names = ['탄수화물', '단백질', '지방']  
    # selected_nutrient = nutrient_names[min_diff_index]

    # required_amount = required_nutrients[min_diff_index]
    # found_amount = closest_food['Essential Nutrients'][min_diff_index]
    # Step 1: Find the index of the lowest required nutrient
    # Step 1: Calculate the percentage missed for each nutrient
    percentage_missed = [((required / daily) * 100) if daily > 0 else 0 for required, daily in zip(required_nutrients, daily_nutrients)]

    # Step 2: Find the nutrient with the highest percentage missed
    highest_missed_index = percentage_missed.index(max(percentage_missed))
    selected_nutrient = nutrient_names[highest_missed_index]
    required_amount_percentage = percentage_missed[highest_missed_index]

    # Step 3: Adjust the recommendation logic
    df['Distance'] = df['Essential Nutrients'].apply(lambda x: distance.euclidean(x, required_nutrients))
    closest_food = df.loc[df['Distance'].idxmin()]

    remaining_data = df.drop(df['Distance'].idxmin())

# 이 데이터에서 가장 작은 'Distance' 값을 가진 행을 찾음
    second_closest_food = remaining_data.loc[remaining_data['Distance'].idxmin()]


    found_amount = closest_food['Essential Nutrients'][highest_missed_index]
    if required_amount_percentage < 0:
        response = {'recommend': f"오늘 {selected_nutrient}을 권장 섭취량보다 더 섭취하셨군요!\n {most_common_words} 애호가 당신!\n {selected_nutrient}이 적은 {closest_food['Food']} 어떠신가요?"}
    elif required_amount_percentage > 30:
        response = {'recommend': f"오늘 {selected_nutrient}이 많이 부족하시군요!! \n {most_common_words} 애호가 당신!\n {selected_nutrient}이 풍부한 {closest_food['Food']} 또는 {second_closest_food['Food']} 어떠신가요?"}
    else:
        response = {'recommend': f"오늘 {selected_nutrient}이 약간 부족하시군요!\n {most_common_words} 애호가 당신!\n {selected_nutrient}이 {found_amount}g 포함된 {closest_food['Food']} 어떠신가요?"} 

    print(response)
    response = make_response(jsonify(response))
    response.headers['Content-Type'] = 'application/json; charset=utf-8'

    return response, 200

@app.route('/api/infer', methods=['GET'])
def infer_image_get():
    print("GET Endpoint hit")
    return "GET Endpoint hit", 200  # 200 is the status code for OK

@app.route('/api/menu', methods=['POST'])
def add_menu():
    pass


@app.route('/api/infer', methods=['POST'])
def infer_image_post():
    print("Image has been successfully received.")
    # Get the image from the POST request
    image_file = request.files['image']
    image = Image.open(BytesIO(image_file.read()))

    print("Image recieved")


    # Transform and predict
    input_tensor = transform(image)
    input_batch = input_tensor.unsqueeze(0)

    with torch.no_grad():
        output = model(input_batch)
    _, predicted = torch.max(output, 1)
    predicted_label = class_labels_dict[predicted.item()]

    print(f'result of model: {predicted_label}')

    # 2. API 이용해 영양성분 출력
    # Define the API endpoint and parameters
    base_url = "http://openapi.foodsafetykorea.go.kr/api/"
    key_id = "8949399a8f1247669991"  # Replace with your API key
    service_id = "I2790"  # Replace with the service ID if different
    data_type = "json"
    start_idx = "1"
    end_idx = "5"
    desc_kor = predicted_label

    # Construct the URL
    url = f"{base_url}{key_id}/{service_id}/{data_type}/{start_idx}/{end_idx}/DESC_KOR={desc_kor}"
    # Send a GET request to the API
    res = requests.get(url)

    # Check for a successful response
    if res.status_code == 200:
        print("Request was successful!")
    else:
        print(f"Failed to retrieve data: {res.status_code}")

    # Parse the JSON response
    data = res.json()
    # Assuming the data is stored in a variable called data
    first_item = data['I2790']['row'][0]
    # For now, let's return a mockup food_info
    response = {
        '식품명': predicted_label,
        '열량(kcal)': first_item['NUTR_CONT1'],
        '탄수화물(g)': first_item['NUTR_CONT2'],
        '단백질(g)': first_item['NUTR_CONT3'],
        '지방(g)': first_item['NUTR_CONT4'],
        '당류(g)': first_item['NUTR_CONT5'],
        '나트륨(mg)': first_item['NUTR_CONT6'],
        '콜레스테롤(mg)': first_item['NUTR_CONT7'],
        '포화지방산(g)': first_item['NUTR_CONT8'],
        '트랜스지방(g)': first_item['NUTR_CONT9']
    }

    response = make_response(jsonify(response))
    response.headers['Content-Type'] = 'application/json; charset=utf-8'

    return response

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=8000)

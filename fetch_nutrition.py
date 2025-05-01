import pandas as pd
import requests
import time

# Load CSV
df = pd.read_csv(r"F:\PRATIK\software projects\CookShook\dataset\cuisines.csv")

# Spoonacular API Key (Replace with your own)
API_KEY = " b565b011ba2f4b4eb3d2db1aa1bb2865"

# Base URL for the API
BASE_URL = "https://api.spoonacular.com/recipes/parseIngredients"

# Function to get nutrition data
def get_nutrition(ingredients):
    params = {
        "ingredientList": ingredients,
        "servings": 1,
        "apiKey": API_KEY
    }
    response = requests.post(BASE_URL, params=params)
    
    if response.status_code == 200:
        data = response.json()
        if data:
            calories = sum(item["nutrition"]["nutrients"][0]["amount"] for item in data if "nutrition" in item)
            protein = sum(item["nutrition"]["nutrients"][1]["amount"] for item in data if "nutrition" in item)
            carbs = sum(item["nutrition"]["nutrients"][2]["amount"] for item in data if "nutrition" in item)
            fat = sum(item["nutrition"]["nutrients"][3]["amount"] for item in data if "nutrition" in item)
            return calories, protein, carbs, fat
    return None, None, None, None

# Add new columns
df["calories"] = None
df["protein"] = None
df["carbs"] = None
df["fat"] = None

# Process each recipe
for index, row in df.iterrows():
    ingredients = row["ingredients"]
    if pd.notna(ingredients):  # Check if ingredients exist
        cal, prot, carb, fat = get_nutrition(ingredients)
        df.at[index, "calories"] = cal
        df.at[index, "protein"] = prot
        df.at[index, "carbs"] = carb
        df.at[index, "fat"] = fat
        time.sleep(1)  # To avoid API rate limits

# Save updated CSV
df.to_csv("cuisines_with_nutrition.csv", index=False)
print("✅ Nutrition data added successfully!")

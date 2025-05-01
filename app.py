from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import os
from flask_migrate import Migrate
from datetime import datetime
from flask_mail import Mail, Message
import random

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'  
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'  
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
migrate = Migrate(app, db)

# User model for authentication
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), nullable=False, unique=True)
    email = db.Column(db.String(150), nullable=False, unique=True)
    password = db.Column(db.String(150), nullable=False)

# Model to store favorite recipes
class Favorite(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    recipe_id = db.Column(db.Integer, nullable=False)  # Recipe ID from the dataset
    user = db.relationship('User', backref='favorites', lazy=True)

class SearchHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    search_query = db.Column(db.String(255), nullable=False)  # The search term (ingredients)
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())  # Time of search
    user = db.relationship('User', backref='search_history', lazy=True)

class SharedRecipe(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    name = db.Column(db.String(255), nullable=False)
    ingredients = db.Column(db.Text, nullable=False)
    instructions = db.Column(db.Text, nullable=False)
    image_url = db.Column(db.String(255), nullable=True)  # Optional image URL
    timestamp = db.Column(db.DateTime, default=db.func.current_timestamp())

    user = db.relationship('User', backref='shared_recipes', lazy=True)

# Create the new table in the database
with app.app_context():
    db.create_all()

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Loading existing recipe data and TF-IDF model
base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(base_dir, "dataset", "Cleaned_Indian_Food_Dataset.csv") 
df = pd.read_csv(csv_path)


df = df.rename(columns={
    'TranslatedRecipeName': 'name',
    'Cleaned-Ingredients': 'ingredients',
    'TranslatedInstructions': 'instructions',
    'image-url': 'image_url',
    'URL': 'video_url',  # ✅ new!
    'Cuisine': 'cuisine',
    'TotalTimeInMins': 'total_time'
})

# ✅ Reset index to add an ID
df.reset_index(drop=False, inplace=True)
df.rename(columns={'index': 'id'}, inplace=True)

vectorizer = TfidfVectorizer(stop_words='english', max_features=2000)
tfidf_matrix = vectorizer.fit_transform(df['ingredients'])

def generate_nutrition(ingredients_text):
    """Simulate nutrition values for now."""
    calories = random.randint(200, 600)
    protein = round(random.uniform(5, 20), 1)
    carbs = round(random.uniform(20, 50), 1)
    fat = round(random.uniform(5, 25), 1)
    return calories, protein, carbs, fat

# Add nutrition fields
df['calories'], df['protein'], df['carbs'], df['fat'] = zip(*df['ingredients'].apply(generate_nutrition))

@app.route('/dashboard')
@login_required
def dashboard():
    recent_searches = SearchHistory.query.filter_by(user_id=current_user.id).order_by(SearchHistory.timestamp.desc()).limit(3).all()

    # Fetch Favorites
    favorite_ids = [fav.recipe_id for fav in current_user.favorites]
    df['id'] = df['id'].astype(int)
    favorite_recipes = df[df['id'].isin(favorite_ids)].to_dict(orient='records')

    # Fetch Shared Favorites
    shared_favorites = SharedRecipe.query.filter(SharedRecipe.id.in_(favorite_ids)).all()
    shared_fav_data = [{
        'id': recipe.id,
        'name': recipe.name,
    } for recipe in shared_favorites]

    favorites = favorite_recipes + shared_fav_data
    favorites = favorites[:3]  # Only top 3

    # Meal Plan
    meal_plans = MealPlan.query.filter_by(user_id=current_user.id).order_by(MealPlan.date.asc()).limit(3).all()
    meals = []
    for meal in meal_plans:
        meals.append({
            'meal_type': meal.meal_type,
            'date': meal.date.strftime("%Y-%m-%d"),
            'recipe_id': meal.recipe_id
        })

    return render_template('dashboard.html', recent_searches=recent_searches, favorites=favorites, upcoming_meals=meals)


# Function to recommend recipes based on input ingredients
def recommend_recipes_by_ingredients(input_ingredients, tfidf_matrix, df, vectorizer):
    input_combined = ' '.join(input_ingredients)
    input_vector = vectorizer.transform([input_combined])
    sim_scores = cosine_similarity(input_vector, tfidf_matrix).flatten()
    
    # Get top 5 matches from dataset
    top_indices = sim_scores.argsort()[-5:][::-1]
    dataset_recipes = df.iloc[top_indices][['id', 'name', 'ingredients', 'instructions', 'image_url']].to_dict(orient='records')

    # Get similar recipes from user-shared recipes
    shared_recipes = SharedRecipe.query.all()  # Fetch all shared recipes from DB
    matched_shared_recipes = []
    for recipe in shared_recipes:
        recipe_ingredients = recipe.ingredients.lower().split(', ')
        if any(ingredient in recipe_ingredients for ingredient in input_ingredients):
            matched_shared_recipes.append({
                'id': recipe.id,
                'name': recipe.name,
                'ingredients': recipe.ingredients,
                'instructions': recipe.instructions,
                'image_url': recipe.image_url
            })

    # Combine both dataset & user-shared recipes
    return dataset_recipes + matched_shared_recipes



@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    if request.method == 'POST':
        user_ingredients = request.form['ingredients'].split(', ')
        
        # Log search only if user is authenticated
        if current_user.is_authenticated:
            search_query = ', '.join(user_ingredients)
            new_search = SearchHistory(user_id=current_user.id, search_query=search_query)
            db.session.add(new_search)
            db.session.commit()

        # Get recipe recommendations
        recommendations = recommend_recipes_by_ingredients(user_ingredients, tfidf_matrix, df, vectorizer)

        if not recommendations:
            flash('No recipes found for the provided ingredients', 'warning')
            return render_template('recommendations.html', recipes=[], user_ingredients=user_ingredients)

        return render_template('recommendations.html', recipes=recommendations, user_ingredients=user_ingredients)

    # For GET request
    return render_template('recommendations.html', recipes=[])

# About page route
@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/')
def home():
    if current_user.is_authenticated:
        # If logged in, show personalized home page
        personalized_recipes = []
        search_history = SearchHistory.query.filter_by(user_id=current_user.id).order_by(SearchHistory.timestamp.desc()).all()
        if search_history:
            last_search = search_history[0].search_query.split(', ')
            personalized_recipes = recommend_recipes_by_ingredients(last_search, tfidf_matrix, df, vectorizer)
        else:
            personalized_recipes = []

        favorite_ids = [fav.recipe_id for fav in current_user.favorites]

        return render_template('index.html', personalized_recipes=personalized_recipes, favorite_ids=favorite_ids)
    
    else:
        # Not logged in? Show landing page!
        return render_template('landing.html')


@app.route('/recipe/<int:recipe_id>')
def recipe_detail(recipe_id):
    # Check if the recipe exists in the shared database
    shared_recipe = SharedRecipe.query.filter_by(id=recipe_id).first()
    
    if shared_recipe:
        recipe = {
            'id': shared_recipe.id,
            'name': shared_recipe.name,
            'ingredients': shared_recipe.ingredients,
            'instructions': shared_recipe.instructions,
            'image_url': shared_recipe.image_url,
            'video_url': None,
            'calories': None,
            'protein': None,
            'carbs': None,
            'fat': None,
            'cuisine': None,
            'total_time': None
        }
    else:
        recipe_row = df[df['id'] == recipe_id]
        if not recipe_row.empty:
            recipe = recipe_row.iloc[0].to_dict()
        else:
            flash("Recipe not found!", "danger")
            return redirect(url_for('home'))

    return render_template('recipe_detail.html', recipe=recipe)


# Email validation function
def is_valid_email(email):
    email_regex = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
    return re.match(email_regex, email)


def is_valid_password(password):
    return re.fullmatch(r'[A-Za-z0-9]{1,8}', password) is not None

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        # ✅ Email validation
        if not is_valid_email(email):
            flash('Invalid email format. Please enter a valid email.', 'danger')
            return redirect(url_for('register'))

        # ✅ Password validation
        if not is_valid_password(password):
            flash('Password must be max 8 characters and contain only letters and numbers.', 'danger')
            return redirect(url_for('register'))

        # ✅ Check for existing user
        if User.query.filter_by(email=email).first():
            flash('Email address already exists. Please log in.', 'danger')
            return redirect(url_for('register'))

        # ✅ Create new user
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        new_user = User(username=username, email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('auth.html')



otp_storage = {}

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form['email']
        user = User.query.filter_by(email=email).first()

        if user:
            otp = random.randint(100000, 999999)
            otp_storage[email] = otp

            msg = Message("Your CookShook OTP", recipients=[email])
            msg.body = f"Your OTP to reset password is: {otp}"
            try:
                mail.send(msg)
                flash("OTP sent to your email. Please check your inbox.", "success")
                return redirect(url_for('reset_password', email=email))
            except Exception as e:
                flash(f"Error sending email: {str(e)}", "danger")
        else:
            flash("Email not found in our records.", "warning")

    return render_template('forgot_password.html')


@app.route('/reset_password/<email>', methods=['GET', 'POST'])
def reset_password(email):
    if request.method == 'POST':
        entered_otp = request.form['otp']
        new_password = request.form['password']

        # Basic validations
        if not re.match(r'^[a-zA-Z0-9]{1,8}$', new_password):
            flash("Password must be max 8 characters and contain only letters and numbers.", "danger")
            return redirect(url_for('reset_password', email=email))

        if str(otp_storage.get(email)) == entered_otp:
            user = User.query.filter_by(email=email).first()
            user.password = generate_password_hash(new_password, method='pbkdf2:sha256')
            db.session.commit()
            otp_storage.pop(email, None)  # Remove OTP after use

            flash("Password updated successfully! Please log in.", "success")
            return redirect(url_for('login'))
        else:
            flash("Invalid OTP. Please try again.", "danger")

    return render_template('reset_password.html', email=email)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        remember = 'remember' in request.form

        user = User.query.filter_by(email=email).first()
        
        if user and check_password_hash(user.password, password):
            login_user(user, remember=remember)
            flash('Logged in successfully!', 'success')
            return redirect(url_for('home'))
           
        else:
            flash('Invalid email or password.', 'danger')
            return redirect(url_for('login'))

    return render_template('auth.html')

# Route to save a recipe as a favorite
@app.route('/save_favorite/<int:recipe_id>', methods=['POST'])
@login_required  
def save_favorite(recipe_id):
    try:
        # Check if recipe exists in dataset
        recipe_exists = df[df['id'] == recipe_id].any().any()
        shared_recipe_exists = SharedRecipe.query.get(recipe_id)

        if not recipe_exists and not shared_recipe_exists:
            flash("Recipe not found.", "danger")
            return redirect(url_for('home'))

        # Check if already saved
        favorite = Favorite.query.filter_by(user_id=current_user.id, recipe_id=recipe_id).first()
        if favorite:
            flash("Recipe is already in favorites.", "info")
        else:
            # Save to favorites
            new_favorite = Favorite(user_id=current_user.id, recipe_id=recipe_id)
            db.session.add(new_favorite)
            db.session.commit()
            flash("Recipe added to favorites!", "success")

        return redirect(url_for('recipe_detail', recipe_id=recipe_id))

    except Exception as e:
        db.session.rollback()
        flash(f"Error saving favorite: {str(e)}", "danger")
        return redirect(url_for('home'))

# Route to display user's favorite recipes
@app.route('/favorites')
@login_required  # Ensure user is logged in
def view_favorites():
    try:
        # Fetch user's saved favorite recipe IDs
        favorite_ids = [fav.recipe_id for fav in current_user.favorites]

        if not favorite_ids:
            flash("You haven't added any favorite recipes yet.", "info")
            return render_template('favorites.html', recipes=[])

        # Ensure dataset `df` is loaded
        if 'df' not in globals():
            flash("Recipe dataset is not available!", "danger")
            return render_template('favorites.html', recipes=[])

        # Convert all IDs to int
        df['id'] = df['id'].astype(int)
        favorite_ids = list(map(int, favorite_ids))  

        # Fetch recipes from dataset
        dataset_recipes = df[df['id'].isin(favorite_ids)].to_dict(orient='records')

        # Fetch shared recipes from the database
        shared_recipes = SharedRecipe.query.filter(SharedRecipe.id.in_(favorite_ids)).all()

        # Convert shared recipes to dictionary format
        shared_recipes_data = [{
            'id': recipe.id,
            'name': recipe.name,
            'ingredients': recipe.ingredients,
            'instructions': recipe.instructions,
            'image_url': recipe.image_url
        } for recipe in shared_recipes]

        # Merge dataset & shared recipes
        all_favorite_recipes = dataset_recipes + shared_recipes_data

        if not all_favorite_recipes:
            flash("No matching favorite recipes found!", "info")
            return render_template('favorites.html', recipes=[])

        return render_template('favorites.html', recipes=all_favorite_recipes)

    except Exception as e:
        flash(f"Error loading favorites: {str(e)}", "danger")
        return render_template('favorites.html', recipes=[])

@app.route('/toggle_favorite', methods=['POST'])
@login_required
def toggle_favorite():
    recipe_id = int(request.json.get('recipe_id'))

    favorite = Favorite.query.filter_by(user_id=current_user.id, recipe_id=recipe_id).first()
    if favorite:
        db.session.delete(favorite)
        db.session.commit()
        return {'status': 'removed'}
    else:
        new_fav = Favorite(user_id=current_user.id, recipe_id=recipe_id)
        db.session.add(new_fav)
        db.session.commit()
        return {'status': 'added'}


@app.route('/remove_favorite/<int:recipe_id>', methods=['POST'])
@login_required
def remove_favorite(recipe_id):
    try:
        fav = Favorite.query.filter_by(user_id=current_user.id, recipe_id=recipe_id).first()
        if fav:
            db.session.delete(fav)
            db.session.commit()
            flash("❌ Recipe removed from favorites.", "success")
        else:
            flash("Recipe was not in favorites.", "info")
    except Exception as e:
        db.session.rollback()
        flash(f"Error removing favorite: {str(e)}", "danger")

    return redirect(url_for('view_favorites'))

#Route to share recipe
@app.route('/share_recipe', methods=['GET', 'POST'])
@login_required
def share_recipe():
    if request.method == 'POST':
        name = request.form['name']
        ingredients = request.form['ingredients']
        instructions = request.form['instructions']
        image_url = request.form.get('image_url', '')

        # Save to database
        new_recipe = SharedRecipe(
            user_id=current_user.id,
            name=name,
            ingredients=ingredients,
            instructions=instructions,
            image_url=image_url
        )
        db.session.add(new_recipe)
        db.session.commit()

        # Append the new recipe to the CSV file
        new_data = pd.DataFrame([{
            'id': len(df) + 1,  # Auto-generate ID
            'name': name,
            'ingredients': ingredients,
            'instructions': instructions,
            'image_url': image_url
        }])
        new_data.to_csv(csv_path, mode='a', header=False, index=False)  # Append to CSV

        flash('Your recipe has been shared successfully!', 'success')
        return redirect(url_for('view_shared_recipes'))

    return render_template('share_recipe.html')

#Route to view shared recipes

@app.route('/shared_recipes')
def view_shared_recipes():
    # Fetch shared recipes from the database
    shared_recipes = SharedRecipe.query.order_by(SharedRecipe.timestamp.desc()).all()

    # Fetch pre-existing recipes from the dataset
    dataset_recipes = df[['id', 'name', 'ingredients', 'instructions', 'image_url']].to_dict(orient='records')

    # Combine both lists
    all_recipes = shared_recipes + dataset_recipes  # Merging database and dataset recipes

    return render_template('shared_recipes.html', shared_recipes=all_recipes)

class MealPlan(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    recipe_id = db.Column(db.Integer, nullable=False)
    date = db.Column(db.Date, nullable=False) 
    meal_type = db.Column(db.String(50), nullable=False)  # Breakfast, Lunch, Dinner, Snack

    user = db.relationship('User', backref='meal_plans', lazy=True)

@app.route('/add_meal_plan', methods=['POST'])
@login_required
def add_meal_plan():
    recipe_id = request.form.get('recipe_id')
    meal_type = request.form.get('meal_type')
    date = request.form.get('date')

    if not recipe_id or not meal_type or not date:
        flash("All fields are required!", "danger")
        return redirect(url_for('view_meal_plan'))

    new_meal = MealPlan(
        user_id=current_user.id, 
        recipe_id=recipe_id, 
        meal_type=meal_type, 
        date=datetime.strptime(date, "%Y-%m-%d")
    )
    db.session.add(new_meal)
    db.session.commit()

    # ✅ Get logged-in user's email
    user_email = current_user.email  

    # ✅ Send Email Notification
    msg = Message(
        "Meal Plan Added",
        sender=app.config['MAIL_USERNAME'],
        recipients=[user_email]  # ✅ Uses user's email
    )
    msg.body = f"Your meal plan has been updated.\nMeal: {meal_type}\nDate: {date}\nRecipe ID: {recipe_id}"
    
    try:
        mail.send(msg)
        flash("Meal added to your planner! An email has been sent.", "success")
    except Exception as e:
        flash(f"Meal added, but email failed: {str(e)}", "warning")

    return redirect(url_for('view_meal_plan'))

from datetime import date  # Add this if not already

@app.route('/meal_plan')
@login_required
def view_meal_plan():
    today = date.today()

    # Fetch meal plans that are today or in the future
    meal_plans = MealPlan.query.filter(
        MealPlan.user_id == current_user.id,
        MealPlan.date >= today
    ).order_by(MealPlan.date.asc()).all()  # sort by nearest upcoming date

    planned_meals = []
    for meal in meal_plans:
        recipe_data = df[df['id'] == meal.recipe_id].to_dict(orient='records')
        
        if recipe_data:
            recipe = recipe_data[0]
            planned_meals.append({
                'id': meal.id,  # meal plan ID
                'recipe_id': meal.recipe_id,  # recipe ID
                'date': meal.date.strftime("%Y-%m-%d"),
                'meal_type': meal.meal_type,
                'recipe_name': recipe.get('name', 'Unknown Recipe'),
                'image_url': recipe.get('image_url', 'default.jpg')
            })

    # For dropdown while adding new meal
    recipes = df[['id', 'name']].drop_duplicates().to_dict(orient='records')

    return render_template('meal_plan.html', meals=planned_meals, recipes=recipes)


# Flask-Mail Configuration
app.config['MAIL_SERVER'] = 'smtp.gmail.com'
app.config['MAIL_PORT'] = 587
app.config['MAIL_USE_TLS'] = True
app.config['MAIL_USERNAME'] = 'your_email@gmail.com'  # Replace with your email
app.config['MAIL_PASSWORD'] = 'your_app_password'  # Use App Password, not personal password
app.config['MAIL_DEFAULT_SENDER'] = 'your_email@gmail.com'  # Ensure sender is set

mail = Mail(app)

# Route to logout user
@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'success')
    return redirect(url_for('home'))



if __name__ == '__main__':
    app.run(debug=True)

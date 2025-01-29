from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
import os
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key'  
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'  # SQLite database for user authentication
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = 'login'
#hello
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

# Create the new table in the database
with app.app_context():
    db.create_all()

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Loading existing recipe data and TF-IDF model
base_dir=os.path.dirname(os.path.abspath(__file__))
csv_path=os.path.join(base_dir,"dataset","cuisines.csv")
df=pd.read_csv(csv_path)

# Resetting the index to create an ID column
df.reset_index(drop=False, inplace=True)  # Keep the original index and create a new one
df.rename(columns={'index': 'id'}, inplace=True)  # Rename the index column to 'id'

# Now 'id' will be your unique identifier for each recipe.
vectorizer = TfidfVectorizer(stop_words='english', max_features=2000)
tfidf_matrix = vectorizer.fit_transform(df['ingredients'])

# Function to recommend recipes based on input ingredients
def recommend_recipes_by_ingredients(input_ingredients, tfidf_matrix, df, vectorizer):
    input_combined = ' '.join(input_ingredients)
    input_vector = vectorizer.transform([input_combined])
    sim_scores = cosine_similarity(input_vector, tfidf_matrix).flatten()
    top_indices = sim_scores.argsort()[-5:][::-1]
    # Ensure to fetch 'name', 'ingredients', and 'instructions'
    recommended_recipes = df.iloc[top_indices][['id', 'name', 'ingredients', 'instructions', 'image_url']]  
    return recommended_recipes


@app.route('/recommend', methods=['GET', 'POST'])
@login_required  # Ensure the user is logged in
def recommend():
    if request.method == 'POST':
        # Capture user input ingredients
        user_ingredients = request.form['ingredients'].split(', ')
        
        # Log the search to SearchHistory
        search_query = ', '.join(user_ingredients)
        new_search = SearchHistory(user_id=current_user.id, search_query=search_query)
        db.session.add(new_search)
        db.session.commit()

        # Perform the recommendation based on input ingredients
        recommendations = recommend_recipes_by_ingredients(user_ingredients, tfidf_matrix, df, vectorizer)
        
        # Check if any recommendations are found
        if recommendations.empty:
            flash('No recipes found for the provided ingredients', 'warning')
            return render_template('recommendations.html', recipes=[], user_ingredients=user_ingredients)

        # Convert results to a dictionary
        recipes = recommendations.to_dict(orient='records')

        return render_template('recommendations.html', recipes=recipes, user_ingredients=user_ingredients)

    # If GET request, render the form or previous recommendations
    return render_template('recommendations.html', recipes=[])


# About page route
@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/')
@login_required
def home():
    # Fetch personalized recipes based on the user's search history
    search_history = SearchHistory.query.filter_by(user_id=current_user.id).order_by(SearchHistory.timestamp.desc()).all()

    if search_history:
        # Get the most recent search query
        last_search = search_history[0].search_query.split(', ')  # Take the latest search history entry
        personalized_recommendations = recommend_recipes_by_ingredients(last_search, tfidf_matrix, df, vectorizer)
        personalized_recipes = personalized_recommendations.to_dict(orient='records')
    else:
        personalized_recipes = []  # Default to an empty list if no search history

    return render_template('index.html', personalized_recipes=personalized_recipes)

@app.route('/recipe/<int:recipe_id>')
def recipe_detail(recipe_id):
    # Fetch the recipe based on the index (make sure to access the correct row)
    recipe = df.loc[df['id'] == recipe_id].iloc[0]  # Adjusted to match your DataFrame structure
    return render_template('recipe_detail.html', recipe=recipe)


# Email validation function
def is_valid_email(email):
    email_regex = r'^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$'
    return re.match(email_regex, email)

# Route to register a new user
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        # Server-side email format validation
        if not is_valid_email(email):
            flash('Invalid email format. Please enter a valid email.', 'danger')
            return redirect(url_for('register'))

        # Check if the user already exists
        user = User.query.filter_by(email=email).first()
        if user:
            flash('Email address already exists. Please log in.', 'danger')
            return redirect(url_for('register'))

        # Create a new user with hashed password
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')

        new_user = User(username=username, email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()

        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('auth.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        user = User.query.filter_by(email=email).first()
        if user and check_password_hash(user.password, password):
            login_user(user)
            flash('Logged in successfully!', 'success')
            return redirect(url_for('home'))
           
        else:
            flash('Invalid email or password.', 'danger')
            return redirect(url_for('login'))

    return render_template('auth.html')

# Route to save a recipe as a favorite
@app.route('/save_favorite/<int:recipe_id>', methods=['POST'])
@login_required  # User must be logged in to save favorites
def save_favorite(recipe_id):
    # Check if the recipe is already in the user's favorites
    favorite = Favorite.query.filter_by(user_id=current_user.id, recipe_id=recipe_id).first()
    if not favorite:
        # Add the recipe to the user's favorites
        new_favorite = Favorite(user_id=current_user.id, recipe_id=recipe_id)
        db.session.add(new_favorite)
        db.session.commit()
        flash('Recipe added to your favorites!', 'success')
    else:
        flash('This recipe is already in your favorites.', 'info')
    
    return redirect(url_for('recipe_detail', recipe_id=recipe_id))

# Route to display user's favorite recipes
@app.route('/favorites')
@login_required  # User must be logged in to view favorites
def view_favorites():
    # Get the IDs of the user's favorite recipes
    favorite_ids = [fav.recipe_id for fav in current_user.favorites]
    
    # Query the recipes from the dataset based on the favorite IDs
    favorite_recipes = df[df['id'].isin(favorite_ids)]
    
    return render_template('favorites.html', recipes=favorite_recipes.to_dict(orient='records'))

# Route to logout user
@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'success')
    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)

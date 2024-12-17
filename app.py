from flask import Flask, request, render_template, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import UserMixin, login_user, LoginManager, login_required, logout_user, current_user
import os
from collections import Counter
import matplotlib.pyplot as plt
from werkzeug.security import generate_password_hash, check_password_hash
import pickle
from io import BytesIO
import base64
import matplotlib.pyplot as plt

# Setup Flask and SQLAlchemy
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret_key_here'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Load the trained Random Forest model and TF-IDF vectorizer
model_path = 'sentiment_ensemble_model_v4.pkl'
vectorizer_path = 'tfidf_vectorizer_ensemble_v4.pkl'

# Load model and vectorizer
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

with open(vectorizer_path, 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Setup Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# User model for the database
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    entries = db.relationship('JournalEntry', backref='user', lazy=True)

# JournalEntry model for storing user journal entries
class JournalEntry(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    sentiment = db.Column(db.String(50), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Route to handle the home page
@app.route('/')
def home():
    return render_template('home-page.html')  # Load home page directly

# Route to register a new user
@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        
        if User.query.filter_by(username=username).first() or User.query.filter_by(email=email).first():
            flash('Username or email already exists', 'danger')
            return redirect(url_for('register'))
        
        new_user = User(username=username, email=email, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        
        flash('Account created successfully! Please log in.', 'success')
        return redirect(url_for('login'))

    return render_template('register.html')

# Route to login
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password, password):
            login_user(user)
            flash('Successfully logged in!', 'success')
            return redirect(url_for('journal'))
        else:
            flash('Invalid credentials. Please try again.', 'danger')
    
    return render_template('login.html')

# Route to logout
@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out successfully.', 'info')
    return redirect(url_for('login'))

# User-specific dashboard
@app.route('/journal')
@login_required
def journal():
    user_entries = JournalEntry.query.filter_by(user_id=current_user.id).all()
    show_weekly_report = len(user_entries) >= 7
    return render_template('journal.html', entries=user_entries, show_weekly_report=show_weekly_report)

# Route to handle journal entry submission and sentiment prediction
@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if request.method == 'POST':
        journal_entry = request.form['journal_entry']
        
        if not journal_entry.strip():
            flash("Please enter a valid journal entry.", "danger")
            return redirect(url_for('journal'))

        # Transform the input text using the TF-IDF vectorizer
        transformed_entry = vectorizer.transform([journal_entry])
        
        # Predict the sentiment
        sentiment = model.predict(transformed_entry)[0]
        
        # Save the entry and sentiment in the database
        new_entry = JournalEntry(content=journal_entry, sentiment=sentiment, user_id=current_user.id)
        db.session.add(new_entry)
        db.session.commit()
        
        flash(f'Your entry has been saved with sentiment: {sentiment}', 'success')
        return redirect(url_for('journal'))

# Route for weekly report
@app.route('/weekly_report')
@login_required
def weekly_report():
    user_entries = JournalEntry.query.filter_by(user_id=current_user.id).all()

    if len(user_entries) < 7:
        flash("Not enough data for a weekly report. Please add more entries.", "warning")
        return redirect(url_for('journal'))

    # Extract sentiments from the last 7 entries
    sentiments = [entry.sentiment for entry in user_entries[-7:]]
    sentiment_counts = Counter(sentiments)

    # Calculate sentiment percentages
    total = sum(sentiment_counts.values())
    happy_percentage = (sentiment_counts.get('happy', 0) / total) * 100
    sad_percentage = (sentiment_counts.get('sad', 0) / total) * 100

    # Create a bar chart for sentiment distribution
    labels = ['Happy', 'Sad']
    percentages = [happy_percentage, sad_percentage]
    plt.figure(figsize=(6, 4))
    plt.bar(labels, percentages, color=['#66b3ff', '#ff9999'])
    plt.ylabel('Percentage')
    plt.title('Sentiment Analysis')

    # Save the chart to a BytesIO object and encode it as base64
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return render_template('weekly_report.html',
                           total_entries=len(user_entries),
                           happy_percentage=round(happy_percentage, 2),
                           sad_percentage=round(sad_percentage, 2),
                           plot_url=plot_url)

# Other routes
@app.route('/consultation')
def consultation():
    return render_template('consultation.html')

@app.route('/emergency-helpline')
def emergency_helpline():
    return render_template('emergency-helpline.html')

@app.route('/about-us')
def about_us():
    return render_template('about-us.html')

@app.route('/music')
def music():
    return render_template('music.html')

@app.route('/age')
def age_selection():
    return render_template('age.html')



@app.route('/daily-routine')
def daily_routine():
    return render_template('daily_routine.html')

@app.route('/mental-states')
def mental_states():
    age = request.args.get('age')
    return render_template('mental-states.html', age=age)




# Run the application
if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # Create database tables if they don't exist
    app.run(debug=True)

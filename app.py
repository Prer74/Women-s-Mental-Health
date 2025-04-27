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
from flask_socketio import SocketIO, join_room, leave_room, send
from flask import Flask, render_template, request, session, redirect, url_for, jsonify
from flask_socketio import SocketIO, join_room, leave_room, emit
from flask_login import current_user
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
import pandas as pd
import numpy as np
import pdfkit
from flask import make_response

# config = pdfkit.configuration(wkhtmltopdf=r'C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe')  # Windows path


# Setup Flask and SQLAlchemy
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret.key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
socketio = SocketIO(app)  # Initialize Flask-SocketIO for real-time communication

# Load the trained Random Forest model and TF-IDF vectorizer
model_path = 'logistic_model.pkl'
vectorizer_path = 'tfidf_vectorizer.pkl'
decision_tree_model_path = 'decision_tree_model.pkl'

# Load model and vectorizer
with open(model_path, 'rb') as model_file:
    model = pickle.load(model_file)

with open(vectorizer_path, 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# Load trained model and encoders
with open("decision_tree_model.pkl", "rb") as f:
    model1 = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("label_encoders.pkl", "rb") as f:
    label_encoders = pickle.load(f)

with open("y_encoder.pkl", "rb") as f:
    y_encoder = pickle.load(f)


# Setup Flask-Login
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# User model for the database
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    birthdate = db.Column(db.Date, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    entries = db.relationship('JournalEntry', backref='user', lazy=True)


# JournalEntry model for storing user journal entries
class JournalEntry(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    content = db.Column(db.Text, nullable=False)
    sentiment = db.Column(db.String(50), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    date = db.Column(db.DateTime, default=datetime.utcnow)

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
        birthdate_str = request.form['birthdate']

        # Convert birthdate from string to Python date object
        try:
            birthdate = datetime.strptime(birthdate_str, '%Y-%m-%d').date()
        except ValueError:
            flash('Invalid birth date format. Please use YYYY-MM-DD.', 'danger')
            return redirect(url_for('register'))

        hashed_password = generate_password_hash(password, method='pbkdf2:sha256')
        
        if User.query.filter_by(username=username).first() or User.query.filter_by(email=email).first():
            flash('Username or email already exists', 'danger')
            return redirect(url_for('register'))
        
        new_user = User(username=username, email=email, birthdate=birthdate, password=hashed_password)
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
            
            return redirect(url_for('home'))
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
    user_entries = (
        JournalEntry.query
        .filter_by(user_id=current_user.id)
        .order_by(JournalEntry.date.desc())
        .limit(5)
        .all()
    )
    # Decrypt each entry before displaying
    for entry in user_entries:
        try:
            entry.content = decrypt_entry(entry.content)  # Decrypt the stored text
        except Exception as e:
            entry.content = "[Decryption Error]"  # Handle decryption failure gracefully

    # Show the weekly report button every time
    show_weekly_report = True

    return render_template('journal.html', user_entries=user_entries, show_weekly_report=show_weekly_report)


@app.route('/weekly_report')
@login_required
def weekly_report():
    # Check if today is Sunday
    if datetime.today().weekday() != 5:  # 6 corresponds to Sunday in Python's weekday() method
        flash("Weekly report is available only on Sundays. Please come back then.", "info")
        return redirect(url_for('journal'))

    # Get the current week's start (Monday) and end (Sunday)
    today = datetime.today().date()
    start_of_week = today - timedelta(days=today.weekday())  # Monday
    end_of_week = start_of_week + timedelta(days=6)  # Sunday

    # Retrieve user's journal entries within this week
    user_entries = JournalEntry.query.filter(
        JournalEntry.user_id == current_user.id,
        JournalEntry.date >= start_of_week,
        JournalEntry.date <= end_of_week
    ).all()

    if not user_entries:
        flash("No journal entries found for this week.", "warning")
        return redirect(url_for('journal'))

    # Organize entries by day of the week
    daily_sentiments = {start_of_week + timedelta(days=i): [] for i in range(7)}
    
    for entry in user_entries:
        day = entry.date.date()
        if day in daily_sentiments:
            daily_sentiments[day].append(entry.sentiment)

    # Calculate daily sentiment percentages
    daily_sentiment_summary = {}
    for day, sentiments in daily_sentiments.items():
        sentiment_counts = Counter(sentiments)
        total = sum(sentiment_counts.values())

        positive_percentage = (sentiment_counts.get('Positive', 0) / total) * 100 if total > 0 else 0
        negative_percentage = (sentiment_counts.get('Negative', 0) / total) * 100 if total > 0 else 0

        daily_sentiment_summary[day.strftime('%A')] = {
            'Positive': round(positive_percentage, 2),
            'Negative': round(negative_percentage, 2)
        }

    # Calculate overall weekly sentiment analysis
    sentiments = [entry.sentiment for entry in user_entries]
    sentiment_counts = Counter(sentiments)
    total = sum(sentiment_counts.values())

    weekly_positive = (sentiment_counts.get('Positive', 0) / total) * 100 if total > 0 else 0
    weekly_negative = (sentiment_counts.get('Negative', 0) / total) * 100 if total > 0 else 0

    # Calculate user's age
    today = datetime.today().date()
    age = today.year - current_user.birthdate.year - ((today.month, today.day) < (current_user.birthdate.month, current_user.birthdate.day))

    # Create pie chart for sentiment distribution
    labels = ['Positive', 'Negative']
    percentages = [weekly_positive, weekly_negative]

    plt.figure(figsize=(6, 4))
    plt.pie(percentages, labels=labels, autopct='%1.1f%%', colors=['#66b3ff', '#ff9999'], startangle=140)
    plt.title('Sentiment Distribution for This Week')

    # Save the chart to a BytesIO object and encode as base64
    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return render_template('weekly_report.html',
                           total_entries=len(user_entries),
                           positive_percentage=round(weekly_positive, 2),
                           negative_percentage=round(weekly_negative, 2),
                           plot_url=plot_url,
                           username=current_user.username,
                           age=age,
                           daily_sentiment_summary=daily_sentiment_summary)


@app.route('/download_weekly_report_pdf')
@login_required
def download_weekly_report_pdf():
    if datetime.today().weekday() != 5:
        flash("Weekly report is available only on Sundays. Please come back then.", "info")
        return redirect(url_for('journal'))

    today = datetime.today().date()
    start_of_week = today - timedelta(days=today.weekday())
    end_of_week = start_of_week + timedelta(days=6)

    user_entries = JournalEntry.query.filter(
        JournalEntry.user_id == current_user.id,
        JournalEntry.date >= start_of_week,
        JournalEntry.date <= end_of_week
    ).all()

    if not user_entries:
        flash("No journal entries found for this week.", "warning")
        return redirect(url_for('journal'))

    daily_sentiments = {start_of_week + timedelta(days=i): [] for i in range(7)}
    for entry in user_entries:
        day = entry.date.date()
        if day in daily_sentiments:
            daily_sentiments[day].append(entry.sentiment)

    daily_sentiment_summary = {}
    for day, sentiments in daily_sentiments.items():
        sentiment_counts = Counter(sentiments)
        total = sum(sentiment_counts.values())
        positive_percentage = (sentiment_counts.get('Positive', 0) / total) * 100 if total > 0 else 0
        negative_percentage = (sentiment_counts.get('Negative', 0) / total) * 100 if total > 0 else 0
        daily_sentiment_summary[day.strftime('%A')] = {
            'Positive': round(positive_percentage, 2),
            'Negative': round(negative_percentage, 2)
        }

    sentiments = [entry.sentiment for entry in user_entries]
    sentiment_counts = Counter(sentiments)
    total = sum(sentiment_counts.values())
    weekly_positive = (sentiment_counts.get('Positive', 0) / total) * 100 if total > 0 else 0
    weekly_negative = (sentiment_counts.get('Negative', 0) / total) * 100 if total > 0 else 0

    today = datetime.today().date()
    age = today.year - current_user.birthdate.year - ((today.month, today.day) < (current_user.birthdate.month, current_user.birthdate.day))

    labels = ['Positive', 'Negative']
    percentages = [weekly_positive, weekly_negative]
    plt.figure(figsize=(6, 4))
    plt.pie(percentages, labels=labels, autopct='%1.1f%%', colors=['#66b3ff', '#ff9999'], startangle=140)
    plt.title('Sentiment Distribution for This Week')

    img = BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

   # 1. Render HTML
    rendered = render_template('weekly_report.html',
                           total_entries=len(user_entries),
                           positive_percentage=round(weekly_positive, 2),
                           negative_percentage=round(weekly_negative, 2),
                           plot_url=plot_url,
                           username=current_user.username,
                           age=age,
                           daily_sentiment_summary=daily_sentiment_summary)

# 2. Configure wkhtmltopdf path
    config = pdfkit.configuration(wkhtmltopdf=r"C:\Program Files\wkhtmltopdf\bin\wkhtmltopdf.exe")  # Adjust path if needed

# 3. Convert HTML to PDF
    pdf = pdfkit.from_string(rendered, False, configuration=config)

# 4. Send PDF as download
    response = make_response(pdf)
    response.headers['Content-Type'] = 'application/pdf'
    response.headers['Content-Disposition'] = 'attachment; filename=weekly_report.pdf'
    return response


@app.route('/community')
@login_required  # Ensure that the user is logged in before accessing this route
def community():
    # Fetch the logged-in user's username from current_user
    username = current_user.username  # This fetches the username of the logged-in user
    return render_template('chat.html', username=username)


messages = [] 
# Socket.IO events
@socketio.on('join')
def handle_join(data):
    username = current_user.username if current_user.is_authenticated else 'Anonymous'
    room = data['room']
    join_room(room)
    
    # Filter messages to only keep those from the last 5 minutes
    recent_messages = [msg for msg in messages if datetime.now() - msg['timestamp'] < timedelta(minutes=5)]
    
    # Send recent messages to the new user
    for msg in recent_messages:
        emit('message', {'username': msg['username'], 'message': msg['message']}, room=room)
    
    # Notify the room that the user has joined
    emit('message', {'username': 'System', 'message': f'{username} has joined the room.'}, room=room)

@socketio.on('message')
def handle_message(data):
    username = current_user.username if current_user.is_authenticated else 'Anonymous'
    room = data['room']
    message = data['message']
    
    # Store the message with the current timestamp
    messages.append({
        'username': username,
        'message': message,
        'timestamp': datetime.now()
    })
    
    # Emit the message to the room
    emit('message', {'username': username, 'message': message}, room=room)
@socketio.on('leave')
def handle_leave(data):
    username = data['username']
    room = data['room']
    leave_room(room)
    emit('message', {'username': 'System', 'message': f'{username} has left the room.'}, room=room)


# Other existing routes remain the same
@app.route('/consultation')
def consultation():
    return render_template('consultation.html')

@app.route('/emergency-helpline')
def emergency_helpline():
    return render_template('emergency-helpline.html')

@app.route('/about-us')
def about_us():
    return render_template('about-us.html')

@app.route('/articles')
def articles():
    return render_template('articles.html')

@app.route('/music')
def music():
    return render_template('music.html')
# Route to handle journal entry submission and sentiment prediction
 #Load the key for encryption and decryption
def load_key():
    return open("secret.key", "rb").read()

# Encrypt the journal entry before saving
def encrypt_entry(entry_text):
    key = load_key()
    fernet = Fernet(key)
    encrypted_entry = fernet.encrypt(entry_text.encode())
    return encrypted_entry

def decrypt_entry(encrypted_text):
    key = load_key()  # Load the same key used for encryption
    fernet = Fernet(key)
    
    try:
        return fernet.decrypt(encrypted_text).decode()  # No need to encode, it's already bytes
    except Exception as e:
        print("Decryption error:", e)
        return "[Decryption Failed]"


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
        
        # Encrypt the journal entry before saving it
        encrypted_entry = encrypt_entry(journal_entry)
        
        # Save the encrypted entry and sentiment in the database
        new_entry = JournalEntry(content=encrypted_entry, sentiment=sentiment, user_id=current_user.id)
        db.session.add(new_entry)
        db.session.commit()
        
        flash(f'Your predicted sentiment: {sentiment}', 'success')
        return redirect(url_for('journal'))
    
@app.route('/age', methods=['GET', 'POST'])
def age_selection():
    if request.method == 'POST':
        session['age'] = request.form['age']
        return redirect(url_for('daily-routine'))
    return render_template('age.html')

@app.route('/daily-routine', methods=['GET', 'POST'])
def daily_routine():
    if request.method == 'POST':
        session['sleep_hours'] = request.form['sleep_hours']
        session['work_hours'] = request.form['work_hours']
        session['activity'] = request.form['activity']
        return redirect(url_for('mental-states'))
    return render_template('daily-routine.html')



@app.route('/mental_states', methods=['POST'])
def mental_states():
    sleep_hours = request.form.get('sleep_hours')
    sleep_quality = request.form.get('sleep_quality')
    work_hours = request.form.get('work_hours')
    activity = request.form.get('activity')

    # Pass data to the mental_states.html page
    return render_template('mental-states.html', sleep_hours=sleep_hours, sleep_quality=sleep_quality, 
                           work_hours=work_hours, activity=activity)



@app.route('/store_mental_state', methods=['POST'])
def store_mental_state():
    data = request.json
    session['selected_mental_state'] = data['mental_state']
    print("Stored Mental State:", session.get('selected_mental_state'))  # Debugging
    return jsonify({"message": "Mental state stored successfully!"})


@app.route("/recommendations", methods=["POST", "GET"])
def recommendations():
    try:
        if request.method == "POST":
            data = request.json
            print("Received Data:", data)

            input_features = pd.DataFrame([{
    "Age Group": int(data.get("Age Group", 0)),
    "Sleep Hours": int(data.get("Sleep Hours", 0)),
    "Work Hours": int(data.get("Work Hours", 0)),
    "Physical Activities": int(data.get("Physical Activities", 0)),
    "Mental State": int(data.get("Mental State", 0)),
    "Severity": int(data.get("Severity", 0))
}])


            if "scaler" not in globals() or "model" not in globals() or "y_encoder" not in globals():
                return jsonify({"error": "Model components not loaded"}), 500

            input_scaled = scaler.transform(input_features)
            encoded_recommendation = model1.predict(input_scaled)[0]
            original_recommendation = y_encoder.inverse_transform([encoded_recommendation])[0]

            # Ensure it's a list (some models return single strings)
            if isinstance(original_recommendation, str):
                recommendations_list = [original_recommendation]
            else:
                recommendations_list = list(original_recommendation)

            # Store in session for retrieval in GET request
            session['recommendations'] = recommendations_list
            print("Predicted Recommendation:", recommendations_list)

            return jsonify({"message": "Recommendations saved successfully!"})

        elif request.method == "GET":
            # Retrieve stored recommendations from session
            recommendations_list = session.get('recommendations', [])

            return render_template("recommendations.html", recommendations=recommendations_list)

    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": str(e)}), 500


@app.route('/questionnaire/<topic>')
def questionnaire(topic):
    try:
        return render_template(f'questionnaire/{topic}.html')
    except:
        return "The requested questionnaire does not exist.", 404


#new code start here

def get_user_severity(user_answers, dataset_path):
    """
    Matches user answers with the dataset to determine severity.
    """
    # Load the dataset
    df = pd.read_csv(dataset_path)

    # Convert user_answers to a DataFrame
    user_df = pd.DataFrame([user_answers], columns=[f"Q{i+1}" for i in range(len(user_answers))])

    # Match with dataset
    matching_row = df.merge(user_df, on=[f"Q{i+1}" for i in range(len(user_answers))], how="inner")

    if not matching_row.empty:
        return matching_row['Category'].iloc[0]
    else:
        return "No match found in the dataset."



@app.route('/<topic>', methods=['GET', 'POST'])
def handle_questionnaire(topic):
    """
    Handles questionnaire submission for various topics.
    """
    dataset_mapping = {
        "sexual_and_reproductive_health": "dataset/sex_reproductive_dataset.csv",
        "relationship_and_family": "dataset/relationship_family_dataset.csv",
        "Anxiety": "dataset/anxiety_dataset.csv",
        "body_image": "dataset/body_image_dataset.csv",
        "depression_issues": "dataset/depression_dataset.csv",
        "eating_disorder": "dataset/eating_disorder_dataset.csv",
        "family_dynamics": "dataset/family_dynamics_dataset.csv",
        "hormonal_changes": "dataset/harmonal_dataset.csv",
        "suicidal_thoughts": "dataset/suicidal_thoughts_dataset.csv",
    }

    if topic not in dataset_mapping:
        return "Invalid topic.", 404

    dataset_path = dataset_mapping[topic]

    if request.method == 'POST':
        # Dynamically determine the number of questions based on dataset columns
        user_answers = [request.form.get(f'Q{i+1}') for i in range(len(pd.read_csv(dataset_path).columns) - 1)]
        severity = get_user_severity(user_answers, dataset_path)
        return render_template(f'{topic}_result.html', severity=severity)

    return render_template(f'{topic}.html')


# Run the application
if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # Create database tables if they don't exist
    socketio.run(app, debug=True)
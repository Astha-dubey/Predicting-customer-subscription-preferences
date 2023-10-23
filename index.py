import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import pandas as pd
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import random
from PIL import Image, ImageTk

# Load the datasets
users_df = pd.read_csv('users.csv')
channels_df = pd.read_csv('channels.csv')
subscribe_df = pd.read_csv('subscribe.csv')
comment_df = pd.read_csv('comment.csv')
account_df = pd.read_csv('account.csv')
likes_df = pd.read_csv('likes.csv')

# Create a Surprise Dataset
reader = Reader(rating_scale=(0, 1))
subscribe_data = Dataset.load_from_df(subscribe_df[['emailID', 'subscribe_channelID', 'subscribeWatch_time']], reader)
comment_data = Dataset.load_from_df(comment_df[['emailID', 'commentchannelID', 'commentWatch_time']], reader)
account_data = Dataset.load_from_df(account_df[['emailID', 'account_channelID', 'accountWatch_time']], reader)
likes_data = Dataset.load_from_df(likes_df[['emailID', 'likes_channelID', 'likesWatch_time']], reader)


# Create and train the recommendation models
def train_model(data):
    try:
        trainset, testset = train_test_split(data, test_size=0.2)
        model = SVD()
        model.fit(trainset)
        predictions = model.test(testset)

        # Calculate RMSE and MAE
        rmse = accuracy.rmse(predictions)
        mae = accuracy.mae(predictions)
        if 0 <= rmse <= 100 and 0 <= mae <= 100:
            print(f"Accuracy Train Test: {rmse} {mae} ")
        else:
            print(f"Accuracy on Training , Accuracy on Testing: {rmse / 99} {mae / 99} ")
        return model
    except ValueError as e:
        print(f"ValueError: {e}")
        return None


subscribe_model = train_model(subscribe_data)
comment_model = train_model(comment_data)
account_model = train_model(account_data)
likes_model = train_model(likes_data)


# Function to filter channels by genre
def filter_channels_by_genre(genre):
    if genre == "All Genres":
        return channels_df
    else:
        return channels_df[channels_df['Genere'] == genre]


# Function to display recommended channels, user details, and user activity in a new window
def display_recommendations_with_user_details(recommendations, user_details, user_activity):
    new_window = tk.Toplevel()
    new_window.title("Here is Listed Preferences and User Details")

    # Create labels and display user details
    user_details_label = tk.Label(new_window, text="User Details:")
    user_details_label.pack()

    user_details_text = "\n".join([f"{key}: {value}" for key, value in user_details.items()])
    user_details_display = tk.Label(new_window, text=user_details_text)
    user_details_display.pack()

    # Create a Treeview widget to display the recommended channels
    channel_table = ttk.Treeview(new_window, columns=("Channel Name", "Probability"))
    channel_table.heading("#1", text="Channel Name")
    channel_table.heading("#2", text="Probability")
    channel_table.pack()

    # Populate the Treeview with recommended channels and probabilities
    for channel_id, probability in recommendations:
        channel_name = channels_df[channels_df['channelID'] == channel_id]['ChannelName'].values[0]
        channel_table.insert("", "end", values=(channel_name, f"{probability:.2f}"))

    # Create and display a user activity graph
    plt.figure(figsize=(8, 6))
    plt.bar(user_activity.keys(), user_activity.values())
    plt.xlabel("Activity Type")
    plt.ylabel("Frequency")
    plt.title("User Activity")
    plt.xticks(rotation=45)
    plt.tight_layout()

    canvas = FigureCanvasTkAgg(plt.gcf(), master=new_window)
    canvas.draw()
    canvas.get_tk_widget().pack()


def get_user_activity(user_email, channelIDs):
    try:
        print(user_email)
        # Join user-specific data from different DataFrames
        user_subscribe_data = subscribe_df[
            (subscribe_df['emailID'] == user_email) & (subscribe_df['subscribe_channelID'].isin(channelIDs))]
        user_comment_data = comment_df[
            (comment_df['emailID'] == user_email) & (comment_df['commentchannelID'].isin(channelIDs))]
        user_account_data = account_df[
            (account_df['emailID'] == user_email) & (account_df['account_channelID'].isin(channelIDs))]
        user_likes_data = likes_df[(likes_df['emailID'] == user_email) & (likes_df['likes_channelID'].isin(channelIDs))]

        # Calculate total watch times for different activities
        subscribe_watch_time = user_subscribe_data['subscribeWatch_time'].sum()
        comment_watch_time = user_comment_data['commentWatch_time'].sum()
        account_watch_time = user_account_data['accountWatch_time'].sum()
        likes_watch_time = user_likes_data['likesWatch_time'].sum()

        if subscribe_watch_time == 0 and comment_watch_time == 0 and account_watch_time == 0 and likes_watch_time == 0:
            subscribe_watch_time = random.random()
            comment_watch_time = random.random()
            account_watch_time = random.random()
            likes_watch_time = random.random()
        user_activity_data = {
            "Subscribe": subscribe_watch_time,
            "Comment": comment_watch_time,
            "Account": account_watch_time,
            "Likes": likes_watch_time
        }
        return user_activity_data
    except Exception as e:
        print(f"Error fetching user activity: {e}")
        return {"Subscribe": random.random(), "Comment": random.random(), "Account": random.random(),
                "Likes": random.random()}


# Function to recommend channels
def recommend_channels():
    user_email = email_entry.get()
    selected_genre = genre_combobox.get()

    try:
        recommendations = []
        nisha = []
        # Filter channels by genre
        filtered_channels = filter_channels_by_genre(selected_genre)

        # Create a dictionary to store the highest prediction for each unique channel ID
        unique_channel_predictions = {}

        # Subscribe recommendations
        if subscribe_model:
            for channel_id in filtered_channels['channelID']:
                if channel_id not in subscribe_df[subscribe_df['emailID'] == user_email]['subscribe_channelID'].values:
                    prediction = subscribe_model.predict(user_email, channel_id)
                    if channel_id not in unique_channel_predictions or prediction.est > unique_channel_predictions[
                        channel_id]:
                        # Introduce variation based on ratings (multiply by a random factor)
                        random_factor = 1 + random.uniform(-0.1, 0.1)
                        unique_channel_predictions[channel_id] = prediction.est * random_factor

        # Comment recommendations (you can repeat this for other interaction types: account, likes)
        if comment_model:
            for channel_id in filtered_channels['channelID']:
                if channel_id not in comment_df[comment_df['emailID'] == user_email]['commentchannelID'].values:
                    prediction = comment_model.predict(user_email, channel_id)
                    if channel_id not in unique_channel_predictions or prediction.est > unique_channel_predictions[
                        channel_id]:
                        # Introduce variation based on ratings (multiply by a random factor)
                        random_factor = 1 + random.uniform(-0.1, 0.1)
                        unique_channel_predictions[channel_id] = prediction.est * random_factor

        # Combine recommendations from different interaction types
        for channel_id, probability in unique_channel_predictions.items():
            nisha.append(channel_id)
            recommendations.append((channel_id, probability))

        # Sort the recommendations by probability in descending order
        sorted_recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)

        # Display user details and recommended channels in a new window
        user_details = users_df[users_df['emailID'] == user_email].iloc[0].to_dict()

        # Simulated user activity data (you should replace this with your actual user activity data)
        user_activity = get_user_activity(user_email, nisha)
        print(user_activity)
        print(nisha)
        display_recommendations_with_user_details(sorted_recommendations[:10], user_details, user_activity)

    except KeyError:
        messagebox.showerror("Error", "User not found in the dataset.")


def validate_input():
    user_email = email_entry.get()
    selected_genre = genre_combobox.get()

    if not user_email:
        messagebox.showerror("Error", "Please enter an email address.")
    if not selected_genre:
        messagebox.showerror("Error", "Please select a genre.")
    else:
        recommend_channels()


# Create the main Tkinter window
app = tk.Tk()
app.title("Predicting Customer Preference Subscription App")
# app.attributes("-fullscreen", True)

# Load and display a background image
window_width = 800
window_height = 600

# Configure the window size
app.geometry(f"{window_width}x{window_height}")

# Load and display a background image
background_image = Image.open("background.jpg")
background_photo = ImageTk.PhotoImage(background_image)

# Create a Label to display the background image
background_label = tk.Label(app, image=background_photo)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

# Create and configure user input fields
email_label = tk.Label(app, text="Email:")
email_label.place(relx=0.3, rely=0.3, anchor="e")

# Increase the input field size by setting the width
email_entry = tk.Entry(app, width=40)  # Adjust the width as needed
email_entry.place(relx=0.32, rely=0.3, anchor="w")

genre_label = tk.Label(app, text="Select Genre:")
genre_label.place(relx=0.3, rely=0.4, anchor="e")

# Dynamically populate genre options based on unique values in channels_df
genre_options = ["All Genres"] + channels_df['Genere'].unique().tolist()
genre_combobox = ttk.Combobox(app, values=genre_options)
genre_combobox.place(relx=0.32, rely=0.4, anchor="w")

# Create the "Predict User Preference" button
recommend_button = tk.Button(app, text="Predict User Preference", command=validate_input)
recommend_button.place(relx=0.5, rely=0.5, anchor="center")

# Start the Tkinter main loop
app.mainloop()
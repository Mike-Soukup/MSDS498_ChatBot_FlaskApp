"""Main Flask App for ChatBot."""
import os
from flask import Flask, redirect, render_template, request, url_for

app = Flask(__name__)


@app.route("/")
def welcome():
    """Chatbot API Home Page."""
    return render_template("home.html")


@app.route("/upload", methods=["GET", "POST"])
def upload():
    """Upload image."""
    if request.method == "POST":
        print("Post")
    else:
        return render_template("upload.html")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)

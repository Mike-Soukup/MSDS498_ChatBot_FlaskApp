"""Main Flask App for ChatBot."""
import os
from flask import Flask, redirect, render_template, request, url_for

app = Flask(__name__)

@app.route("/")
def welcome():
    """Main ChatBot HTML"""
    return render_template("home.html")

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, debug = True)
"""Main Flask App for ChatBot."""
from fileinput import filename
from flask import Flask, render_template, request

app = Flask(__name__)


@app.route("/")
def welcome():
    """Chatbot API Home Page."""
    return render_template("home.html")


@app.route("/upload", methods=["GET", "POST"])
def upload():
    """Upload image."""
    return render_template("upload.html")


@app.route("/success", methods=["POST"])
def success():
    """Return ackowledgement of image submission."""
    if request.method == "POST":
        f = request.files["img"]
        f.save(f.filename)
        return render_template("acknowledgement.html", name=f.filename)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)

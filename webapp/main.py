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
        f1 = request.files["img1"]
        f2 = request.files["img2"]
        f1.save(f1.filename)
        f2.save(f2.filename)
        return render_template(
            "acknowledgement.html",
            name_1=f1.filename,
            name_2=f2.filename,
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)

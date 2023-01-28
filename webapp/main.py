"""Main Flask App for ChatBot."""
from fileinput import filename
import os
from flask import Flask, render_template, request, session
from werkzeug.utils import secure_filename

# Define upload folder path:
UPLOAD_FOLDER = os.path.join("static",'uploads')
# Define allowed files:
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def welcome():
    """Chatbot API Home Page."""
    return render_template("home.html")


@app.route("/upload", methods=["GET", "POST"])
def upload():
    """Upload image."""
    return render_template("upload.html")


# @app.route("/success", methods=["POST"])
# def success():
#     """Return ackowledgement of image submission."""
#     if request.method == "POST":
#         f1 = request.files["img1"]
#         f2 = request.files["img2"]
#         f1.save(f1.filename)
#         f2.save(f2.filename)
#         return render_template(
#             "acknowledgement.html",
#             name_1= f1.filename,
#             name_2=f2.filename,
#         )

@app.route("/output", methods=["POST"])
def output():
    """Return output of image submission."""
    if request.method == "POST":
        # Get uploaded files
        f1 = request.files["img1"]
        f2 = request.files["img2"]

        # Extract uploaded data files
        img1_filename = secure_filename(f1.filename)
        img2_filename = secure_filename(f2.filename)

        # Upload file:
        f1.save(os.path.join(app.config['UPLOAD_FOLDER'], img1_filename))
        f2.save(os.path.join(app.config['UPLOAD_FOLDER'], img2_filename))

        # Create file path:
        img1_path = os.path.join(app.config['UPLOAD_FOLDER'], img1_filename)
        img2_path = os.path.join(app.config['UPLOAD_FOLDER'], img2_filename)

        return render_template(
            "output.html",
            image_1=img1_path,
            image_2=img2_path,
        )


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)

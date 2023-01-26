# MSDS 498 Chat Bot Flask App Repo

### To Run:

1. Activate the virtual environment: `source .flaskapp/bin/activate`
2. Start the Flask App: `python3 main.py`
3. Look at App: Go to localhost:8080


### Functionality:

- This webapp will direct the user to upload an image file.
- Once the image file is submitted, an acknowledgement will be shared.
- File will be placed in the `webapp` directory

### To Do:
- Data quality testing to ensure only specific files are upload...
    - .png
    - .jpeg
        - Will raise errors if files like .txt, .docx, .xml, etc. are uploaded.
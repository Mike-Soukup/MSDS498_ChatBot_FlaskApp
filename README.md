# MSDS 498 Chat Bot Flask App Repo

### To Run:

1. Activate the virtual environment: `source .flaskapp/bin/activate`
2. Start the Flask App: `python3 app.py`
3. Look at App: Go to localhost:8080


### Functionality:

##### Demo Prediction:
- This webapp will direct the user to upload an image file.
- Once the image file is submitted, an acknowledgement will be shared.
- File will be placed in the `static/uploads` directory

##### Demo Text API:
- Either ping an API text output
- OR can user input text that will be sent back as a JSON.

##### Demo Image API:
- Submit and Image and see how API Endpoint will provide content back to user.

### To Do:
- Demonstrate the Azure Health Bot can hit our API endpoint and "talk" with it.
- Data quality testing to ensure only specific files are upload...
    - .png
    - .jpeg
        - Will raise errors if files like .txt, .docx, .xml, etc. are uploaded.

### 2022-01-28 Comments:
- For reference: You can report issues at https://github.com/Microsoft/Oryx/issues
- Having issues pushing code to azure for development updates
    - Maybe due to pushing from local git...
        - Not sure why but using GitHub would likely be a better approach
    - Can submit an issue and see what is going on...


### Test increasing http.postBuffer to push code to Azure
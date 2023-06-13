# MSDS 498 Chat Bot Flask App Repo

## Product Demonstration:

Link to [Product Overview and Demonstration.](https://www.youtube.com/watch?v=mLMHo4MMB8Y)

### To Run:

1. Activate the virtual environment: `source .flaskapp/bin/activate`
2. Start the Flask App: `python3 app.py`
3. Look at App: Go to localhost:5000


### Functionality:

#### ChatBot Functionality:
- Visit: https://msds498swam.azurewebsites.net
- Demo REST API Test:
    - Type `begin demo_rest_api` ask a question, and the API Endpoint hit will return your question!

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

### 2022-02-12 Comments:
- Still having deployment issues. Not a robust process
- Image recognition feature is working in dev environment.

### Fix Azure Deployment:
- Garbage Collect: `git gc --aggressive`
- Increase HTTP Post Buffer: `git config --global http.postBuffer 157286400`
- Remove remote azure and add back: `git remote remove azure` && `git remote add https://demorestapimsds498.scm.azurewebsites.net:443/demorestapimsds498.git`
- Added to Makefile as make azure
- Oryx Python Detection https://github.com/microsoft/Oryx/blob/main/doc/runtimes/python.md 
- Azure App Service looks for a file named app.py

## New app service -- US East for Basic Plan -- Try 2

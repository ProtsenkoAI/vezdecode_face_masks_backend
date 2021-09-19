# Effects server for Android 50

### before you start
* install pipenv
### install
```
git clone git@github.com:ProtsenkoAI/vezdecode_face_masks_backend.git
cd vezdecode_face_masks_backend/

pipenv install
pipenv shell

sudo apt-get install -y build-essential cmake libgtk-3-dev libboost-all-dev
pip install it
pip install dlib
```

### run
```
uvicorn run_server:app --reload
```
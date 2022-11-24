# yolo5_test
my yolo5 in colab

```bash
!pip install flask-ngrok flask==0.12.2 pyngrok==4.1.1
!ngrok authtoken 'xxxxx' # 填自己Authtoken
```

## Use
```python
from flask_ngrok import run_with_ngrok
from flask import Flask
app = Flask(__name__)
run_with_ngrok(app)   # 将flask app对象传递给run_with_ngrok函数
@app.route("/")
def home():
    return "<h1>Hello World!</h1>"
  
app.run()
```python

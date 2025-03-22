from flask import Flask
import onnxruntime as ort
print("Available providers:", ort.get_available_providers())
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)

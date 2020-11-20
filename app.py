from flask import Flask
import cv2

app = Flask(__name__)
@app.route('/home')

def index():
  return "hello"

if __name__ == "__main__":
  app.run
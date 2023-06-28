from flask import *
import os
import predict

app = Flask(__name__)


@app.route('/')
def upload():
    return render_template("upload.html")


@app.route('/success', methods=['POST'])
def success():
    if request.method == 'POST':
        f = request.files['file']
        f.save(f.filename)
        caption = predict.predict_cap(f.filename)
        print(caption)
        os.remove(f.filename)
        return render_template("success.html", name=caption)


if __name__ == '__main__':
    app.run()

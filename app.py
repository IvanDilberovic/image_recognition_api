from flask import Flask, jsonify, request
from flask_cors import CORS
import base64
import numpy as np
from PIL import Image
import io
from io import BytesIO
import pylab as pl
import matplotlib.ticker
import matplotlib.cm as cm
import os
from keras import backend as K
from keras.models import load_model
import tensorflow as tf

# For 2D data (e.g.  image), "tf" assumes (rows, cols, channels) while "th"
# assumes (channels, rows, cols).
K.set_image_dim_ordering("th")

app = Flask(__name__)
CORS(app)

load_model_name = "model.h5"
load_model_path = os.getcwd() + "/save/"
model = load_model(os.path.join(load_model_path, load_model_name))
model._make_predict_function()
graph = tf.get_default_graph()
print("Model loaded, printing summary...")
model.summary()
global_image = None


def nice_imshow(ax, data, vmin=None, vmax=None, cmap=None, name=None, size=None):

    if cmap is None:
        cmap = cm.get_cmap("jet")
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()

    ax.imshow(data, vmin=vmin, vmax=vmax, interpolation="nearest", cmap=cmap)
    ax.yaxis.set_visible(False)
    ax.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
    ax.xaxis.set_visible(False)
    ax.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
    byte_io = io.BytesIO()
    pl.savefig(byte_io, bbox_inches="tight", dpi=size)
    byte_io.seek(0)
    base_64 = base64.b64encode(byte_io.read())
    base_64 = base_64.decode("utf-8")

    return name, base_64


def make_mosaic(imgs, nrows, ncols, border=1):

    nimgs = imgs.shape[0]
    imshape = imgs.shape[1:]

    mosaic = np.zeros(
        (
            nrows * imshape[0] + (nrows - 1) * border,
            ncols * imshape[1] + (ncols - 1) * border,
        ),
        dtype=np.float32,
    )

    paddedh = imshape[0] + border
    paddedw = imshape[1] + border

    for i in range(nimgs):
        row = int(np.floor(i / ncols))
        col = i % ncols
        mosaic[
            row * paddedh : row * paddedh + imshape[0],
            col * paddedw : col * paddedw + imshape[1],
        ] = imgs[i]

    return mosaic


def set_cmap(layer):
    if layer == "conv2d_1" or layer == "conv2d_2":
        return cm.get_cmap("gray")
    else:
        return cm.get_cmap("binary")


def prepare_image(imageData):
    output = BytesIO(base64.b64decode(imageData))
    output.seek(0)

    image = Image.open(output).convert(
        "L"
    )  # need to convert because the image comes as ARGB
    image.thumbnail(size=(28, 28), resample=Image.ANTIALIAS)

    img_for_prediction = np.array(image).reshape(1, 1, 28, 28)
    img_for_prediction = img_for_prediction.astype("float32")
    img_for_prediction /= 255

    return img_for_prediction


@app.route("/")
def index():
    return "Hello world :)"


@app.route("/api/GetLayerNames", methods=["GET"])
def get_layer_names():

    try:
        list = []
        message = ""

        for layer in model.layers:
            if layer.output.shape.ndims == 2:
                continue
            list.append(layer.name)

        message = "OK"
        print("GetLayerNames()", message)

        return jsonify(
            {"success": True, "status_code": 200, "message": message, "results": list}
        )

    except Exception as ex:
        print(str(ex))
        return jsonify(
            {"success": False, "status_code": 500, "message": str(ex), "results": None}
        )


@app.route("/api/GetPrediction", methods=["GET", "POST"])
def get_prediction():

    try:
        imageData = request.json["image"].split("base64,")[1]

        img_for_prediction = prepare_image(imageData)

        global graph
        with graph.as_default():
            result = model.predict(img_for_prediction, batch_size=1, verbose=1)
            listaRezultata = result[0]
            list = []
            for i in sorted(
                enumerate(listaRezultata), key=lambda x: x[1], reverse=True
            ):
                dic = {}
                dic["key"] = i[0]
                dic["value"] = "{0:.16f}".format(i[1])
                list.append(dic)

            return jsonify(
                {"success": True, "status_code": 200, "message": "", "results": list}
            )

    except Exception as ex:
        print(str(ex))
        return jsonify(
            {"success": False, "status_code": 500, "message": str(ex), "results": None}
        )


@app.route("/api/GetLayerImage", methods=["GET", "POST"])
def get_layer_image():

    try:

        layer = request.json["layer"]
        image = request.json["image"].split("base64,")[1]
        image = prepare_image(image)
        message = ""
        list = []

        global graph
        with graph.as_default():

            for layer in layer:
                layer = layer["layer"]
                if model.get_layer(layer).output.shape.ndims == 2:
                    message = "Given layer does not have correct output dimensions so the image will not be created."

                inputs = [K.learning_phase()] + model.inputs
                _f = K.function(inputs, [model.get_layer(layer).output])
                C1 = _f([0] + [image])
                C1 = np.squeeze(C1)
                x, y = nice_imshow(
                    pl.gca(),
                    make_mosaic(C1, 4, 8),
                    cmap=set_cmap(layer),
                    name=layer + ".png",
                    size=400,
                )
                d = {}
                d["name"] = x
                d["picture"] = y
                list.append(d)

        message = "OK"

        print("GetLayerImage()", message)

        return jsonify(
            {
                "success": True,
                "status_code": 200,
                "message": message,
                "results": None,
                "images": list,
            }
        )

    except Exception as ex:
        print(str(ex))
        return jsonify(
            {
                "success": False,
                "status_code": 500,
                "message": str(ex),
                "results": None,
                "images": None,
            }
        )


@app.route("/api/GetWeightImage", methods=["GET"])
def get_weight_image():

    try:

        message = ""
        list = []

        global graph
        with graph.as_default():
            W = np.rollaxis(np.squeeze(model.layers[0].get_weights()[0]), 2, 0)
            x, y = nice_imshow(
                pl.gca(),
                make_mosaic(W, 4, 8),
                cmap=cm.get_cmap("gray"),
                name="Weights_1.png",
                size=50,
            )
            d = {}
            d["name"] = x
            d["picture"] = y
            list.append(d)
            WW = np.rollaxis(
                np.squeeze(model.layers[2].get_weights()[0][:, :, :, 0]), 2, 0
            )
            x, y = nice_imshow(
                pl.gca(),
                make_mosaic(WW, 4, 8),
                cmap=cm.get_cmap("gray"),
                name="Weights_2.png",
                size=50,
            )
            d = {}
            d["name"] = x
            d["picture"] = y
            list.append(d)

        message = "OK"
        print("GetWeightImage()", message)

        return jsonify(
            {
                "success": True,
                "status_code": 200,
                "message": message,
                "results": None,
                "images": list,
            }
        )

    except Exception as ex:
        print(str(ex))
        return jsonify(
            {
                "success": False,
                "status_code": 500,
                "message": str(ex),
                "results": None,
                "images": None,
            }
        )


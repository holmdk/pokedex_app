from starlette.applications import Starlette
from starlette.responses import JSONResponse, HTMLResponse, RedirectResponse

from fastai.vision import *

import torch
from pathlib import Path
from io import BytesIO
import sys
import uvicorn
import aiohttp
import asyncio


defaults.device = torch.device('cpu')

async def get_bytes(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()


app = Starlette()

cat_images_path = Path("D://pokedex_app/dataset")

cat_learner = load_learner(cat_images_path)


# cat_data = ImageDataBunch.from_folder(cat_images_path, train=".", valid_pct=0.2,
#         ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)
#
# cat_learner = create_cnn(cat_data, models.resnet34)
# cat_learner.load("model")



@app.route("/upload", methods=["POST"])
async def upload(request):
    data = await request.form()
    bytes = await (data["file"].read())
    return predict_image_from_bytes(bytes)


@app.route("/classify-url", methods=["GET"])
async def classify_url(request):
    bytes = await get_bytes(request.query_params["url"])
    return predict_image_from_bytes(bytes)


async def get_bytes(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.read()

# def predict_image_from_bytes(bytes):
#     img = open_image(BytesIO(bytes))
#     pred_class,pred_idx,outputs = cat_learner.predict(img)
#     return JSONResponse({
#         "prediction": str(pred_class),
#         "scores": sorted(
#             zip(cat_learner.data.classes, map(float, outputs)),
#             key=lambda p: p[1],
#             reverse=True
#         )
#     })

def predict_image_from_bytes(bytes):
    img = open_image(BytesIO(bytes))
    pred_class,pred_idx,losses = cat_learner.predict(img)
    return JSONResponse({
        "predictions": sorted(
            zip(cat_learner.data.classes, map(float, losses)),
            key=lambda p: p[1],
            reverse=True
        )
    })

# def predict_image_from_bytes(bytes):
#     img = open_image(BytesIO(bytes))
#     pred_class,pred_idx,outputs = cat_learner.predict(img)
#     return JSONResponse({
#         "prediction": str(pred_class),
#         "scores": sorted(
#             zip(cat_learner.data.classes, map(float, outputs)),
#             key=lambda p: p[1],
#             reverse=True
#         )
#     })


@app.route("/")
def form(request):
    return HTMLResponse(
        """
        <form action="/upload" method="post" enctype="multipart/form-data">
            Select image to upload:
            <input type="file" name="file">
            <input type="submit" value="Upload Image">
        </form>
        Or submit a URL:
        <form action="/classify-url" method="get">
            <input type="url" name="url">
            <input type="submit" value="Fetch and analyze image">
        </form>
    """)


@app.route("/form")
def redirect_to_homepage(request):
    return RedirectResponse("/")


if __name__ == "__main__":
    if "serve" in sys.argv:
        uvicorn.run(app, host="127.0.0.1", port=5000)

# pokedex_app
SEE RUNNING CODE HERE (might not be running at all time due to cost of service)  
https://pokedex-classifier.onrender.com/  
  
FastAI implementation of a Resnet50 Pokedex classifier (Gen1) using Starlette web app  
Based on the following repo: https://github.com/render-examples/fastai-v3  
See this guide for more details https://course.fast.ai/deployment_render.html  


### Performance
Accuracy of around 85% for 10,000 images of Gen1 pokemon - acquired from the following database:  
https://www.kaggle.com/thedagger/pokemon-generation-one/version/1  
  
Primary errors comes from Pokemon evolutions (Machamp / Machop for example) as expected


#### Testing locally
python app/server.py serve

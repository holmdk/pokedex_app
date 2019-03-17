from fastai.vision import *
from fastai.metrics import error_rate


bs = 64
# bs = 16   # uncomment this line if you run out of memory even after clicking Kernel->Restart

path_img = "D:/pokedex_app/dataset"


data = ImageDataBunch.from_folder(path_img, train=".", valid_pct=0.2,
        ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)


# Train model

learn = create_cnn(data, models.resnet34, metrics=error_rate)

learn.fit_one_cycle(4)


learn.save('stage-1')

learn.unfreeze()

learn.lr_find()


learn.fit_one_cycle(2, max_lr=slice(3e-5,3e-4))

learn.save('stage-2')

# Interpretation
learn.load('stage-2');
#
# interp = ClassificationInterpretation.from_learner(learn)
#
# interp.plot_confusion_matrix()
#
#
# learn.recorder.plot_losses()
#
#
# # evaluation
# interp.plot_top_losses(9, figsize=(15,11))
#
# interp.most_confused(min_val=1)


learn.export()
#
# img_test = cv2.imread('D:/test1.png')
# plt.imshow(img_test)
#
# defaults.device = torch.device('cpu')
#
# img = open_image('D:/test3.png')
#
# pred_class,pred_idx,outputs = learn.predict(img)
# pred_class
#

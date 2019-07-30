from fastai.groups.default_cnn import *
PATH = '/data/msnow/nih_cxr/'


arch = resnet34
sz = 64
bs = 64
# data = get_data(sz,bs)
# learn = ConvLearner.pretrained(arch, data)


# def get_data(sz, bs):
#     tfms = tfms_from_model(arch, sz, aug_tfms=transforms_basic, max_zoom=1.05)
#     return ImageClassifierData.from_csv(PATH, 'trn', f'{PATH}data_trn.csv', tfms=tfms,
#                     val_idxs=val_idx, test_name='tst', bs=bs, cat_separator='|')
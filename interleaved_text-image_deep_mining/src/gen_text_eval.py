import re, math, sys, os, csv
import nltk.data
import skimage.io
import numpy as np
from random import shuffle
from scipy.ndimage import zoom
from skimage.transform import resize
from gensim.models import word2vec
from nltk.corpus import stopwords
import h5py
import pickle
from scipy import spatial


ostype = sys.platform
homedir = '___home directory___'
hostadr = '___localhost___'


with open(os.path.join(homedir, '___path to the workspace___', 'ontology_DO_list.pickle')) as f:
  ontology_list = pickle.load(f)

model_file = os.path.join(homedir, '___path to the workspace___', 'wordvecs',\
  'ris_openi_alphanumeric-non-numbers_vectors_cbow0_size256_window10_negative0_hs1_sample1e-4.bin')
model = word2vec.Word2Vec.load_word2vec_format(model_file, binary=True)


caffe_root = os.path.expanduser('___path to the Caffe installation___')
sys.path.append(caffe_root + 'python')


import caffe

###
TXT_MODEL_BASE = '__path to the workspace__/models/ris_annotation/'+\
  'vgg_19_rgb_to_ris_sentpf_tp1000_txt/'
TXT_MODEL_FILE = TXT_MODEL_BASE + 'deploy.prototxt'
TXT_PRETRAINED = TXT_MODEL_BASE + 'ris_sentpf1000_VGG_ILSVRC_19_rgb2txt_iter_30000.caffemodel'

IM_MODEL_BASE_H1 = '__path to the workspace__/models/ris_annotation/'+\
  'vgg_19_rgb_to_ris_tp80_im_startImagenet/'
IM_MODEL_FILE_H1 = IM_MODEL_BASE_H1 + 'deploy.prototxt'
IM_PRETRAINED_H1 = IM_MODEL_BASE_H1 + 'ris80_VGG_ILSVRC_19_rgb2im_startImagenet_iter_200000.caffemodel'

IM_MODEL_BASE_H2 = '__path to the workspace__/models/ris_annotation/'+\
  'vgg_19_rgb_to_ris_tp80_h2_im_startTp80H1/'
IM_MODEL_FILE_H2 = IM_MODEL_BASE_H2 + 'deploy.prototxt'
IM_PRETRAINED_H2 = IM_MODEL_BASE_H2 + 'ris80_h2_VGG_ILSVRC_19_rgb2im_startTp80H1_iter_200000.caffemodel'

IM_MODEL_BASE_H3 = '__path to the workspace__/models/ris_annotation/'+\
  'vgg_19_rgb_to_ris_sentpf_tp1000_im_startTp80H2/'
IM_MODEL_FILE_H3 = IM_MODEL_BASE_H3 + 'deploy.prototxt'
IM_PRETRAINED_H3 = IM_MODEL_BASE_H3 + 'ris_sentpf1000_VGG_ILSVRC_19_rgb2im_startTp80H2_iter_200000.caffemodel'

txt_net_1 = caffe.Classifier(TXT_MODEL_FILE, TXT_PRETRAINED,
  mean = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'),
  channel_swap=(2,1,0),
  raw_scale=255,
  image_dims=(256,256))
###
im_net_1_H1 = caffe.Classifier(IM_MODEL_FILE_H1, IM_PRETRAINED_H1,
  mean = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'),
  channel_swap=(2,1,0),
  raw_scale=255,
  image_dims=(256,256))
###
im_net_1_H2 = caffe.Classifier(IM_MODEL_FILE_H2, IM_PRETRAINED_H2,
  mean = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'),
  channel_swap=(2,1,0),
  raw_scale=255,
  image_dims=(256,256))
###
im_net_1_H3 = caffe.Classifier(IM_MODEL_FILE_H3, IM_PRETRAINED_H3,
  mean = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'),
  channel_swap=(2,1,0),
  raw_scale=255,
  image_dims=(256,256))
###

txt_net_1.set_phase_test()
txt_net_1.set_mode_cpu()

im_net_1_H1.set_phase_test()
im_net_1_H1.set_mode_cpu()

im_net_1_H2.set_phase_test()
im_net_1_H2.set_mode_cpu()

im_net_1_H3.set_phase_test()
im_net_1_H3.set_mode_cpu()
###

wordlen = 2
vecsize = 256

lbDest = os.path.join('./gen_test_eval')
if not os.path.exists(lbDest):
  os.makedirs(lbDest)
testImDest = os.path.join('./gen_test_eval/test_ims')
if not os.path.exists(testImDest):
  os.makedirs(testImDest)


imloc0 = os.path.expanduser('all_images_256x256')

outFileName = os.path.join(lbDest, 'gen_text_eval.csv')
outFile = open(outFileName, 'w')

outFile.write('matchCount,avgDist,fname,keyWords,diseaseTruth,diseaseTop1,diseaseTop1Prob,'+\
  'diseaseTop2,diseaseTop2Prob,diseaseTop3,diseaseTop3Prob,diseaseTop4,diseaseTop4Prob,'+\
  'diseaseTop5,diseaseTop5Prob,origSents\n')

diseaseTerms = []
with open('t047_terms.csv', 'rb') as csvf:
  fr = csv.reader(csvf, delimiter=',')
  for row in fr:
    diseaseTerms.append(row[1])

with open('umls_radlex_t047_im2lbTpSentpfs_test.txt', 'rb') as csvfr:
  lbtr = csv.reader(csvfr, delimiter=',')
  for row in lbtr:
    if row[0]=='fname': continue

    fname = row[0]
    dslabel = row[1]
    tpsentpf = row[5]
    imloc = os.path.join(imloc0, fname)
    
    try:
      input_image = caffe.io.load_image(imloc)
      with open(imloc, 'rb') as f:
        imjpg = f.read()
    except:
      continue
    testedIm = os.path.join(testImDest, fname)
    if os.path.isfile(testedIm):
      continue
    with open(testedIm, 'wb') as f:
      f.write(imjpg)

    txt_prediction = txt_net_1.predict([input_image])*2-1

    im_prediction_1_H1 = im_net_1_H1.predict([input_image])
    im_prediction_1_H2 = im_net_1_H2.predict([input_image])
    im_prediction_1_H3 = im_net_1_H3.predict([input_image])
    
    im_prediction_H1 = im_prediction_1_H1[0].argmax()
    im_prediction_H2 = im_prediction_1_H2[0].argmax()
    im_prediction_H3 = im_prediction_1_H3[0].argmax()

    ###

    oldLH1 = []
    newLH1 = []
    mappingsH1File = '__path to the workspace__/src/ris_annotation/lda_top_words/mappings_risImNtp80Ntr1.csv'
    with open(mappingsH1File) as csvfile:
      reader = csv.reader(csvfile, delimiter=',')
      for row in reader:
        if not row[0][0].isdigit() or int(row[1])<0:
          continue
        newLH1.append(row[0])
        oldLH1.append(row[1])

    topicsH1 = []
    termsH1 = []
    topTermsFileH1 = '__path to the workspace__/src/ris_annotation/lda_top_words/ris_reports-top-terms.csv'
    f = open(topTermsFileH1, 'r')
    termsH1 = f.readlines()
    termsH1.pop(0)     


    oldLH2 = []
    newLH2 = []
    mappingsH2File = '__path to the workspace__/src/ris_annotation/lda_top_words/mappings_risImH2Ntp80h10Ntr1.csv'
    with open(mappingsH2File) as csvfile:
      reader = csv.reader(csvfile, delimiter=',')
      for row in reader:
        if not row[0][0].isdigit() or int(row[1])<0:
          continue
        newLH2.append(row[0])
        oldLH2.append(row[1])

    topicsH2 = []
    termsH2 = []
    topTermsFileH2 = '__path to the workspace__/src/ris_annotation/lda_top_words/ris_reports_per_h1_topic_'+\
      str(im_prediction_H1) + '-top-terms.csv'
    f = open(topTermsFileH2, 'r')
    termsH2 = f.readlines()
    termsH2.pop(0)


    oldLH3 = []
    newLH3 = []
    mappingsH3File = '__path to the workspace__/src/ris_annotation/lda_top_words/mappings_risSentPfNtp1000Ntr1.csv'
    with open(mappingsH3File) as csvfile:
      reader = csv.reader(csvfile, delimiter=',')
      for row in reader:
        if not row[0][0].isdigit() or int(row[1])<0:
          continue
        newLH3.append(row[0])
        oldLH3.append(row[1])

    topicsH3 = []
    termsH3 = []
    topTermsFileH1 = '__path to the workspace__/src/ris_annotation/lda_top_words/ris_sentences_previous_following-top-terms.csv'
    f = open(topTermsFileH1, 'r')
    termsH3 = f.readlines()
    termsH3.pop(0)

    ###

    topic = int(newLH1[im_prediction_H1])
    topic_terms_h1 = termsH1[topic].rstrip().split(',')
    topic_terms_h1.pop(0)

    topic0 = newLH2[im_prediction_H2]
    topic0l = topic0.split('_')
    topic = int(topic0l[1])
    topic_terms_h2 = termsH2[topic].rstrip().split(',')
    topic_terms_h2.pop(0)

    topic = int(newLH3[im_prediction_H3])
    topic_terms_h3 = termsH3[topic].rstrip().split(',')
    topic_terms_h3.pop(0)

    ###

    txt_prediction_i = txt_prediction

    ###
    max_dist_H1_1 = 0
    max_dist_H1_2 = 0
    term_H1_1 = ''
    term_H1_2 = ''
    txt_prediction_i_H1 = np.zeros((1,512))
    topic_terms_h1_DO = list(set(topic_terms_h1).intersection(set(ontology_list)))
    for term_h1 in topic_terms_h1_DO:
      try:
        ttv = model.syn0norm[model.vocab[term_h1].index]
      except:
        pass
      txt_prediction_i_H1 += np.concatenate((ttv, ttv), axis=1)
      #

    for term_h1 in topic_terms_h1_DO:
      txt_pred_1 = txt_prediction_i[0,0:256]*txt_prediction_i_H1[0,0:256]
      txt_pred_2 = txt_prediction_i[0,256::]*txt_prediction_i_H1[0,256::]
      dist_1 = spatial.distance.cosine(ttv, txt_pred_1)
      if dist_1>max_dist_H1_1:
        max_dist_H1_1 = dist_1
        term_H1_1 = term_h1

      dist_2 = spatial.distance.cosine(ttv, txt_pred_2)
      if dist_2>max_dist_H1_2:
        max_dist_H1_2 = dist_2
        term_H1_2 = term_h1
    term_H1s = [term_H1_1, term_H1_2]


    max_dist_H2_1 = 0
    max_dist_H2_2 = 0
    term_H2_1 = ''
    term_H2_2 = ''
    txt_prediction_i_H2 = np.zeros((1,512))
    topic_terms_h2_DO = list(set(topic_terms_h2).intersection(set(ontology_list)))
    for term_h2 in topic_terms_h2_DO:
      try:
        ttv = model.syn0norm[model.vocab[term_h2].index]
      except:
        pass
      txt_prediction_i_H2 += np.concatenate((ttv, ttv), axis=1)
      #

    for term_h2 in topic_terms_h2_DO:
      txt_pred_1 = txt_prediction_i[0,0:256]*txt_prediction_i_H2[0,0:256]
      txt_pred_2 = txt_prediction_i[0,256::]*txt_prediction_i_H2[0,256::]
      dist_1 = spatial.distance.cosine(ttv, txt_pred_1)
      if dist_1>max_dist_H2_1:
        max_dist_H2_1 = dist_1
        term_H2_1 = term_h2

      dist_2 = spatial.distance.cosine(ttv, txt_pred_2)
      if dist_2>max_dist_H2_2:
        max_dist_H2_2 = dist_2
        term_H2_2 = term_h2
    term_H2s = [term_H2_1, term_H2_2]


    max_dist_H3_1 = 0
    max_dist_H3_2 = 0
    term_H3_1 = ''
    term_H3_2 = ''
    txt_prediction_i_H3 = np.zeros((1,512))
    topic_terms_h3_DO = list(set(topic_terms_h3).intersection(set(ontology_list)))
    for term_h3 in topic_terms_h3_DO:
      try:
        ttv = model.syn0norm[model.vocab[term_h3].index]
      except:
        pass
      txt_prediction_i_H3 += np.concatenate((ttv, ttv), axis=1)

    for term_h3 in topic_terms_h3_DO:
      txt_pred_1 = txt_prediction_i[0,0:256]*txt_prediction_i_H3[0,0:256]
      txt_pred_2 = txt_prediction_i[0,256::]*txt_prediction_i_H3[0,256::]
      dist_1 = spatial.distance.cosine(ttv, txt_pred_1)
      if dist_1>max_dist_H3_1:
        max_dist_H3_1 = dist_1
        term_H3_1 = term_h3

      dist_2 = spatial.distance.cosine(ttv, txt_pred_2)
      if dist_2>max_dist_H3_2:
        max_dist_H3_2 = dist_2
        term_H3_2 = term_h3
    term_H3s = [term_H3_1, term_H3_2]

    term_list = list(set(term_H1s + term_H2s + term_H3s))
    terms = ' '.join(term_list)
    term_list2 = terms.lstrip().rstrip().split()


    ###
    modelFileDT = os.path.join('__path to the workspace__',\
      'models', 'ris_annotation', 'googlenet_rgb_to_ris_terms_umls_radlex_t047',\
      'deploy.prototxt')
    
    ptfileDT = os.path.join('__path to the workspace__',\
      'models', 'ris_annotation', 'googlenet_rgb_to_ris_terms_umls_radlex_t047',\
      'googlenet_rgb_to_ris_terms_umls_radlex_t047_startTp1000H3_iter_200000.caffemodel')
    
    meanfileDT = os.path.join('__path to the workspace__',\
      'models', 'ris_annotation', 'googlenet_rgb_to_ris_terms_umls_radlex_t047',\
      'ilsvrc_2014_mean.npy')
        
    netDT = caffe.Classifier(modelFileDT, ptfileDT, mean=np.load(meanfileDT),\
      channel_swap=(2,1,0), raw_scale=255, image_dims=(256,256))

    netDT.set_phase_test()
    netDT.set_mode_cpu()

    pred = netDT.predict([input_image])
    predcs0 = np.argsort(pred[0])
    predcs = predcs0[::-1]
    probis = pred[0][predcs[:5]]

    ###
    dist = 0
    matchCount = 0
    totCount = 0
    for tpsentpfi in tpsentpf.split():
      if tpsentpfi in term_list2:
        matchCount += 1
    for term_list2i in term_list:
      if term_list2i in ontology_list:
        try:
          dist += model.similarity(tpsentpfi, term_list2i)
          totCount += 1
        except:
          pass
    if totCount>0:
      avgDist = 1.0*dist/totCount
    else:
      avgDist = 0
    outFile.write(str(matchCount) + ',' + str(avgDist) + ',' +\
      fname + ',' + ' '.join(term_list2) + ',' +\
      diseaseTerms[int(dslabel)] + ',' + diseaseTerms[predcs[0]] + ',' + str(probis[0]) + ',' +\
      diseaseTerms[predcs[1]] + ',' + str(probis[1]) + ',' +\
      diseaseTerms[predcs[2]] + ',' + str(probis[2]) + ',' +\
      diseaseTerms[predcs[3]] + ',' + str(probis[3]) + ',' +\
      diseaseTerms[predcs[4]] + ',' + str(probis[4]) + ',' +\
      tpsentpf + '\n')
outFile.close()


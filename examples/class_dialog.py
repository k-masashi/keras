# coding: utf-8

from keras.layers.core import Dense
from keras.layers.core import Masking
from keras.layers.core import Activation
from keras.layers.core import Dropout
from keras.layers import Input
from keras.layers import Lambda
from keras.layers import Multiply
from keras.layers import Maximum
from keras.layers import Concatenate
from keras.layers import Add
from keras.layers import Reshape
from keras.models import Model
from keras.layers.recurrent import SimpleRNN
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.initializers import glorot_uniform
from keras.initializers import uniform
from keras.initializers import orthogonal
from keras.initializers import TruncatedNormal
from keras.initializers import Ones
from keras import regularizers
from keras import backend as K
from keras.utils import np_utils
from keras.utils import plot_model
from keras.constraints import max_norm
from keras.constraints import unit_norm

import tensorflow as tf
import numpy as np
import csv
import random
import numpy.random as nr
import keras
import sys
import math
import pickle
import time
import glob
import gc
import os

class Color:
    BLACK     = '\033[30m'
    RED       = '\033[31m'
    GREEN     = '\033[38;5;10m'
    YELLOW    = '\033[33m'
    BLUE      = '\033[34m'
    PURPLE    = '\033[35m'
    CYAN      = '\033[38;5;14m'
    WHITE     = '\033[37m'
    END       = '\033[0m'
    BOLD      = '\038[1m'
    UNDERLINE = '\033[4m'
    INVISIBLE = '\033[08m'
    REVERCE   = '\033[07m'

#*******************************************************************************
#                                                                              *
# レイヤークラス定義                                                           *
#                                                                              *
#*******************************************************************************
class Layer_LSTM :
    def __init__(self, reg_lambda=0.01, seed=20170719):
        self.seed      = seed
        self.reg_lambda = reg_lambda

    def create_LSTM(self, lstm_units, 
                    lstm_return_state=False, lstm_return_sequences=False, 
                    lstm_go_backwards=False, lstm_name='LSTM') :
        layer = LSTM(lstm_units, name=lstm_name  ,        
                 return_state=lstm_return_state,
                 return_sequences=lstm_return_sequences,
                 go_backwards=lstm_go_backwards, 
                 #recurrent_regularizer=regularizers.l2(self.reg_lambda) ,
                 #kernel_regularizer=regularizers.l2(self.reg_lambda) ,
                 kernel_initializer=glorot_uniform(seed=self.seed), 
                 recurrent_initializer=orthogonal(gain=1.0, seed=self.seed),
                 bias_initializer=Ones(),
                 dropout=0.5, recurrent_dropout=0.5
                )  
        return layer

class Layer_Dense :
    def __init__(self, reg_lambda=0.01, seed=20170719):
        self.seed      = seed
        self.reg_lambda = reg_lambda

    def create_Dense(self, 
                     dense_units, dense_activation=None, dense_name='Dense'):
        if dense_activation==None :
            act_reg = None 
        else :
            act_reg = regularizers.l1(self.reg_lambda)
        layer = Dense(dense_units, name=dense_name, 
                      activation=dense_activation,
                      kernel_initializer=glorot_uniform(seed=self.seed),
                      #kernel_regularizer=regularizers.l2(self.reg_lambda) ,
                      #bias_regularizer=regularizers.l2(self.reg_lambda) ,
                      activity_regularizer=act_reg,
                     )        
        return layer        

class Layer_BatchNorm :
    def __init__(self, max_value=2, reg_lambda=0.01):
        self.max_value = max_value
        self.reg_lambda = reg_lambda
    def create_BatchNorm(self, bn_name='BatchNorm'):
        layer = BatchNormalization(axis=-1,
                   name=bn_name,
                   #beta_regularizer=regularizers.l2(self.reg_lambda) ,
                   #gamma_regularizer=regularizers.l2(self.reg_lambda) ,
                   beta_constraint=max_norm(max_value=self.max_value, axis=0),
                   gamma_constraint=max_norm(max_value=self.max_value, axis=0)
                  )  
        return layer   

#*******************************************************************************    
#                                                                              *
# ニューラルネットワーククラス定義                                             *
#                                                                              *
#*******************************************************************************        
class Dialog :
    def __init__(self, maxlen_e, maxlen_d, n_hidden, 
                 input_dim, vec_dim, output_dim):
        self.n_hidden   = n_hidden
        self.maxlen_e   = maxlen_e
        self.maxlen_d   = maxlen_d
        self.input_dim  = input_dim
        self.vec_dim    = vec_dim
        self.output_dim = output_dim

    #***************************************************************************
    #                                                                          *
    # ニューラルネットワーク定義                                               *
    #                                                                          *
    #***************************************************************************
    def create_model(self):
        len_norm = 2                               # constraintの最大ノルム長
        r_lambda = 0.00005                          # regularizerのラムダ

        #***********************************************************************
        #                                                                      *
        #  レイヤクラス生成                                                    *
        #                                                                      *
        #***********************************************************************           
        class_Dense = Layer_Dense(reg_lambda=r_lambda)
        class_LSTM  = Layer_LSTM(reg_lambda=r_lambda)
        class_BatchNorm = Layer_BatchNorm(max_value=len_norm, 
                                          reg_lambda=r_lambda)
        print('#3')
        #***********************************************************************
        #                                                                      *
        #  エンコーダー（学習／応答文作成兼用）                                *
        #                                                                      *
        #***********************************************************************   
        #---------------------------------------------------------
        #レイヤー定義
        #---------------------------------------------------------
        embedding = Embedding(output_dim=self.vec_dim, input_dim=self.input_dim, 
                              mask_zero=True, name='Embedding', 
                              embeddings_initializer=uniform(seed=20170719),
                              #embeddings_regularizer=regularizers.l2(r_lambda),
                             )
        input_mask = Masking(mask_value=0, name="input_Mask")
        encoder_BatchNorm \
        = class_BatchNorm.create_BatchNorm(bn_name='encoder_BatchNorm')
        encoder_LSTM  = class_LSTM.create_LSTM(self.n_hidden, 
                                               lstm_return_state=True,  
                                               lstm_name='encoder_LSTM')     

        #---------------------------------------------------------
        # 入力定義
        #---------------------------------------------------------       
        encoder_input = Input(shape=(self.maxlen_e,), 
                              dtype='int32', 
                              name='encorder_input')
        e_input       = input_mask(encoder_input)
        e_input       = embedding(e_input)
        e_input       = encoder_BatchNorm(e_input)
        #---------------------------------------------------------
        # メイン処理
        #---------------------------------------------------------
        encoder_outputs, \
        encoder_state_h, \
        encoder_state_c = encoder_LSTM(e_input)
        #---------------------------------------------------------
        # エンコーダモデル定義
        #---------------------------------------------------------        
        encoder_model = Model(inputs=encoder_input,
                              outputs=[encoder_outputs, 
                                       encoder_state_h, 
                                       encoder_state_c
                                      ])                                

        print('#4')
        #***********************************************************************
        # デコーダー（学習用）                                                 *
        # デコーダを、完全な出力シークエンスを返し、内部状態もまた返すように   *
        # 設定します。                                                         *
        # 訓練モデルではreturn_sequencesを使用しませんが、推論では使用します。 *     
        #***********************************************************************                
        #=======================================================================
        #レイヤー定義
        #=======================================================================
        #---------------------------------------------------------
        # デコーダー入力Batch Normalization
        #---------------------------------------------------------
        decoder_BatchNorm \
        = class_BatchNorm.create_BatchNorm(bn_name='decoder_BatchNorm')

        #---------------------------------------------------------
        # デコーダーLSTM
        #---------------------------------------------------------
        decoder_LSTM = class_LSTM.create_LSTM(self.n_hidden, 
                                              lstm_return_state=True, 
                                              lstm_return_sequences=True, 
                                              lstm_name='decode_LSTM')  

        #---------------------------------------------------------
        # 全結合
        #---------------------------------------------------------
        decoder_Dense = class_Dense.create_Dense(self.output_dim, 
                                                 dense_activation='softmax', 
                                                 dense_name='decoder_Dense')        

        #=======================================================================
        # 関数定義
        #=======================================================================
        #--------------------------------------------------------
        # decoderメイン処理
        #-------------------------------------------------------- 
        def decoder_main(d_i, encoder_states) : 
            # LSTM
            d_outputs, \
            decoder_state_h, \
            decoder_state_c = decoder_LSTM(d_i, initial_state=encoder_states)   
            # 全結合
            decoder_outputs = decoder_Dense(d_outputs)

            return decoder_outputs, decoder_state_h, decoder_state_c

        #=======================================================================    
        # 手続き部
        #=======================================================================            
        #---------------------------------------------------------
        #入力定義
        #---------------------------------------------------------
        decoder_inputs = Input(shape=(self.maxlen_d,), 
                               dtype='int32', name='decoder_inputs')        
        d_i = Masking(mask_value=0)(decoder_inputs)   
        d_i = embedding(d_i)
        d_i = decoder_BatchNorm(d_i)
        d_input = d_i                                      # 応答文生成で使う             

        #-----------------------------------------------------
        # decoder処理実行
        #-----------------------------------------------------     
        encoder_states = [encoder_state_h, encoder_state_c]
        decoder_outputs, _, _  = decoder_main(d_i, encoder_states) 

        #======================================================================= 
        # 損失関数、評価関数とモデル定義
        #=======================================================================                    
        #---------------------------------------------------------
        # 損失関数
        #---------------------------------------------------------       
        mask = Lambda(lambda x: K.sign(x))(decoder_inputs)
        def cross_loss(y_true, y_pred) :
            perp_mask      = K.cast(mask,dtype='float32')
            sum_mask       = K.sum(perp_mask, axis=-1, keepdims= True)
            #print('perp_mask1',K.int_shape(perp_mask))
            epsilons       = 1 / 2 * y_pred + K.epsilon()
            cliped         = K.maximum(y_pred, epsilons)
            log_pred       = -K.log(cliped)
            cross_e        = y_true * log_pred
            cross_e        = K.sum(cross_e, axis=-1)
            masked_entropy = perp_mask * cross_e
            sum_entropy    = K.sum(masked_entropy, axis=-1, keepdims= True)
            celoss         = sum_entropy / sum_mask
            celoss         = K.repeat(celoss , self.maxlen_d)            
            return celoss

        #---------------------------------------------------------
        # perplexity
        #---------------------------------------------------------       
        def get_perplexity(y_true, y_pred) :
            perp_mask      = K.cast(mask,dtype='float32')
            sum_mask       = K.sum(perp_mask, axis=-1, keepdims= True)
            epsilons       = 1 / 2 * y_pred + K.epsilon()
            cliped         = K.maximum(y_pred, epsilons)
            log_pred       = -K.log(cliped)
            cross_e        = y_true * log_pred
            cross_e        = K.sum(cross_e, axis=-1)
            masked_entropy = perp_mask * cross_e
            sum_entropy    = K.sum(masked_entropy, axis=-1, keepdims= True)
            perplexity     = sum_entropy / sum_mask
            perplexity     = K.exp(perplexity) 
            perplexity     = K.repeat(perplexity , self.maxlen_d)
            return perplexity

        #---------------------------------------------------------
        # 評価関数
        #---------------------------------------------------------  
        def get_accuracy(y_true, y_pred) :
            y_pred_argmax = K.argmax(y_pred, axis=-1)
            y_true_argmax = K.argmax(y_true, axis=-1)
            n_correct     = K.abs(y_true_argmax - y_pred_argmax)
            n_correct     = K.sign(n_correct)
            n_correct     = K.ones_like(n_correct, dtype='int64') - n_correct
            n_correct     = K.cast(n_correct, dtype='int32')
            n_correct     = n_correct * mask
            n_correct     = K.cast(K.sum(n_correct, axis=-1, keepdims= True), 
                                   dtype='float32')
            n_total       = K.cast(K.sum(mask,axis=-1, keepdims= True), 
                                   dtype='float32')
            accuracy      = n_correct / n_total
            accuracy      = K.repeat(accuracy , self.maxlen_d)
            #print('accuracy',K.int_shape(accuracy))
            return accuracy

        #---------------------------------------------------------
        # モデル定義、コンパイル
        #---------------------------------------------------------
        model = Model(inputs=[encoder_input, decoder_inputs],
                      outputs=decoder_outputs) 
        model.compile(loss=cross_loss,
                      optimizer="Adam", metrics=[get_perplexity, get_accuracy])    

        #***********************************************************************
        #                                                                      *
        # デコーダー（応答文作成）                                             *
        #                                                                      *     
        #***********************************************************************                
        print('#6')
        #---------------------------------------------------------
        #入力定義
        #---------------------------------------------------------        
        decoder_input_state_h = Input(shape=(self.n_hidden,),
                                      name='decoder_input_state_h')
        decoder_input_state_c = Input(shape=(self.n_hidden,),
                                      name='decoder_input_state_c')
        #---------------------------------------------------------
        # デコーダー実行
        #--------------------------------------------------------- 
        decoder_input_state = [decoder_input_state_h, decoder_input_state_c]
        res_decoder_outputs, \
        res_decoder_state_h, \
        res_decoder_state_c = decoder_main(d_input, decoder_input_state)                       

        print('#7')
        #---------------------------------------------------------
        # モデル定義
        #---------------------------------------------------------  
        decoder_model = Model(inputs= [decoder_inputs, 
                                       decoder_input_state_h, 
                                       decoder_input_state_c],
                              outputs=[res_decoder_outputs, 
                                       res_decoder_state_h, 
                                       res_decoder_state_c] )    

        return model, encoder_model, decoder_model

    #***********************************************************************
    #                                                                      *
    # 学習                                                                 *
    #                                                                      *     
    #***********************************************************************               
    def train(self, e_input, d_input, target, 
              batch_size, epochs, emb_param)  :

        print ('#2',target.shape)
        model ,encoder_model , decoder_model = self.create_model()  

        if os.path.isfile(emb_param) :
            model.load_weights(emb_param)    #埋め込みパラメータセット

        # ネットワーク図出力    
        plot_model(model, show_shapes=True,to_file='model.png') 
        plot_model(encoder_model, show_shapes=True,
                   to_file='encoder_model.png') 
        plot_model(decoder_model, show_shapes=True,
                   to_file='decoder_model.png') 
        print ('#8 number of params :', model.count_params())    

        #===================================================================
        # train on batch
        #===================================================================
        row=d_input.shape[0]
        loss_bk = 10000
        perplexity_bk = 10000
        accuracy_bk = 0
        patience = 0
        print(Color.CYAN,model.metrics_names[0]+" "
              +model.metrics_names[1]+" "
              +model.metrics_names[2] ,
              Color.END)

        for j in range(0,epochs) :
            print(Color.CYAN,"Epoch ",j+1,"/",epochs,Color.END)
            loss, \
            perplexity, \
            accuracy = self.on_batch(model,  
                                     e_input, d_input, target,
                                     batch_size, emb_param)

            #-----------------------------------------------------
            # EarlyStopping
            #-----------------------------------------------------            
            if j == 0 or (loss       <= loss_bk and 
                          perplexity <= perplexity_bk and 
                          accuracy   >= accuracy_bk):
                loss_bk = loss 
                perplexity_bk = perplexity
                accuracy_bk = accuracy
                patience = 0
            elif patience < 1  :
                patience += 1
            else :
                print('EarlyStopping') 
                break 

        return model        

    #***********************************************************************
    #                                                                      *
    # train_on_batch処理                                                   *
    #                                                                      *     
    #*********************************************************************** 
    def on_batch(self, model, e_input, d_input, target, 
                 batch_size, emb_param) :

        n_split = int(d_input.shape[0]*0.1)  
        e_val   = e_input[:n_split,:]
        d_val   = d_input[:n_split,:]
        t_val   = target[:n_split,:]
        e_train = e_input[n_split:,:]
        d_train = d_input[n_split:,:]
        t_train = target[n_split:,:]

        params = {'model'     : model,
                  'batch_size': batch_size,
                  'emb_param' : emb_param }
        _, _, _ = self.train_test_main('train', 
                                       e_train, d_train, t_train, params)
        model.save_weights(emb_param)

        return self.eval_perplexity(model, e_val, d_val, t_val, batch_size)

    #***********************************************************************
    #                                                                      *
    # perplexity計算                                                       *
    #                                                                      *     
    #***********************************************************************     
    def eval_perplexity(self, model, e_test, d_test, t_test, batch_size) :
        params = {'model'     : model,
                  'save_model': '',
                  'batch_size': batch_size,
                  'emb_param' : '' }
        return self.train_test_main('test', e_test, d_test, t_test, params)   

    #***********************************************************************
    #                                                                      *
    # 訓練／テストメイン計算                                               *
    #                                                                      *     
    #***********************************************************************         
    def train_test_main(self, kind, e_train, d_train, t_train, params) :  
        model      = params['model']
        batch_size = params['batch_size'] 
        emb_param  = params['emb_param'] 
        #損失関数、評価関数の平均計算用リスト
        list_loss = []
        list_perplexity =[]
        list_accuracy =[]

        s_time = time.time()
        row=d_train.shape[0]
        n_batch = math.ceil(row/batch_size)
        for i in range(0,n_batch) :
            s = i*batch_size
            e = min([(i+1) * batch_size,row])
            e_on_batch = e_train[s:e,:]
            d_on_batch = d_train[s:e,:]
            t_on_batch = t_train[s:e,:]
            t_on_batch = np_utils.to_categorical(t_on_batch, 
                                                 self.output_dim)
            if kind == 'train' :
                result = model.train_on_batch([e_on_batch, d_on_batch], 
                                              t_on_batch)
            else :
                result = model.test_on_batch([e_on_batch, d_on_batch], 
                                             t_on_batch)

            list_loss.append(result[0])
            list_perplexity.append(result[1])
            list_accuracy.append(result[2])
            elapsed_time = time.time() - s_time
            if i % 100 == 0 :
                sys.stdout.write("\r"
                                 +"                "
                                 +"                "
                                 +"                "
                                 +"                "
                                 +"                "
                                 +"                "
                                )
                sys.stdout.flush()
            if kind == 'train' :
                ctl_color = Color.CYAN
            else :
                ctl_color = Color.GREEN
            sys.stdout.write(ctl_color 
                 + "\r"+str(e)+"/"+str(row)+" "
                 + str(int(elapsed_time))+"s      "+"\t"
                 + "{0:.4f}".format(np.average(list_loss)) + "\t"
                 + "{0:.4f}".format(np.average(list_perplexity)) + "\t"
                 + "{0:.4f}".format(np.average(list_accuracy)) 
                 + Color.END) 
            sys.stdout.flush()
            if i % 100 == 99 and kind == 'train':
                model.save_weights(emb_param)
            del e_on_batch, d_on_batch, t_on_batch 
        print()
        return np.average(list_loss), \
                          np.average(list_perplexity), \
                          np.average(list_accuracy)

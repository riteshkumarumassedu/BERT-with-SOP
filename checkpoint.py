import numpy as np
import tensorflow as tf
import torch

import yaml
from yaml import Loader

"""
Load the BERT model parameter details from the config file
"""
with open("bert_model_params.yml", 'r') as config_file:
    config_params = yaml.load(config_file, Loader=Loader)

def tensorflow_to_pytorch_params_mapping(checkpoint_file, conversion_table):
    """
    method to map tensorflow checkpoint params to pytorch params
    :param checkpoint_file: tensorflow checkpoint file
    :param conversion_table: tensorflow to pytorch convrsion table
    :return:
    """
    for pyt_param, tf_param_name in conversion_table.items():
        tf_param = tf.train.load_variable(checkpoint_file, tf_param_name)

        # for weight(kernel), we should do transpose
        if tf_param_name.endswith('kernel'):
            tf_param = np.transpose(tf_param)

        assert pyt_param.size() == tf_param.shape, \
            'Dim Mismatch: %s vs %s ; %s' % \
                (tuple(pyt_param.size()), tf_param.shape, tf_param_name)
        
        # assign pytorch tensor from tensorflow param
        pyt_param.data = torch.from_numpy(tf_param)


def load_model_from_checkpoint(model, checkpoint_file):
    """
     Load the pytorch model from checkpoint file
     with mapping the params of tensorflow to pytorch
    """

    # Embedding layer
    e, p = model.embed, 'bert/embeddings/'
    tensorflow_to_pytorch_params_mapping(checkpoint_file, {
        e.tok_embed.weight: p + config_params['word_emb'],
        e.pos_embed.weight: p + config_params['pos_emb'],
        e.seg_embed.weight: p + config_params['token_emb'],
        e.norm.gamma:       p + config_params['gamma'],
        e.norm.beta:        p + config_params['beta']
    })

    # Transformer blocks
    for i in range(len(model.blocks)):
        b, p = model.blocks[i], "bert/encoder/layer_%d/"%i
        tensorflow_to_pytorch_params_mapping(checkpoint_file, {
            b.attn.proj_q.weight:   p + config_params['attn_query_kernal'],
            b.attn.proj_q.bias:     p + config_params['attn_query_bias'],
            b.attn.proj_k.weight:   p + config_params['attn_key_kernal'],
            b.attn.proj_k.bias:     p + config_params['attn_key_bias'],
            b.attn.proj_v.weight:   p + config_params['attn_value_kernal'],
            b.attn.proj_v.bias:     p + config_params['attn_value_bias'],
            b.proj.weight:          p + config_params['attn_dense_kernal'],
            b.proj.bias:            p + config_params['attn_dense_bias'],
            b.pwff.fc1.weight:      p + config_params['imdt_dense_kernal'],
            b.pwff.fc1.bias:        p + config_params['imdt_dense_bias'],
            b.pwff.fc2.weight:      p + config_params['out_dense_kernal'],
            b.pwff.fc2.bias:        p + config_params['out_dense_bias'],
            b.norm1.gamma:          p + config_params['attn_out_gamma'],
            b.norm1.beta:           p + config_params['attn_out_beta'],
            b.norm2.gamma:          p + config_params['out_gamma'],
            b.norm2.beta:           p + config_params['out_beta'],
        })


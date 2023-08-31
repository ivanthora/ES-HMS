# coding=utf-8

import torch
import sys

sys.path.append('../')
import model.clip as clip
import numpy as np
import torch.nn as nn
import torch.nn.init
import torch.backends.cudnn as cudnn
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.cuda.amp import autocast as autocast, GradScaler
from transformers import BertTokenizer, BertModel
# import vmz.models as models

import evaluation
from . import ReRank
import util
from loss import *
from loss import l2norm
from bigfile import BigFile
from common import logger
from generic_utils import Progbar
from model.Attention import *


def _initialize_weights(m):
    """Initialize module weights
    """
    if type(m) == nn.Linear:
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif type(m) == nn.BatchNorm1d:
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)


def to_device_and_float16(x: torch.Tensor):
    x = x.to(device)
    # if float16:
    #     x = x.half()
    return x


def get_attention_layer(attention_type: str, common_space_dim, encoder_num, opt):
    """
    :parm attention_type: 选择的 attention
    :param common_space_dim:
    :param encoder_num:
    :return:
    """
    def cal_params():
        common_space_dim = 2048
        heads = 8
        dim_per_head = common_space_dim // heads
        split_head = True
        net = Multi_head_MyApply_Attention(
            common_space_dim, heads,
            dim_per_head,
            split_head=split_head,
        )
        net = Attention_multi_head_official(
                            common_space_dim,
                            heads)
        net = nn.Linear(2048, 1)
        net_params = sum(p.numel() for p in net.parameters())
        print('Net params: %.8fM' % (net_params / 1000000.0))
        pass
    try:
        attention_layers = {'attention_noAverageMul_Ave': Attention_1(common_space_dim, with_ave=True, mul=False),
                            'attention_noAveNoAverageMul': Attention_1(common_space_dim, with_ave=False, mul=False),
                            'attention_averageMul': Attention_1(common_space_dim, with_ave=True, mul=True),
                            'average_AverageMul_noAve': Attention_1(common_space_dim, with_ave=False, mul=True),
                            'con_attention': nn.Sequential(nn.Conv1d(encoder_num, 1, 1)),
                            'fc_attention': FcAttention(encoder_num),
                            'just_average': JustAverage(),
                            'muti_head_attention': Attention_2(common_space_dim, opt),
                            'attention3': Attention_3(common_space_dim),
                            'muti_head_attention_official': Attention_multi_head_official(
                                common_space_dim,
                                8, opt.multi_head_attention['dropout'],
                                opt.muti_head_attention_official['agg']
                            ),
                            'Attention_MMT': Attention_MMT(
                                common_space_dim,
                                8, opt.multi_head_attention['dropout']),
                            'my_self_attention': Multi_head_MyApply_selfAttention(
                                common_space_dim, opt.multi_head_attention['heads'],
                                # opt.multi_head_attention['embed_dim_qkv'],
                                common_space_dim // opt.multi_head_attention['heads'],
                                opt.multi_head_attention['dropout'],
                                output_type=opt.my_self_attention_output_type,
                                encoder_num=encoder_num,
                                l2norm_each_head=opt.attention_l2norm,
                                opt=opt
                            ),
                            'Multi_head_MyApply_Attention': Multi_head_MyApply_Attention(
                                common_space_dim, opt.multi_head_attention['heads'],
                                common_space_dim // opt.multi_head_attention['heads'],
                                with_ave=opt.attention_param_each_head['with_ave'],
                                mul=opt.attention_param_each_head['mul'],
                                split_head=opt.attention_param_each_head['split_head'],
                                l2norm_each_head=opt.attention_l2norm,
                            ),
                            'Multi_head_MyApply_FusionAttention': Multi_head_MyApply_FusionAttention(
                                common_space_dim, opt.multi_head_attention['heads'],
                                common_space_dim // opt.multi_head_attention['heads'],
                                opt.attention_param_each_head['split_head'],
                            ),
                            'Multi_head_Attention_distinct_fc': Multi_head_Attention_distinct_fc(
                                common_space_dim, opt.multi_head_attention['heads'],
                                common_space_dim // opt.multi_head_attention['heads'],
                                with_ave=opt.attention_param_each_head['with_ave'],
                                mul=opt.attention_param_each_head['mul'],
                                split_head=opt.attention_param_each_head['split_head'],
                            ),
                            'Multi_head_Attention_layer_norm': Multi_head_Attention_layer_norm(
                                common_space_dim, opt.multi_head_attention['heads'],
                                common_space_dim // opt.multi_head_attention['heads'],
                                with_ave=opt.attention_param_each_head['with_ave'],
                                mul=opt.attention_param_each_head['mul'],
                                split_head=opt.attention_param_each_head['split_head'],
                            ),
                            }
    except Exception as e:
        print(e)
    attention_layers = {'attention_noAverageMul_Ave': Attention_1(common_space_dim, with_ave=True, mul=False),
                        'attention_noAveNoAverageMul': Attention_1(common_space_dim, with_ave=False, mul=False),
                        'attention_averageMul': Attention_1(common_space_dim, with_ave=True, mul=True),
                        'average_AverageMul_noAve': Attention_1(common_space_dim, with_ave=False, mul=True),
                        'con_attention': nn.Sequential(nn.Conv1d(encoder_num, 1, 1)),
                        'fc_attention': FcAttention(encoder_num),
                        'just_average': JustAverage(),
                        'muti_head_attention': Attention_2(common_space_dim, opt),
                        'attention3': Attention_3(common_space_dim),
                        'muti_head_attention_official': Attention_multi_head_official(
                            common_space_dim,
                            8, opt.multi_head_attention['dropout'],
                            opt.muti_head_attention_official['agg']
                        ),
                        'Attention_MMT': Attention_MMT(
                            common_space_dim,
                            8, opt.multi_head_attention['dropout']),
                        'my_self_attention': Multi_head_MyApply_selfAttention(
                            common_space_dim, opt.multi_head_attention['heads'],
                            # opt.multi_head_attention['embed_dim_qkv'],
                            common_space_dim // opt.multi_head_attention['heads'],
                            opt.multi_head_attention['dropout'],
                            output_type=opt.my_self_attention_output_type,
                            encoder_num=encoder_num,
                            l2norm_each_head=opt.attention_l2norm,
                            opt=opt
                        ),
                        'Multi_head_MyApply_Attention': Multi_head_MyApply_Attention(
                            common_space_dim, opt.multi_head_attention['heads'],
                            common_space_dim // opt.multi_head_attention['heads'],
                            with_ave=opt.attention_param_each_head['with_ave'],
                            mul=opt.attention_param_each_head['mul'],
                            split_head=opt.attention_param_each_head['split_head'],
                            l2norm_each_head=opt.attention_l2norm,
                        ),
                        'Multi_head_MyApply_FusionAttention': Multi_head_MyApply_FusionAttention(
                            common_space_dim, opt.multi_head_attention['heads'],
                            common_space_dim // opt.multi_head_attention['heads'],
                            opt.attention_param_each_head['split_head'],
                        ),
                        'Multi_head_Attention_distinct_fc': Multi_head_Attention_distinct_fc(
                            common_space_dim, opt.multi_head_attention['heads'],
                            common_space_dim // opt.multi_head_attention['heads'],
                            with_ave=opt.attention_param_each_head['with_ave'],
                            mul=opt.attention_param_each_head['mul'],
                            split_head=opt.attention_param_each_head['split_head'],
                        ),
                        'Multi_head_Attention_layer_norm': Multi_head_Attention_layer_norm(
                            common_space_dim, opt.multi_head_attention['heads'],
                            common_space_dim // opt.multi_head_attention['heads'],
                            with_ave=opt.attention_param_each_head['with_ave'],
                            mul=opt.attention_param_each_head['mul'],
                            split_head=opt.attention_param_each_head['split_head'],
                        ),
                        }

    return attention_layers[attention_type]


class TransformNet(nn.Module):
    """
    fc_layers: (dim_in, dim_out)
    加入 BatchNorm, activation, dropout
    """

    def __init__(self, fc_layers, opt=None, dropout=None, batch_norm=None, activation=None, fc=True):
        super(TransformNet, self).__init__()

        if opt is not None:
            if batch_norm is None:
                batch_norm = opt.batch_norm
            if activation is None:
                activation = opt.activation
            if dropout is None:
                dropout = opt.dropout
        if fc:
            self.fc1 = nn.Linear(fc_layers[0], fc_layers[1])
        else:
            self.fc1 = None
        if batch_norm:
            self.bn1 = nn.BatchNorm1d(fc_layers[1])
        else:
            self.bn1 = None

        if activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            self.activation = None

        if dropout is not None and dropout > 1e-3:
            self.dropout = nn.Dropout(p=dropout)
        else:
            self.dropout = None

        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        self.apply(_initialize_weights)

    def forward(self, input_x):
        """
        一般来说顺序：-> CONV/FC -> ReLu(or other activation) -> Dropout -> BatchNorm -> CONV/FC
        不过有了 bn 一般不用 dropout
        https://stackoverflow.com/questions/39691902/ordering-of-batch-normalization-and-dropout
        """
        features = input_x.to(device)
        if self.fc1 is not None:
            features = self.fc1(features)

        if self.activation is not None:
            features = self.activation(features)

        if self.dropout is not None:
            features = self.dropout(features)

        if self.bn1 is not None:
            features = self.bn1(features)

        return features


class VisTransformNet(TransformNet):
    """
    把拼接的 video_emb 映射到公共空间
    """

    def __init__(self, opt):
        super(VisTransformNet, self).__init__((np.sum(list(opt.vis_fc_layers[0].values())), opt.vis_fc_layers[1]), opt)

    def forward(self, vis_input: dict, txt_emb=None, vis_frame_feat_dict_input=None):
        """
        一般来说顺序：-> CONV/FC -> ReLu(or other activation) -> Dropout -> BatchNorm -> CONV/FC
        不过有了 bn 一般不用 dropout
        https://stackoverflow.com/questions/39691902/ordering-of-batch-normalization-and-dropout
        """
        if type(vis_input) == dict:
            vis_feature = to_device_and_float16(torch.cat(list(vis_input.values()), dim=1))
        else:
            vis_feature = to_device_and_float16(vis_input)
        features = self.fc1(vis_feature)

        if self.activation is not None:
            features = self.activation(features)

        if self.dropout is not None:
            features = self.dropout(features)

        if self.bn1 is not None:
            features = self.bn1(features)

        return features


class TxtEncoder(nn.Module):
    def __init__(self, opt):
        super(TxtEncoder, self).__init__()

    def forward(self, caption_feat_dict, task3=False):
        output = {}
        output['text_features'] = caption_feat_dict['caption']

        return output


class GruTxtEncoder(TxtEncoder):
    def _init_rnn(self, opt):
        if opt.rnn_size == 0:
            return
        self.rnn = nn.GRU(int(opt.we_dim), int(opt.rnn_size), int(opt.rnn_layer), batch_first=True, bidirectional=False)

    def __init__(self, opt):
        super().__init__(opt)
        self.bigru = False
        self.pooling = opt.pooling
        self.rnn_size = opt.rnn_size
        self.t2v_idx = opt.t2v_idx
        self.we = nn.Embedding(len(self.t2v_idx.vocab), opt.we_dim)
        if opt.we_dim == 500:
            self.we.weight = nn.Parameter(opt.we)  # initialize with a pre-trained 500-dim w2v

        self._init_rnn(opt)

    def forward(self, caption_feat_dict, task3=False):
        txt_input = caption_feat_dict['caption']
        batch_size = len(txt_input)

        # caption encoding
        idx_vecs = [self.t2v_idx.encoding(caption) for caption in txt_input]
        lengths = [len(vec) for vec in idx_vecs]

        x = to_device_and_float16(torch.zeros(batch_size, max(lengths)).long())
        for i, vec in enumerate(idx_vecs):
            end = lengths[i]
            x[i, :end] = torch.Tensor(vec)

        # caption embedding
        x = self.we(x)
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)

        # Forward propagate RNN
        out, _ = self.rnn(packed)
        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)

        if self.pooling == 'mean':
            # out = torch.zeros(batch_size, padded[0].shape[-1]).to(device)
            out = x.new_zeros((batch_size, padded[0].shape[-1])).to(device)
            for i, ln in enumerate(lengths):
                out[i] = torch.mean(padded[0][i][:ln], dim=0)
        elif self.pooling == 'last':
            I = torch.LongTensor(lengths).view(-1, 1, 1)
            I = I.expand(batch_size, 1, self.rnn_size) - 1
            I = I.to(device)
            out = torch.gather(padded[0], 1, I).squeeze(1)
        elif self.pooling == 'mean_last':
            # out1 = torch.zeros(batch_size, self.rnn_size).to(device)
            out1 = torch.zeros(batch_size, self.rnn_size).to(device)
            for i, ln in enumerate(lengths):
                out1[i] = torch.mean(padded[0][i][:ln], dim=0)

            I = torch.LongTensor(lengths).view(-1, 1, 1)
            I = I.expand(batch_size, 1, self.rnn_size) - 1
            I = I.to(device)
            out2 = torch.gather(padded[0], 1, I).squeeze(1)
            out = torch.cat((out1, out2), dim=1)

        output = {}
        output['text_features'] = out

        return output


class BiGruTxtEncoder(GruTxtEncoder):
    def _init_rnn(self, opt):
        self.rnn = nn.GRU(opt.we_dim, opt.rnn_size, opt.rnn_layer, batch_first=True, bidirectional=True)

    def __init__(self, opt):
        super().__init__(opt)
        self.bigru = True


class BoWTxtEncoder(TxtEncoder):
    def __init__(self, opt):
        super(BoWTxtEncoder, self).__init__(opt)
        self.t2v_bow = opt.t2v_bow

    def forward(self, caption_feat_dict,task3=False):
        txt_input = caption_feat_dict['caption']
        t = np.empty((len(txt_input), self.t2v_bow.ndims), )
        for i, caption in enumerate(txt_input):
            t[i] = self.t2v_bow.encoding(caption)

        # bow_out = torch.Tensor([self.t2v_bow.encoding(caption) for caption in txt_input]).to(device)
        bow_out = to_device_and_float16(torch.Tensor(t))
        # print(bow_out.shape)
        output = {}
        output['text_features'] = bow_out

        return output


class W2VTxtEncoder(TxtEncoder):
    def __init__(self, opt):
        super(W2VTxtEncoder, self).__init__(opt)
        self.t2v_w2v = opt.t2v_w2v

    def forward(self, caption_feat_dict, task3=False):
        txt_input = caption_feat_dict['caption']
        t = np.empty((len(txt_input), self.t2v_w2v.ndims), )
        for i, caption in enumerate(txt_input):
            t[i] = self.t2v_w2v.encoding(caption)

        w2v_out = to_device_and_float16(torch.Tensor(t))
        output = {}
        output['text_features'] = w2v_out

        return output





class CLIPEncoder(nn.Module):
    """
    CLIP encoder.
    transform image into features.
    """

    def __init__(self, opt=None):
        super().__init__()
        self.opt = opt
        self.Clip_name = opt.text_encoding['CLIP_encoding']['name']
        self.frozen = opt.clip_opt['frozen']
        self.dim = opt.clip_opt['size']
        self.tokenizer = clip.tokenize
        self.simple_tokenizer = clip.simple_tokenizer.SimpleTokenizer()
        self.ClipModel, self.preprocess = clip.load(self.Clip_name, device=device, jit=False)

    def forward(self, caption_feat_dict, vis_origin_frame_tuple=None, task3=False,
                frame_agg_method='mean'):
        """

        :param caption_feat_dict:
        :param vis_origin_frame_tuple: ([sample_frame, 3, 224, 224], ...)
        :param task3:
        :return: (batch_size, dim)
        """
        output = {}
        # For text encoding
        if caption_feat_dict is not None:
            if 'CLIP_encoding' in caption_feat_dict and self.frozen:
                text_features = caption_feat_dict['CLIP_encoding']
            else:
                txt_input = caption_feat_dict['caption']
                text = to_device_and_float16(self.tokenizer(txt_input))
                if self.frozen and (not task3):
                    with torch.no_grad():
                        text_features = self.ClipModel.encode_text(text)
                else:
                    text_features = self.ClipModel.encode_text(text)
            output['text_features'] = text_features

        # For visual encoding
        if vis_origin_frame_tuple is not None:
            batch_size = len(vis_origin_frame_tuple)
            origin_frames = to_device_and_float16(torch.cat(vis_origin_frame_tuple, dim=0))

            if self.frozen:
                with torch.no_grad():
                    frame_features = self.ClipModel.encode_image(origin_frames)
            else:
                frame_features = self.ClipModel.encode_image(origin_frames)
            frame_features = frame_features.reshape((batch_size, -1, self.dim))
            if frame_agg_method == 'mean':
                visual_features = torch.mean(frame_features, dim=1)
            else:
                raise Exception("frame_agg_method is not applied.")

            output['visual_features'] = visual_features

        return output

class NetVLADTxtEncoder(TxtEncoder):
    def __init__(self, opt):
        super().__init__(opt)
        self.t2v_w2v = opt.t2v_w2v
        self.netvlad = NetVLAD(opt.t2v_w2v.ndims,
                               opt.NetVLAD_opt['num_clusters'],
                               opt.NetVLAD_opt['alpha'],
                               )

    def forward(self, caption_feat_dict, task3=False):
        captions = caption_feat_dict['caption']

        w2v_out = []
        for caption in captions:
            x = to_device_and_float16(torch.Tensor(self.t2v_w2v.raw_encoding(caption)))
            w2v_out.append(x)

        netvlad_out = self.netvlad(w2v_out)
        out_dict = {}
        out_dict['text_features'] = netvlad_out
        return out_dict











if __name__ == '__main__':
    global device


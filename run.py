import argparse
import os
import torch
import torch.backends
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
from exp.exp_TimeMosaic import Exp_TimeMosaic
from exp.exp_new import Exp_TimeMosaic as Exp_TimeMosaic_new
from exp.exp_TimeMosaic_MoE import Exp_TimeMosaic_MoE
from exp.exp_Fusion import Exp_Fusion
from exp.exp_MosaicTimer import Exp_MosaicTimer
from exp.exp_Fusion_Text import Exp_Fusion_Text
from exp.exp_TimeFilter import Exp_TimeFilter
from exp.exp_PathFormer import Exp_PathFormer
from exp.exp_DUET import Exp_DUET
from utils.print_args import print_args
import random
import numpy as np
import sys
sys.dont_write_bytecode = True

os.environ['CUDA_VISIBLE_DEVICE']='0,1,2,3,4,5,6,7'

if __name__ == '__main__':
    fix_seed = 2025
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='TimeMosaic')

    # basic config
    parser.add_argument('--task_name', type=str, default='long_term_forecast')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='iTransformer',
                        help='model name, options: [iTransformer]')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='custom', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/electricity/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='electricity.csv', help='data csv file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],
                        help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type')  # cuda or mps
    parser.add_argument('--visualize_attn', action='store_true', help='Whether to visualize attention')


    # forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=0, help='start token length') # no longer needed in inverted Transformers
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')

    # model define
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=7, help='output size') # applicable on arbitrary number of variates in inverted Transformers
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', action='store_false',
                        help='whether to use distilling in encoder, using this argument means not using distilling',
                        default=True)
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--accumulation_steps', type=int, default=1, help='gradient accumulation steps')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--channel', type=str, default='CI', help='CI or CD')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--result_file', type=str, default='result_long_term_forecast.txt', help='result output file')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # iTransformer
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
    # parser.add_argument('--target_root_path', type=str, default='./data/electricity/', help='root path of the data file')
    # parser.add_argument('--target_data_path', type=str, default='electricity.csv', help='data file')

    # TimeMosaic
    parser.add_argument('--fc_dropout', type=float, default=0.1, help='fc_dropout')
    parser.add_argument('--fixed_weight', type=bool, default=False, help='fixed task emb weight')
    parser.add_argument('--adjust_lr', action='store_true', default=True, help='adjust learnring rate')
    parser.add_argument('--num_latent_token', type=int, default=4, help='Number of prompt tokens')
    parser.add_argument('--scale_rate', type=float, default=0.001, help='emb init scale rate')
    parser.add_argument('--patch_len_list', type=str, default='[8,16,32]',
                    help='List of candidate patch lengths for adaptive splitting')
    parser.add_argument('--mask_ratio', type=float, default=0, help='mask_ratio')
    parser.add_argument('--mask_ratio_patch', type=float, default=0, help='mask_ratio_patch')
    parser.add_argument('--pre96', type=int, default=32, help='')
    parser.add_argument('--pre192', type=int, default=64, help='')
    parser.add_argument('--pre336', type=int, default=168, help='')
    parser.add_argument('--pre720', type=int, default=240, help='')
    parser.add_argument('--pre12', type=int, default=6, help='')
    parser.add_argument('--counts', type=int, default=0, help='')
    parser.add_argument('--num_moe_experts', type=int, default=8,
                        help='Number of virtual experts per segment in MoE prompt generator')
    parser.add_argument('--lam_moe', type=float, default=0.001,
                        help='Weight of MoE load balancing loss')
    parser.add_argument('--num_moe_prefix_experts', type=int, default=4,
                        help='Number of virtual experts in MoE prefix generator')
    parser.add_argument('--prefix_len', type=int, default=4,
                        help='Number of prefix tokens per layer for K/V injection')
    parser.add_argument('--lam_prefix_moe', type=float, default=0.001,
                        help='Weight of MoE prefix load balancing loss')
    parser.add_argument('--freq_num', type=int, default=4,
                        help='Number of spectral views (0=disable spectral decomposition)')
    parser.add_argument('--use_prefix', type=int, default=1,
                        help='Enable prefix attention K/V injection (0/1)')
    parser.add_argument('--lam_entropy', type=float, default=0.03,
                        help='Weight of entropy regularization loss')
    parser.add_argument('--lam_diversity', type=float, default=0.008,
                        help='Weight of region diversity loss')
    parser.add_argument('--lam_orthogonal', type=float, default=0.01,
                        help='Weight of orthogonal loss on prompt embeddings')

    # MosaicTimer
    parser.add_argument('--timer_model', type=str, default='thuml/timer-base-84m',
                        help='Pretrained Timer model name (uses hf-mirror.com)')
    parser.add_argument('--timer_num_layers', type=int, default=6,
                        help='Number of top Timer layers to use (frozen)')
    parser.add_argument('--mixer_layers', type=int, default=1,
                        help='Number of bidirectional mixer layers after Timer')
    parser.add_argument('--use_lora', type=int, default=1,
                        help='Enable LoRA fine-tuning of Timer attention (0/1)')
    parser.add_argument('--lora_rank', type=int, default=8,
                        help='LoRA rank for Timer attention adapters')
    parser.add_argument('--lora_alpha', type=int, default=16,
                        help='LoRA scaling factor (alpha/rank)')

    # Fusion-Text (LLM calendar injection)
    parser.add_argument('--use_text_llm', type=int, default=0,
                        help='Enable LLM calendar text injection (0/1)')
    parser.add_argument('--llm_model', type=str, default='prajjwal1/bert-mini',
                        help='LLM model for calendar text (BERT-mini recommended)')

    # SimpleTM
    parser.add_argument('--kernel_size', default=None, help='Specify the length of randomly initialized wavelets (if not None)')
    parser.add_argument('--alpha', type=float, default=1, help='Weight of the inner product score in geometric attention')
    parser.add_argument('--geomattn_dropout', type=float, default=0.5, help='dropout rate of the projection layer in the geometric attention')
    parser.add_argument('--requires_grad', type=bool, default=True, help='Set to True to enable learnable wavelets')
    parser.add_argument('--wv', type=str, default='db1', help='Wavelet filter type. Supports all wavelets available in PyTorch Wavelets')
    parser.add_argument('--m', type=int, default=3, help='Number of levels for the stationary wavelet transform')
    parser.add_argument('--l1_weight', type=float, default=5e-5, help='Weight of L1 loss')
    
    # TimeMixer
    parser.add_argument('--down_sampling_layers', type=int, default=3, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=2, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default='avg',
                        help='down sampling method, only support avg, max, conv')
    parser.add_argument('--channel_independence', type=int, default=1,
                        help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',
                    help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--use_norm', type=int, default=1, help='whether to use normalize; True 1 False 0')

    # WPMixer
    parser.add_argument('--patch_len', type=int, default=16, help='patch length')
    
    # TimesNet
    parser.add_argument('--top_k', type=int, default=3, help='for TimesBlock')
    
    # DUET
    parser.add_argument('--num_experts', type=int, default=3, help="num_experts")
    parser.add_argument('--k', type=int, default=1, help="DUET k")
    parser.add_argument('--hidden_size', type=int, default=256, help="DUET hidden_size")
    parser.add_argument('--CI', action='store_true', help='DUET CI', default=True)
    parser.add_argument('--noisy_gating', action='store_true', help='DUET noisy_gating', default=True)

    # PathFormer
    parser.add_argument('--layer_nums', type=int, default=2)
    parser.add_argument('--num_nodes', type=int, default=4)
    parser.add_argument('--patch_size_list', nargs='+', type=int, default=[32,16,8])
    parser.add_argument('--revin', type=int, default=1, help='whether to apply RevIN')
    parser.add_argument('--num_experts_list', type=list, default=[2, 2, 2])
    parser.add_argument('--residual_connection', type=int, default=0)
    parser.add_argument('--batch_norm', type=int, default=0)
    parser.add_argument('--pct_start', type=float, default=0.4, help='pct_start')
    
    # TimeFilter
    parser.add_argument('--alpha_TimeFilter', type=float, default=0.1, help='KNN for Graph Construction')
    parser.add_argument('--top_p', type=float, default=0.5, help='Dynamic Routing in MoE')
    parser.add_argument('--pos', type=int, choices=[0, 1], default=1, help='Positional Embedding. Set pos to 0 or 1')

    # xPatch
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--ma_type', type=str, default='ema', help='reg, ema, dema')
    parser.add_argument('--alpha_xPatch', type=float, default=0.3, help='alpha')
    parser.add_argument('--beta', type=float, default=0.3, help='beta')
    parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
    
    # TimeMxierPP
    parser.add_argument('--channel_mixing', type=int, default=1, help='channel_mixing')
    
    # zero-shot
    parser.add_argument('--target_root_path', type=str, default='./dataset/ETT-small/', help='root path of the data file')
    parser.add_argument('--target_data_path', type=str, default='ETTh2.csv', help='data file')


    args = parser.parse_args()
    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device('cuda:{}'.format(args.gpu))
        print('Using GPU')
    else:
        if hasattr(torch.backends, "mps"):
            args.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        else:
            args.device = torch.device("cpu")
        print('Using cpu or mps')

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print_args(args)

    if args.task_name == 'long_term_forecast':
        Exp = Exp_Long_Term_Forecast
    elif args.task_name == 'Exp_TimeFilter':
        Exp = Exp_TimeFilter
    elif args.task_name == 'Exp_PathFormer':
        Exp = Exp_PathFormer
    elif args.task_name == 'Exp_DUET':
        Exp = Exp_DUET
    elif args.task_name == 'Exp_TimeMosaic_MoE':
        Exp = Exp_TimeMosaic_MoE
    elif args.task_name == 'Exp_Fusion':
        Exp = Exp_Fusion
    elif args.task_name == 'Exp_MosaicTimer':
        Exp = Exp_MosaicTimer
    elif args.task_name == 'Exp_Fusion_Text':
        Exp = Exp_Fusion_Text
    elif args.task_name == 'TimeMosaic_new':
        Exp = Exp_TimeMosaic_new
    else:
        Exp = Exp_TimeMosaic

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            exp = Exp(args)  # set experiments
            setting = '{}_{}_{}_{}_fixed{}_{}_{}_{}'.format(args.task_name, args.model_id, args.model, args.d_ff, args.fixed_weight, args.learning_rate, args.scale_rate, args.channel)
            if args.task_name == 'Exp_TimeMosaic_MoE':
                setting += '_E{}_lmoe{}'.format(getattr(args, 'num_moe_experts', 8), getattr(args, 'lam_moe', 0.001))
            if args.task_name in ('Exp_Fusion', 'Exp_Fusion_Text', 'Exp_MosaicTimer'):
                setting += '_E{}_lmoe{}_pfxE{}_plen{}_lamPrefixMoe{}'.format(
                    getattr(args, 'num_moe_experts', 8),
                    getattr(args, 'lam_moe', 0.001),
                    getattr(args, 'num_moe_prefix_experts', 4),
                    getattr(args, 'prefix_len', 4),
                    getattr(args, 'lam_prefix_moe', 0.001))
            if args.task_name in ('Exp_Fusion', 'Exp_Fusion_Text') and getattr(args, 'freq_num', 4) != 4:
                setting += '_freq{}'.format(getattr(args, 'freq_num', 4))
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))

            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            if args.gpu_type == 'mps':
                torch.backends.mps.empty_cache()
            elif args.gpu_type == 'cuda':
                torch.cuda.empty_cache()
    # else:
    #     ii = 0
    #     setting = '{}_{}_{}_{}_fixed{}_{}_{}_{}'.format(args.task_name, args.model_id, args.model, args.d_ff, args.fixed_weight, args.learning_rate, args.scale_rate, args.channel)
    #     exp = Exp(args)  # set experiments
    #     print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    #     exp.test(setting, test=1)
    #     if args.gpu_type == 'mps':
    #         torch.backends.mps.empty_cache()
    #     elif args.gpu_type == 'cuda':
    #         torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_{}_fixed{}_{}_{}_{}'.format(args.task_name, args.model_id, args.model,
                                                        args.d_ff, args.fixed_weight, args.learning_rate,
                                                        args.scale_rate, args.channel)
        exp = Exp(args)
        if args.visualize_attn:
            print(f'>>>>>>>visualizing attention : {setting}<<<<<<<<<<<<<<<<<<<')
            exp.visualize_attn(setting)
        else:
            print(f'>>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<')
            # setting = 'long_term_forecast_ETTh1_96_96_PatchTST_2048_fixedFalse_0.0001_0.001_CI'
            exp.test(setting, test=1)

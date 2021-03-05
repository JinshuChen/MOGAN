#!/usr/bin/env python
# coding: utf-8
import re
import logging
import configparser
import colorama
class LogbookFormatter(logging.Formatter):
    def __init__(self, fmt=None, datefmt=None):
        super(LogbookFormatter, self).__init__(fmt=fmt, datefmt=datefmt)
        self._re = re.compile(r"\033\[[0-9]+m")

    def remove_colors_from_msg(self, msg):
        msg = re.sub(self._re, "", msg)
        return msg

    def format(self, record=None):
        record.msg = self.remove_colors_from_msg(record.msg)
        return super(LogbookFormatter, self).format(record)
def get_configs(parser, logger):

    red = colorama.Fore.RED
    green = colorama.Fore.GREEN
    white = colorama.Fore.WHITE
    cyan = colorama.Fore.CYAN
    reset = colorama.Style.RESET_ALL
    bright = colorama.Style.BRIGHT

    logger.info('-' * 80)
    logger.info(green + '[INFO]: reading configs...' + reset)

    int_list = ['seed', 'channels', 'kernel_size', 'G_num_layer', 'D_num_layer', 'scales', 'out_channels', 'D_steps', 'G_steps', 'stride', 'gen_num', 'n_segments', 'compactness', 'start_label']
    float_list = ['weight4style', 'scale_base', 'lr_g', 'lr_d', 'lambda_grad', 'alpha4rec', 'alpha4cos', 'alpha4vgg', 'noise_weight', 'p4flip', 'scale_h', 'scale_w', 'sigma']
    string_list = ['img_input_dir', 'anot_input_dir', 'output_dir', 'suffix']
    otherformat_list = ['iters_list', 'if_padding', 'if_lazy']

    opts, _ = parser.parse_known_args()
    path = opts.config_file
    mode = opts.mode

    config = configparser.ConfigParser()
    config.optionxform = str
    config.read(path)

    logger.info('-' * 80)
    logger.info(bright+white + '[GLOBAL] ' + reset+cyan + '==>' +reset)
    for name in config['GLOBAL']:
        if name not in int_list + string_list + float_list + otherformat_list:
            logger.error(red + '[ERROR]: %s is not included' % name + reset)
        else:
            logger.info('%s : %s' % (name, config['GLOBAL'][name]))
            if name in int_list:
                parser.add_argument('--'+name, type=int, default=config['GLOBAL'].getint(name))
            elif name in string_list:
                parser.add_argument('--'+name, type=str, default=config['GLOBAL'][name])
            elif name in float_list:
                parser.add_argument('--'+name, type=float, default=config['GLOBAL'].getfloat(name))

    if mode != 'f' and mode != 'b':
        logger.error(red + '[ERROR]: wrong mode!' + reset)
    else:
        if mode == 'f':
            logger.info('-' * 80)
            logger.info(bright+white + '[HYPER_F] ' + reset+cyan + '==>' +reset)
            for name in config['HYPER_F']:
                if name not in int_list + string_list + float_list + otherformat_list:
                    logger.error(red + '[ERROR]: %s is not included' % name + reset)
                else:
                    logger.info('%s : %s' % (name, config['HYPER_F'][name]))
                    if name in int_list:
                        parser.add_argument('--'+name, type=int, default=config['HYPER_F'].getint(name))
                    elif name in float_list:
                        parser.add_argument('--'+name, type=float, default=config['HYPER_F'].getfloat(name))
                    elif name in string_list:
                        parser.add_argument('--'+name, type=str, default=config['HYPER_F'][name])
                    elif name in otherformat_list:
                        if name == 'iters_list':
                            parser.add_argument('--'+name, type=list, default=config['HYPER_F'][name].split(','))
                        elif name == 'if_padding':
                            parser.add_argument('--'+name, type=bool, default=config['HYPER_F'].getboolean(name))
                        elif name == 'if_lazy':
                            parser.add_argument('--'+name, type=bool, default=config['HYPER_F'].getboolean(name))
        elif mode == 'b':
            logger.info('-' * 80)
            logger.info(bright+white + '[HYPER_B] ' + reset+cyan + '==>' +reset)
            for name in config['HYPER_B']:
                if name not in int_list + string_list + float_list + otherformat_list:
                    logger.error(red + '[ERROR]: %s is not included' % name + reset)
                else:
                    logger.info('%s : %s' % (name, config['HYPER_B'][name]))
                    if name in int_list:
                        parser.add_argument('--'+name, type=int, default=config['HYPER_B'].getint(name))
                    elif name in float_list:
                        parser.add_argument('--'+name, type=float, default=config['HYPER_B'].getfloat(name))
                    elif name in string_list:
                        parser.add_argument('--'+name, type=str, default=config['HYPER_B'][name])
                    elif name in otherformat_list:
                        if name == 'iters_list':
                            parser.add_argument('--'+name, type=list, default=config['HYPER_B'][name].split(','))
                        elif name == 'if_padding':
                            parser.add_argument('--'+name, type=bool, default=config['HYPER_B'].getboolean(name))
                        elif name == 'if_lazy':
                            parser.add_argument('--'+name, type=bool, default=config['HYPER_B'].getboolean(name))
    logger.info('-' * 80)
    logger.info(green + '[INFO]: configs imported!' + reset)

    return parser
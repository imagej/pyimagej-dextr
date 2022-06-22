import logging

_logger = logging.getLogger(__name__)

_model_name = ""
_pad = 0
_thres = 0
_gpu_id = 0
_device = None

def get_model_name():
    global _model_name
    return _model_name


def set_model_name(name):
    global _model_name
    _logger.debug(f"Setting model name to {name} (previous value: {_model_name})")
    _model_name = name


def get_pad():
    global _pad
    return _pad


def set_pad(value):
    global _pad
    _logger.debug(f"Setting pad to {value} (previous value: {_pad})")
    _pad = value


def get_thres():
    global _thres
    return _thres


def set_thres(value):
    global _thres
    _logger.debug(f"Setting threshold to {value} (previous value: {_thres})")
    _thres = value


def get_gpu_id():
    global _gpu_id
    return _gpu_id


def set_gpu_id(id):
    global _gpu_id
    _logger.debug(f"Setting GPU ID to {id} (previous value: {_gpu_id})")
    _gpu_id = id


def get_device():
    global _device
    return _device


def set_device(device):
    global _device
    _logger.debug(f"Setting device to {device} (previous value: {_device})")
    _device = device

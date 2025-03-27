#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 18:44:20 2021

@author: sc
"""
from .trainer_SGFN import Trainer_SGFN, Trainer_FAN
from .trainer_IMP import Trainer_IMP
from .trainer_FAN import Trainer_FAN

trainer_dict = {
    'sgfn': Trainer_SGFN,
    'sgpn': Trainer_SGFN,
    'imp': Trainer_IMP,
    'jointsg': Trainer_SGFN,
    'fan': Trainer_FAN,
}

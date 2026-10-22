from typing import Dict
import torch
import torch.nn as nn
from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.config import Config
from neuralhydrology.modelzoo.layers import get_head
from neuralhydrology.modelzoo.layers import InputLayer

class RNN(BaseModel):
    module_parts = ["embedding_net", "rnn", "head"]
    
    def __init__(self, cfg: Config):
        super().__init__(cfg=cfg)
        
        self.embedding_net = InputLayer(cfg)
        
        self.rnn = nn.RNN(input_size = self.embedding_net.output_size,
                          hidden_size=cfg.hidden_size,
                          nonlinearity="tanh")
        self.dropout = nn.Dropout(p=cfg.output_dropout)
        self.head = get_head(cfg=cfg, n_in=cfg.hidden_size, n_out=self.output_size)
    def forward(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        x = self.embedding_net(data)
        output, h_n = self.rnn(x)
        
        output = output.transpose(0,1)
        h_n = h_n.transpose(0,1)
        
        pred = {"rnn_output": output, "h_n": h_n}
        pred.update(self.head(self.dropout(output)))
        return pred
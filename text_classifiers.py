import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pytorch_lightning as pl

from .rnn_nn import Embedding, RNN, LSTM


class RNNClassifier(pl.LightningModule):
    def __init__(self, num_embeddings, embedding_dim, hidden_size, use_lstm=True, **additional_kwargs):
        """
        Inputs:
            num_embeddings: size of the vocabulary
            embedding_dim: size of an embedding vector
            hidden_size: hidden_size of the rnn layer
            use_lstm: use LSTM if True, vanilla RNN if false, default=True
        """
        super().__init__()

        # Change this if you edit arguments
        
        hparams = {
            'num_embeddings': num_embeddings,
            'embedding_dim': embedding_dim,
            'hidden_size': hidden_size,
            'use_lstm': use_lstm,
            **additional_kwargs
        }

        # if you do not inherit from lightning module use the following line
        #self.hparams = hparams
        
        # if you inherit from lightning module, comment out the previous line and use the following line
        self.hparams.update(hparams)
        
        ########################################################################
        # TODO: Initialize an RNN network for sentiment classification         #
        # hint: A basic architecture can have an embedding, an rnn             #
        # and an output layer                                                  #
        ########################################################################
    
        self.embedding = nn.Embedding(self.hparams['num_embeddings'], self.hparams['embedding_dim'], padding_idx=0)
        #self.embedding = nn.Embedding(self.hparams['num_embeddings'], self.hparams['embedding_dim'])
        
        self.use_lstm = use_lstm
        
        if self.use_lstm:
            #self.rnn = nn.LSTM(self.hparams['embedding_dim'], self.hparams['hidden_size'], num_layers=2, bidirectional=True, dropout=0.5)
            self.rnn = nn.LSTM(self.hparams['embedding_dim'], self.hparams['hidden_size'], num_layers=2, bidirectional = True)
            #self.rnn = nn.LSTM(self.hparams['embedding_dim'], self.hparams['hidden_size'], num_layers=3, dropout=0.2)
            #self.rnn = nn.LSTM(self.hparams['embedding_dim'], self.hparams['hidden_size'])
        else:
            self.rnn = nn.RNN(self.hparams['embedding_dim'], self.hparams['hidden_size'])
            
        self.fc = nn.Linear(self.hparams['hidden_size'], 1)
        self.fc = nn.Linear(self.hparams['hidden_size'] * 2, 1)
        
        self.sigmoid = nn.Sigmoid()
        
        self.dropout = nn.Dropout(0.3)
        
        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################

    def forward(self, sequence, lengths=None):
        """
        Inputs
            sequence: A long tensor of size (seq_len, batch_size)
            lengths: A long tensor of size batch_size, represents the actual
                sequence length of each element in the batch. If None, sequence
                lengths are identical.
        Outputs:
            output: A 1-D tensor of size (batch_size,) represents the probabilities of being
                positive, i.e. in range (0, 1)
        """
        output = None

        ########################################################################
        # TODO: Apply the forward pass of your network                         #
        # hint: Don't forget to use pack_padded_sequence if lenghts is not None#
        # pack_padded_sequence should be applied to the embedding outputs      #
        ########################################################################

        #out = self.dropout(self.embedding(sequence))
        out = self.embedding(sequence)
        
        if lengths is not None: 
            out = pack_padded_sequence(out, lengths.to('cpu'))
        
        if self.use_lstm:
            out, (hidden, c) = self.rnn(out)
        else:
            out, hidden = self.rnn(out)
            
        if lengths is not None:
            out, _ = pad_packed_sequence(out)
            
        #hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim = 1))
        #output = self.fc(hidden.squeeze(0))
        #output = self.dropout(self.fc(out[0]))
        
        out = self.dropout(out)
        out = self.fc(out[-1]).squeeze()
        output = self.sigmoid(out)
        
        ########################################################################
        #                           END OF YOUR CODE                           #
        ########################################################################
        #return output.flatten()
        return output
        
    def general_step(self, batch, batch_idx, mode):
        text = batch["Text"]
        labels = batch["Label"]
        
        text = text.to('cpu')
        labels = labels.to('cpu')

        # forward pass
        out = self.forward(text)

        # loss
        loss = nn.BSELoss(out, labels)
        
        #predictions
        n_correct += ((out > 0.5) == label).sum().item()
        """
        preds = [] 
        batch_size = len(out)
        for idx in range(batch_size):
            if out[i] > 0.5:
                preds[i] = 1
            else:
                preds[i] = 0
        
        n_correct = (targets == preds).sum()
        """
        return loss, n_correct
    
    def training_step(self, batch, batch_idx):
        text, labels, lengths = batch.values()
        
        text = text.to('cpu')
        labels = labels.to('cpu')
        
        out = self.forward(text)
        f_loss = torch.nn.BCELoss()
    
        
        #out = out.view(-1, 15, 2)
        loss = f_loss(out, labels)
        acc = ((out > 0.5).float()).eq(batch['label']).sum().float() / batch['label'].size(0)
        
        self.log('loss', loss, batch_size=16)
        return {'loss': loss, 'acc': acc}
    
    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['acc'] for x in outputs]).mean()

        self.log('loss', avg_loss)
        self.log('acc', avg_acc)
    
    def validation_step(self, batch, batch_idx):
        f_loss = nn.BCELoss()
        predicted = self.forward(batch['data'])
        loss = f_loss(predicted, batch['label'])
        acc = ((predicted > 0.5).float()).eq(batch['label']).sum().float() / batch['label'].size(0)

        self.log('val_loss', loss, batch_size=16)
        return {'val_loss': loss, 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()

        self.log('val_loss', avg_loss)
        self.log('val_acc', avg_acc)
        
    def test_step(self, batch, batch_idx):
        text, labels, lengths = batch.values()
        
        text = text.to('cpu')
        labels = labels.to('cpu')
        
        out = self.forward(text)
        f_loss = torch.nn.BSELoss()
        
        #out = out.view(-1, 15, 2)
        loss = f_loss(out, labels)
        acc = ((out > 0.5).float()).eq(batch['label']).sum().float() / batch['label'].size(0)
        
        self.log('test_loss', loss, batch_size=16)
        return {'test_loss': loss, 'test_acc': acc}
        #return {'loss': loss}
    
    
    def configure_optimizers(self):
        #return torch.optim.SGD(self.parameters(), lr=1e-3, momentum=0.9)
        return torch.optim.Adam(self.parameters(), lr=3e-4)

"""Class to train encoder decoder neural machine translation network."""

import os

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

from tqdm import tqdm

from nmt.nn.trainer import Trainer
from nmt.encoder_decoder.enc_dec import EncDecNMT


class EncDec(Trainer):

    """Class to train EncDecNMT network."""

    def __init__(self, word_embdim=300, word_embeddings=None, vocab_size=50000,
                 enc_hidden_dim=256, dec_hidden_dim=256, enc_dropout=0,
                 dec_dropout=0, num_layers=1, attention=False, batch_size=64,
                 lr=0.01, num_epochs=100):
        """Initialize EncDec."""
        Trainer.__init__(self)
        self.word_embdim = word_embdim
        self.word_embeddings = word_embeddings
        self.enc_hidden_dim = enc_hidden_dim
        self.dec_hidden_dim = dec_hidden_dim
        self.enc_dropout = enc_dropout
        self.dec_dropout = dec_dropout
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.attention = attention
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs
        self.max_sent_len = None

        # Dataset attributes
        self.metadata_csv = None
        self.train_data = None
        self.val_data = None
        self.test_data = None

        # Model attributes
        self.model = None
        self.optimizer = None
        self.loss_func = None
        self.dict_args = None
        self.nn_epoch = None

        # Save load attributes
        self.save_dir = None
        self.model_dir = None

        # Performance attributes
        self.best_score = 0
        self.best_loss = float('inf')
        self.best_score_train = 0
        self.best_loss_train = float('inf')
        self.val_losses = []
        self.train_losses = []

        if torch.cuda.is_available():
            self.USE_CUDA = True
        else:
            self.USE_CUDA = False

    def _init_nn(self):
        """Initialize the nn model for training."""
        self.dict_args = {'word_embdim': self.word_embdim,
                          'word_embeddings': self.word_embeddings,
                          'max_sent_len': self.max_sent_len,
                          'enc_hidden_dim': self.enc_hidden_dim,
                          'dec_hidden_dim': self.dec_hidden_dim,
                          'enc_dropout': self.enc_dropout,
                          'dec_dropout': self.dec_dropout,
                          'vocab_size': self.vocab_size,
                          'batch_size': self.batch_size,
                          'attention': self.attention,
                          'num_layers': self.num_layers}
        self.model = EncDecNMT(self.dict_args)

        self.loss_func = nn.NLLLoss()
        self.optimizer = optim.Adam(self.model.parameters(), self.lr)

        if self.USE_CUDA:
            self.model = self.model.cuda()
            self.loss_func = self.loss_func.cuda()

    def _train_epoch(self, loader):
        """Train epoch."""
        self.model.train()
        train_loss = 0
        samples_processed = 0

        for batch_samples in tqdm(loader):

            # prepare sample
            # batch_size x seqlen
            X = batch_samples['X']
            X_len = batch_samples['X_len']
            # batch_size x seqlen
            y = batch_samples['y']
            y_len = batch_samples['y_len']

            if self.USE_CUDA:
                X = X.cuda()
                X_len = X_len.cuda()
                y = y.cuda()
                y_len = y_len.cuda()

            # forward pass
            self.model.zero_grad()
            log_probs = self.model(X, X_len, y, y_len)

            # backward pass
            loss = self.loss_func(log_probs, y)
            loss.backward()
            self.optimizer.step()

            # compute train loss
            samples_processed += log_probs.size()[0]
            train_loss += loss.item() * log_probs.size()[0]

        train_loss /= samples_processed

        return samples_processed, train_loss

    def _eval_epoch(self, loader):
        """Eval epoch."""
        self.model.eval()
        eval_loss = 0
        samples_processed = 0

        with torch.no_grad():
            for batch_samples in tqdm(loader):

                # prepare sample
                # batch_size x seqlen
                X = batch_samples['X']
                X_len = batch_samples['X_len']
                # batch_size x seqlen
                y = batch_samples['y']
                y_len = batch_samples['y_len']

                if self.USE_CUDA:
                    X = X.cuda()
                    X_len = X_len.cuda()
                    y = y.cuda()
                    y_len = y_len.cuda()

                # forward pass
                self.model.zero_grad()
                log_probs = self.model(X, X_len, y, y_len)

                # backward pass
                loss = self.loss_func(log_probs, y)

                # compute train loss
                samples_processed += log_probs.size()[0]
                eval_loss += loss.item() * log_probs.size()[0]

            eval_loss /= samples_processed

        return samples_processed, eval_loss

    def fit(self, train_dataset, val_dataset, save_dir):
        """
        Train the NN model.

        Args
            metadata_csv: path to the metadata_csv file.
            save_dir: directory to save nn_model
        """
        # Print settings to output file
        print("Word Embedding Dim {}\n\
               Word Embeddings {}\n\
               Vocabulary Size {}\n\
               Encoder Hidden Dim {}\n\
               Decoder Hidden Dim {}\n\
               Encoder Dropout {}\n\
               Decoder Dropout {}\n\
               Encoder Num Layers {}\n\
               Attention {}\n\
               Batch Size {}\n\
               Learning Rate {}\n\
               Save Dir: {}".format(
                   self.word_embdim, self.word_embeddings, self.vocab_size,
                   self.enc_hidden_dim, self.dec_hidden_dim, self.enc_dropout,
                   self.dec_dropout, self.num_layers, self.attention,
                   self.batch_size, self.lr, save_dir))

        self.model_dir = save_dir
        self.train_data = train_dataset
        self.val_data = val_dataset
        self.max_sent_len = max(
            self.train_data.max_sent_len, self.val_data.max_sent_len)

        train_loader = DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True,
            num_workers=4)
        val_loader = DataLoader(
            self.val_data, batch_size=self.batch_size, shuffle=True,
            num_workers=4)

        self._init_nn()

        # init training variables
        train_loss = 0
        samples_processed = 0

        # train loop
        for epoch in range(self.num_epochs + 1):
            self.nn_epoch = epoch
            if epoch > 0:
                print("Initializing train epoch...")
                sp, train_loss = self._train_epoch(train_loader)
                samples_processed += sp

                # report
                print("Epoch: [{}/{}]\tSamples: [{}/{}]\tTrain Loss:{}".format(
                    epoch, self.num_epochs, samples_processed,
                    len(self.train_data)*self.num_epochs, train_loss))

            if epoch % 1 == 0:
                # compute loss
                print("Initializing val epoch...")
                _, val_loss = self._eval_epoch(val_loader)

                self.val_losses += [val_loss]
                # val_score = self.score(val_loader)
                val_score = 0

                if val_score > self.best_score:
                    self.best_score = val_score
                    self.best_loss = val_loss

                    train_score = self.score(train_loader)
                    self.best_score_train = train_score
                    self.best_loss_train = train_loss

                    self.save(save_dir)
                else:
                    train_score = None

                # report
                print("Epoch: [{}/{}]\tSamples: [{}/{}]\tTrain Loss: \
                {}\tValidation Loss: {}".format(
                    epoch, self.num_epochs, samples_processed,
                    len(self.train_data)*self.num_epochs, train_loss,
                    val_loss))

    def predict(self, loader):
        """Predict input."""
        self.model.eval()
        index_sequences = []
        truth = []

        with torch.no_grad():
            for batch_samples in tqdm(loader):
                # prepare sample
                X = batch_samples['X']
                X_len = batch_samples['X_len']
                # batch_size * 1
                y = batch_samples['y']

                if self.USE_CUDA:
                    X = X.cuda()
                    y = y.cuda()

                # forward pass
                self.model.zero_grad()
                index_seq = self.model(X, X_len, inference=True)
                index_sequences += [index_seq]
                truth += [y]

            index_sequences = torch.cat(index_sequences, dim=0)
            truth = torch.cat(truth, dim=0)

        return index_sequences, truth

    def score(self, loader, type='perplexity'):
        """Score model."""
        index_sequences, truth = self.predict(loader)
        if type == 'perplexity':
            # score = perplexity_score(truth, index_sequences)
            raise NotImplementedError("Not implemented yet.")
        elif type == 'bleu':
            # score = bleu_score(truth, index_sequences)
            raise NotImplementedError("Not implemented yet.")
        else:
            raise ValueError("Unknown score type!")

        return score

    def save(self, models_dir=None):
        """
        Save model.

        Args
            models_dir: path to directory for saving NN models.
        """
        if (self.model is not None) and (models_dir is not None):

            model_dir = "ENCDEC_wed_{}_we_{}_vs_{}_ehd_{}_dhd_{}_ed_{}_dd_{}\
            _nl_{}_at_{}_lr_{}".\
                format(self.word_embdim, self.word_embeddings, self.vocab_size,
                       self.enc_hiddem_dim, self.dec_hidden_dim,
                       self.enc_dropout, self.dec_dropout, self.num_layers,
                       self.attention, self.lr)

            if not os.path.isdir(os.path.join(models_dir, model_dir)):
                os.makedirs(os.path.join(models_dir, model_dir))

            filename = "epoch_{}".format(self.nn_epoch) + '.pth'
            fileloc = os.path.join(models_dir, model_dir, filename)
            with open(fileloc, 'wb') as file:
                torch.save({'state_dict': self.model.state_dict(),
                            'dcue_dict': self.__dict__}, file)

    def load(self, model_dir, epoch):
        """
        Load a previously trained model.

        Args
            model_dir: directory where models are saved.
            epoch: epoch of model to load.
        """
        epoch_file = "epoch_"+str(epoch)+".pth"
        model_file = os.path.join(model_dir, epoch_file)
        with open(model_file, 'rb') as model_dict:
            checkpoint = torch.load(model_dict)

        for (k, v) in checkpoint['dcue_dict'].items():
            setattr(self, k, v)

        self._init_nn()
        self.model.load_state_dict(checkpoint['state_dict'])

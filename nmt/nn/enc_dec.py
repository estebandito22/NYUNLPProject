"""Class to train encoder decoder neural machine translation network."""

import os
import multiprocessing
import copy

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset

from tqdm import tqdm

from nmt.nn.trainer import Trainer
from nmt.encoder_decoder.enc_dec import EncDecNMT
from nmt.evaluators.sacrebleu import BleuEvaluator


class EncDec(Trainer):

    """Class to train EncDecNMT network."""

    def __init__(self, word_embdim=300, word_embeddings=(None, None),
                 enc_vocab_size=50000, dec_vocab_size=50000, bos_idx=2,
                 eos_idx=3, pad_idx=1, enc_hidden_dim=256, dec_hidden_dim=256,
                 enc_num_layers=1, enc_dropout=0.0, attention=False,
                 batch_size=64, lr=0.01, weight_decay=0.0, num_epochs=100):
        """Initialize EncDec."""
        Trainer.__init__(self)
        self.word_embdim = word_embdim
        self.word_embeddings = word_embeddings
        self.enc_vocab_size = enc_vocab_size
        self.dec_vocab_size = dec_vocab_size
        self.bos_idx = bos_idx
        self.eos_idx = eos_idx
        self.pad_idx = pad_idx
        self.enc_hidden_dim = enc_hidden_dim
        self.dec_hidden_dim = dec_hidden_dim
        self.enc_num_layers = enc_num_layers
        self.enc_dropout = enc_dropout
        self.batch_size = batch_size
        self.attention = attention
        self.batch_size = batch_size
        self.lr = lr
        self.weight_decay = weight_decay
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
        self.nn_epoch = 0

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

        self.bleu_scorer = BleuEvaluator()

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
                          'enc_num_layers': self.enc_num_layers,
                          'enc_dropout': self.enc_dropout,
                          'enc_vocab_size': self.enc_vocab_size,
                          'dec_vocab_size': self.dec_vocab_size,
                          'bos_idx': self.bos_idx,
                          'eos_idx': self.eos_idx,
                          'pad_idx': self.pad_idx,
                          'batch_size': self.batch_size,
                          'attention': self.attention}
        self.model = EncDecNMT(self.dict_args)

        self.loss_func = nn.NLLLoss(ignore_index=self.pad_idx)
        self.optimizer = optim.Adam(self.model.parameters(), self.lr,
                                    weight_decay=self.weight_decay)

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
            t = batch_samples['y'][:, :-1]
            t_len = batch_samples['y_len'] - 1
            # batch_size x seqlen
            y = batch_samples['y'][:, 1:]
            y_len = batch_samples['y_len'] - 1

            if self.USE_CUDA:
                X = X.cuda()
                X_len = X_len.cuda()
                t = t.cuda()
                t_len = t_len.cuda()
                y = y.cuda()
                y_len = y_len.cuda()

            # forward pass
            self.model.zero_grad()
            log_probs = self.model(X, X_len, t, t_len)

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
                t = batch_samples['y'][:, :-1]
                t_len = batch_samples['y_len'] - 1
                # batch_size x seqlen
                y = batch_samples['y'][:, 1:]
                y_len = batch_samples['y_len'] - 1

                if self.USE_CUDA:
                    X = X.cuda()
                    X_len = X_len.cuda()
                    t = t.cuda()
                    t_len = t_len.cuda()
                    y = y.cuda()
                    y_len = y_len.cuda()

                # forward pass
                self.model.zero_grad()
                log_probs = self.model(X, X_len, t, t_len)

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
            train_dataset: PyTorch Dataset object.
            val_dataset: PyTorch Dataset object.
            save_dir: directory to save model.
        """
        # Print settings to output file
        print("Settings:\n\
               Word Embedding Dim: {}\n\
               Word Embeddings: {}\n\
               Enc Vocabulary Size: {}\n\
               Dec Vocabulary Size: {}\n\
               Encoder Hidden Dim: {}\n\
               Decoder Hidden Dim: {}\n\
               Encoder Num Layers: {}\n\
               Encoder Dropout: {}\n\
               Attention: {}\n\
               Batch Size: {}\n\
               Learning Rate: {}\n\
               Weight Decay: {}\n\
               Save Dir: {}".format(
                   self.word_embdim,
                   False if self.word_embeddings[0] is None else True,
                   self.enc_vocab_size, self.dec_vocab_size,
                   self.enc_hidden_dim, self.dec_hidden_dim,
                   self.enc_num_layers, self.enc_dropout,
                   self.attention, self.batch_size, self.lr, self.weight_decay,
                   save_dir),
              flush=True)

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

        val_score_loader = DataLoader(
            self.val_data, batch_size=1, shuffle=False,
            num_workers=1)

        # train_score_loader = DataLoader(
        #     self.train_data, batch_size=1, shuffle=False,
        #     num_workers=1)

        self._init_nn()

        # init training variables
        train_loss = 0
        samples_processed = 0

        # train loop
        training = True
        while training:

            train_loaders = self._batch_loaders(self.train_data, k=1)
            val_loaders = [val_loader] * len(train_loaders)

            loaders = zip(train_loaders, val_loaders)
            for train_loader, val_loader in loaders:

                if self.nn_epoch > 0:
                    print("Initializing train epoch...", flush=True)
                    sp, train_loss = self._train_epoch(train_loader)
                    samples_processed += sp
                    self.train_losses += [train_loss]

                if self.nn_epoch % 1 == 0:
                    # compute loss
                    print("Initializing val epoch...", flush=True)
                    _, val_loss = self._eval_epoch(val_loader)

                    self.val_losses += [val_loss]
                    val_score = self.score(val_score_loader)

                    if val_score > self.best_score:
                        self.best_score = val_score
                        self.best_loss = val_loss

                        self.best_loss_train = train_loss

                        self.save(save_dir)

                    # report
                    print("Epoch: [{}/{}]\tSamples: [{}/{}]\tTrain Loss: {}\tValidation Loss: {}\t Validation BLEU: {}".
                          format(self.nn_epoch, self.num_epochs,
                                 samples_processed,
                                 len(self.train_data)*self.num_epochs,
                                 train_loss, val_loss, val_score), flush=True)

                if self.nn_epoch >= self.num_epochs:
                    training = False
                else:
                    self.nn_epoch += 1

    def predict(self, loader):
        """Predict input."""
        self.model.eval()
        preds = []
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
                    X_len = X_len.cuda()
                    y = y.cuda()

                # forward pass
                self.model.zero_grad()
                index_seq = self.model(X, X_len, inference=True)

                # convert to tokens
                preds += [' '.join([loader.dataset.y_id2token[idx]
                          for idx in index_seq])]
                # removes bos, eos and pad
                truth += [' '.join([loader.dataset.y_id2token[idx]
                          for idx in y.cpu().tolist()[0] if idx != 0][1:-1])]

        return preds, truth

    def score(self, loader, type='bleu'):
        """Score model."""
        preds, truth = self.predict(loader)

        if type == 'perplexity':
            # score = perplexity_score(truth, index_sequences)
            raise NotImplementedError("Not implemented yet.")
        elif type == 'bleu':
            bleu_tuple = self.bleu_scorer.corpus_bleu(preds, [truth])
            score = bleu_tuple[0]
        else:
            raise ValueError("Unknown score type!")

        return score

    def _batch_loaders(self, dataset, k=None):
        batches = dataset.randomize_samples(k)
        loaders = []
        for subset_batch_indexes in batches:
            subset = Subset(dataset, subset_batch_indexes)
            loader = DataLoader(
                subset, batch_size=self.batch_size, shuffle=True,
                num_workers=multiprocessing.cpu_count())
            loaders += [loader]
        return loaders

    def save(self, models_dir=None):
        """
        Save model.

        Args
            models_dir: path to directory for saving NN models.
        """
        if (self.model is not None) and (models_dir is not None):

            model_dir = "ENCDEC_wed_{}_we_{}_evs_{}_dvs_{}_ehd_{}_dhd_{}_enl_{}_edo_{}_at_{}_lr_{}_wd_{}".\
                format(self.word_embdim, bool(self.word_embeddings),
                       self.enc_vocab_size, self.dec_vocab_size,
                       self.enc_hidden_dim, self.dec_hidden_dim,
                       self.enc_num_layers, self.enc_dropout,
                       self.attention, self.lr, self.weight_decay)

            if not os.path.isdir(os.path.join(models_dir, model_dir)):
                os.makedirs(os.path.join(models_dir, model_dir))

            filename = "epoch_{}".format(self.nn_epoch) + '.pth'
            fileloc = os.path.join(models_dir, model_dir, filename)
            with open(fileloc, 'wb') as file:
                attr_dict = copy.deepcopy(self.__dict__)
                attr_dict.pop('bleu_scorer', None)
                torch.save({'state_dict': self.model.state_dict(),
                            'dcue_dict': attr_dict}, file)

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

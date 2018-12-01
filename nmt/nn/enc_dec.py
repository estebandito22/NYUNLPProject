"""Class to train encoder decoder neural machine translation network."""

import os
import multiprocessing
import copy
import random

import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.lr_scheduler import StepLR

from tqdm import tqdm

from nmt.nn.trainer import Trainer
from nmt.encoder_decoder.enc_dec import EncDecNMT
from nmt.evaluators.sacrebleu import BleuEvaluator


class EncDec(Trainer):

    """Class to train EncDecNMT network."""

    def __init__(self, word_embdim=300, word_embeddings=(None, None),
                 enc_vocab_size=50000, dec_vocab_size=50000, bos_idx=2,
                 eos_idx=3, pad_idx=1, enc_hidden_dim=256, dec_hidden_dim=256,
                 enc_num_layers=1, dec_num_layers=1, enc_dropout=0.0, kernel_size=0,
                 dec_dropout=0.0, dropout_in=0.1, dropout_out=0.1,
                 attention=False, beam_width=1, batch_size=64,
                 optimize='sgd', lr=0.25, weight_decay=0.0, clip_grad=0.1,
                 lr_scheduler='fixed', min_lr=1e-4, num_epochs=100,
                 model_type='lstm', tf_ratio=1):
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
        self.dec_num_layers = dec_num_layers
        self.enc_dropout = enc_dropout
        self.dec_dropout = dec_dropout
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out
        self.kernel_size = kernel_size
        self.batch_size = batch_size
        self.attention = attention
        self.beam_width = beam_width
        self.batch_size = batch_size
        self.optimize = optimize
        self.lr = lr
        self.weight_decay = weight_decay
        self.clip_grad = clip_grad
        self.lr_scheduler = lr_scheduler
        self.min_lr = min_lr
        self.num_epochs = num_epochs
        self.model_type = model_type
        self.tf_ratio = tf_ratio
        self.max_sent_len = None
        self.reversed_in = None

        assert optimize in ['adam', 'sgd'], "optimize must be adam or sgd!"

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
        self.scheduler = None

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
                          'dec_num_layers': self.dec_num_layers,
                          'enc_dropout': self.enc_dropout,
                          'dec_dropout': self.dec_dropout,
                          'dropout_in': self.dropout_in,
                          'dropout_out': self.dropout_out,
                          'kernel_size': self.kernel_size,
                          'enc_vocab_size': self.enc_vocab_size,
                          'dec_vocab_size': self.dec_vocab_size,
                          'bos_idx': self.bos_idx,
                          'eos_idx': self.eos_idx,
                          'pad_idx': self.pad_idx,
                          'batch_size': self.batch_size,
                          'attention': self.attention,
                          'beam_width': self.beam_width,
                          'model_type': self.model_type,
                          'tf_ratio': self.tf_ratio}
        self.model = EncDecNMT(self.dict_args)

        self.loss_func = nn.NLLLoss(ignore_index=self.pad_idx)

        if self.optimize == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), self.lr,
                                        weight_decay=self.weight_decay)
        elif self.optimize == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), self.lr,
                                       weight_decay=self.weight_decay,
                                       nesterov=True, momentum=0.99)

        if self.lr_scheduler == 'reduce_on_plateau':
            self.scheduler = ReduceLROnPlateau(
                self.optimizer, patience=0)
        elif self.lr_scheduler == 'multi_step_lr':
            self.scheduler = MultiStepLR(
                self.optimizer, list(range(8, 20, 1)), 0.5)
        elif self.lr_scheduler == 'fixed':
            self.scheduler = StepLR(
                self.optimizer, step_size=1, gamma=0.1)

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
            if self.clip_grad > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.clip_grad)
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
               Reversed Inputs: {}\n\
               Encoder Hidden Dim: {}\n\
               Decoder Hidden Dim: {}\n\
               Encoder Num Layers: {}\n\
               Decoder Num Layers: {}\n\
               Encoder Dropout: {}\n\
               Decoder Dropout: {}\n\
               Dropout Inputs: {}\n\
               Dropout Outputs: {}\n\
               Attention: {}\n\
               Beam Width: {}\n\
               Batch Size: {}\n\
               Optimizer: {}\n\
               Learning Rate: {}\n\
               Weight Decay: {}\n\
               Clip Grad: {}\n\
               LR Scheduler: {}\n\
               Min LR: {}\n\
               Model Type: {}\n\
               Teacher Forcing Ratio: {}\n\
               Save Dir: {}".format(
                   self.word_embdim,
                   False if self.word_embeddings[0] is None else True,
                   self.enc_vocab_size, self.dec_vocab_size,
                   train_dataset.reversed_in,
                   self.enc_hidden_dim, self.dec_hidden_dim,
                   self.enc_num_layers, self.dec_num_layers,
                   self.enc_dropout, self.dec_dropout,
                   self.dropout_in, self.dropout_out,
                   self.attention, self.beam_width, self.batch_size,
                   self.optimize, self.lr, self.weight_decay, self.clip_grad,
                   self.lr_scheduler, self.min_lr, self.model_type,
                   self.tf_ratio, save_dir), flush=True)

        # initialize dataset attributes
        self.model_dir = save_dir
        self.train_data = train_dataset
        self.val_data = val_dataset
        self.max_sent_len = max(
            self.train_data.max_sent_len, self.val_data.max_sent_len)
        # check for reversed_in setting
        assert self.train_data.reversed_in == self.val_data.reversed_in, \
            "Training data and validation data must have same reversed_in set."
        self.reversed_in = self.train_data.reversed_in

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
                    val_score, correct, total, precisions, brevity_penalty, \
                        sys_len, ref_len = self.score(val_score_loader)

                    if self.scheduler.get_lr()[-1] > self.min_lr:
                        if self.lr_scheduler == 'reduce_on_plateau':
                            self.scheduler.step(val_loss)
                        else:
                            self.scheduler.step()

                    if val_score > self.best_score:
                        self.best_score = val_score
                        self.best_loss = val_loss

                        self.best_loss_train = train_loss

                        self.save(save_dir)

                    # report
                    print("Epoch: [{}/{}]\tSamples: [{}/{}]\tTrain Loss: {}\tValidation Loss: {}\t Validation BLEU: {}\tCorrect: {}\tTotal: {}\tPrecisions: {}\tBrevity Penalty: {}\tSys Len: {}\tRef Len: {}".
                          format(self.nn_epoch, self.num_epochs,
                                 samples_processed,
                                 len(self.train_data)*self.num_epochs,
                                 train_loss, val_loss, val_score, correct,
                                 total, precisions, brevity_penalty,
                                 sys_len, ref_len), flush=True)

                if self.nn_epoch >= self.num_epochs:
                    training = False
                else:
                    self.nn_epoch += 1

    def predict(self, loader):
        """Predict input."""
        self.model.eval()
        preds = []
        truth = []
        attn = []

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
                index_seq, attentions = self.model(X, X_len, inference=True)

                # convert to tokens
                preds += [' '.join([loader.dataset.y_id2token[idx]
                          for idx in index_seq])]
                # removes bos, eos and pad
                truth += [' '.join([loader.dataset.y_id2token[idx]
                          for idx in y.cpu().tolist()[0] if idx != 0][1:-1])]
                # list of all attention matricies
                attn += [attentions]

        return preds, truth, attn

    def score(self, loader, scorer='bleu'):
        """Score model."""
        preds, truth, _ = self.predict(loader)
        idx = random.randint(0, len(loader.dataset))
        print("Preds\n", preds[idx])
        print("Truth\n", truth[idx])

        if scorer == 'perplexity':
            # score = perplexity_score(truth, index_sequences)
            raise NotImplementedError("Not implemented yet.")
        elif scorer == 'bleu':
            bleu_tuple = self.bleu_scorer.corpus_bleu(preds, [truth])
        else:
            raise ValueError("Unknown score type!")

        return bleu_tuple

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

            model_dir = "ENCDEC_wed_{}_we_{}_evs_{}_dvs_{}_ri_{}_ehd_{}_dhd_{}_enl_{}_dnl_{}_edo_{}_ddo_{}_di_{}_do_{}_at_{}_bw_{}_op_{}_lr_{}_wd_{}_cg_{}_ls_{}_ml_{}_mt_{}_tf_{}".\
                format(self.word_embdim, bool(self.word_embeddings),
                       self.enc_vocab_size, self.dec_vocab_size,
                       self.reversed_in, self.enc_hidden_dim,
                       self.dec_hidden_dim, self.enc_num_layers,
                       self.dec_num_layers, self.enc_dropout, self.dec_dropout,
                       self.dropout_in, self.dropout_out,
                       self.attention, self.beam_width, self.optimize, self.lr,
                       self.weight_decay, self.clip_grad,
                       self.lr_scheduler, self.min_lr,
                       self.model_type, self.tf_ratio)

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

import tensorflow as tf
import sys
from tqdm import tqdm
import numpy as np

class Trainer:
    def __init__(
        self,
        generator,
        model,
        clip = None,
    ):
        self.generator = generator
        self.model = model
        self.epoch_n = 1
        self.clip = clip
    
    def _step(self, batch):
        features, target = batch
        with tf.GradientTape() as tape:
            output = self.model(features)
            loss = self.model.loss(output, target)
            #put regularization here if necessary
        grads = tape.gradient(loss, self.model.trainable_variables)
        if self.clip:
            grads, _ = tf.clip_by_global_norm(grads, self.clip)
        self.model.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    def train(self, n_epochs, verbose=True):
        end_at = self.epoch_n + n_epochs - 1
        for epoch in range(n_epochs):
            if verbose:
                progress = tqdm(
                    total = self.generator.final_batch,
                    desc = f'Epoch {self.epoch_n}/{end_at}',
                    unit = 'Batch'
                )
            losses = []
            while self.generator.has_next_batch:
                batch = self.generator.load_batch()
                loss = self._step(batch)
                losses.append(np.average(loss))
                if verbose:
                    progress.set_postfix(loss=np.average(losses))
                    progress.update(1)
            if verbose:
                progress.close()
            self.epoch_n += 1
            #reset and reshuffle the data batches
            self.generator.initialize()

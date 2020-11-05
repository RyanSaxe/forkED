import tensorflow as tf
import sys
from tqdm import tqdm

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
    
    def _step(self, batch):
        features, target = batch
        with tf.GradientTape() as tape:
            output = model(features)
            loss = model.loss(output, target)
            #put regularization here if necessary
        grads = tape.gradient(loss, model.trainable_variables)
        if self.clip:
            grads, _ = tf.clip_by_global_norm(grads, self.clip)
        model.optimizer.apply_gradients(zip(grads, model.trainable_variables))
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
            while self.generator.has_next_batch:
                batch = self.generator.load_batch()
                loss = self._step(batch)
                if verbose:
                    progress.set_postfix(loss=np.average(loss))
                    progress.update(1)
            self.next_epoch_n += 1

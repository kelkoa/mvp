from DataLoader import DataLoader
from Model import MVP
from tensorflow import keras
import tensorflow as tf
import numpy as np
import argparse
import time
import csv
import os

if __name__=='__main__':
    # For replicability
    seed_value = 0
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    keras.utils.set_random_seed(seed_value)

    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', type=bool, default=0)
    parser.add_argument('--load', type=str)
    args = parser.parse_args()
    # print(args)

    dataloader = DataLoader()
    prices = keras.Input(shape=(dataloader.long_seq, 3), name='input_price')
    tweets = keras.Input(shape=(dataloader.long_seq, dataloader.max_daily_tweets, dataloader.embed_size), name='input_text')
    labels = keras.Input(shape=(3,), name='input_labels')

    mirrored_strategy = tf.distribute.MirroredStrategy()
    with mirrored_strategy.scope():
        mvp = MVP()([prices,tweets,labels])
        model = keras.Model(inputs=[prices,tweets,labels], outputs=mvp)

        # model.summary()
        adam_opt = keras.optimizers.Adam(learning_rate=5e-4, decay=5e-4)
        model.compile(optimizer=adam_opt)

        if args.load != None:
            # Load weights
            model.load_weights(args.load)

    if args.eval == False:
        # Save checkpoints
        default_model_name = 'savefile/model'
        count = 1
        model_name = default_model_name + f'{count:05d}'

        while os.path.exists(model_name):
            model_name = default_model_name + f'{count:05d}'
            count += 1

        checkpoint_path = model_name + '/cp-{epoch:04d}.ckpt'
        checkpoint_dir = os.path.dirname(checkpoint_path)
        cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, verbose=1, save_weights_only=True, period=1)

        t = time.time()

        history = model.fit(
            dataloader.get_batch('train'),
            steps_per_epoch=29, # train_length // batch_size = 29
            epochs=500,
            callbacks=[cp_callback],
            verbose=2,
            validation_data=dataloader.get_batch('valid'),
            validation_steps=3, # val_length // batch_size = 3
        )

        print("Time taken:", time.time() - t)

        # early-stopping on MVIR
        best_epoch = np.argmax(history.history['val_mvir']) + 1
        print('best epoch:', best_epoch)
        model.load_weights(model_name + f'/cp-{best_epoch:04d}.ckpt')

    print("Evaluate on test data")
    results = model.predict(dataloader.get_batch('test'))

    print("\n\nDone.")

    with open("output.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(np.array(results).T)

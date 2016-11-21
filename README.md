# EmojiRecognizer
A fun project to manually add noise to emojis, and use tensorflow NN for classification


To run:

```
cd EmojiRecognizer
virtualenv env
source env/bin/activate
pip install -r requirements.txt
python generate_noissy_data.py # generates a bunch of augmented 28x28 emojis from under data/
python train.py                # train on the generated emojis and report accuracy
```

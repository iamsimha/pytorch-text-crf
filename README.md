# PyTorch Text CRF
This package contains a simple wrapper for using conditional random fields(CRF). This code is based on the excellent Allen NLP implementation of CRF.

## Installation
```
pip install pytorch-text-crf
```

## Usage

```python
from crf.crf import ConditionalRandomField

# Initilization
crf = ConditionalRandomField(n_tags,
                            label_encoding="BIO",
                            idx2tag={0:"B-GEO", 1:"I-GEO", 2:"0"} # Index to tag mapping
                            )
# Likelihood estimation
log_likelihood = crf(logits, tags, mask)

# Decoding
best_tag_sequence = crf.best_viterbi_tag(logits, mask)
top_5_viterbi_tags = crf.viterbi_tags(logits, mask, top_k=5)
```
### LSTM CRF Implementation
Refer to https://github.com/iamsimha/pytorch-text-crf/blob/master/examples/pos_tagging/train.ipynb for a complete working implementation.
``` python
from crf.crf import ConditionalRandomField

class LSTMCRF:
    """
    An Example implementation for using a CRF model on top of LSTM.
    """
    def __init__(self):
        ...
        ...
        # Initilize the conditional CRF model
        self.crf = ConditionalRandomField(
            n_class, # Number of tags
            label_encoding="BIO", # Label encoding format
            idx2tag=idx2tag # Dict mapping index to a tag
        )

    def forward(self, inputs, tags):
        logits = self.lstm(inputs) # logits dim:(batch_size, seq_length, num_tags)
        mask = inputs != "<pad token>" # mask for ignoring pad tokens. mask dim: (batch_size, seq_length)
        log_likelihood = self.crf(logits, tags, mask)
        loss = -log_likelihood # Log likelihood is not normalized (It is not divided by the batch size).

        # To obtain the best sequence using viterbi decoding
        best_tag_sequence = self.crf.best_viterbi_tag(logits, mask)

        # To obtain output similar to the lstm prediction we can use the below code
        class_probabilities = out * 0.0
        for i, instance_tags in enumerate(best_tag_sequence):
            for j, tag_id in enumerate(instance_tags[0][0]):
                class_probabilities[i, j, int(tag_id)] = 1
        return {"loss": loss, "class_probabilities": class_probabilities} 

 # Training
 lstm_crf = LSTMCRF()
 output = lstm_crf(sentences, tags)
 loss = output["loss"]
 loss.backward()
 optimizer.step()
``` 

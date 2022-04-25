import torch

class sentimentScore:
    def __init__(self, review, tokenizer, model):
        self.review = review
        self.tokenizer = tokenizer
        self.model = model


    def sentiment_score(self):
        """
        It will use pre-trained model for give sentiment scorse
        :param review:
        :return: sentiment scorse will between 1-5
        """
        try:

            tokens = self.tokenizer.encode(self.review, return_tensors='pt')
            result = self.model(tokens)

            return int(torch.argmax(result.logits)) + 1

        except Exception as e:
            raise e

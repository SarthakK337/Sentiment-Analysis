import string
import re

class preprocessText:
    def __init__(self, text):
        self.text = text

    def preprocess_text(self):
        """
        Use for cleaning text
        :return: it will clean text
        """

        try:
            # Removing numbers
            text = self.text
            text = str(text)
            text = [i if i.isalpha() else i if i.isalnum() == False else "" for i in text.split()]
            text = " ".join(text)

            # Removing punctuations
            text = text.translate(str.maketrans("", "", string.punctuation))

            # Remove user reference "@" and "#" from text
            text = re.sub(r'\@\w+|\#', "", text)

            return text

        except Exception as e:
            raise e

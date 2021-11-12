import bert_retrain


class Test_Bert_retrain_Normalize_text:
    def test_normalize_text_1(self):
        result = bert_retrain.normalize_text("Hello, world!")

    def test_normalize_text_2(self):
        result = bert_retrain.normalize_text("This is a Text")

    def test_normalize_text_3(self):
        result = bert_retrain.normalize_text("Foo bar")

    def test_normalize_text_4(self):
        result = bert_retrain.normalize_text("foo bar")

    def test_normalize_text_5(self):
        result = bert_retrain.normalize_text("")

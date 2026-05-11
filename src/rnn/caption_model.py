import numpy as np
from .layers import EmbeddingLayer, DenseLayer
from .rnn    import SimpleRNNCell, SimpleRNNDecoder
from .lstm   import LSTMCell, LSTMDecoder


class CaptionModel:
    def __init__(self, decoder_type: str = "lstm"):
        if decoder_type.lower() not in ("rnn", "lstm"):
            raise ValueError("decoder_type must be 'rnn' or 'lstm'")
        self.decoder_type = decoder_type.lower()

        self.embedding:        EmbeddingLayer = EmbeddingLayer()
        self.image_projection: DenseLayer     = DenseLayer()        # CNN → embed_dim
        self.decoder = None                                          # filled by load_*
        self.output_layer: DenseLayer = DenseLayer(activation="softmax")

        self.word2idx: dict = None
        self.idx2word: dict = None
        self.n_layers: int  = None

    def load_from_keras(self, keras_model, word2idx: dict, idx2word: dict) -> None:

        self.word2idx = word2idx
        self.idx2word = idx2word

        self.embedding.load_weights(keras_model.get_layer("embedding"))

        self.image_projection.load_weights(keras_model.get_layer("image_projection"))

        if self.decoder_type == "rnn":
            prefix  = "simple_rnn"
            CellCls = SimpleRNNCell
            DecCls  = SimpleRNNDecoder
        else:
            prefix  = "lstm"
            CellCls = LSTMCell
            DecCls  = LSTMDecoder

        cells = []
        i = 0
        while True:
            try:
                layer = keras_model.get_layer(f"{prefix}_{i}")
                cell = CellCls()
                cell.load_weights(layer)
                cells.append(cell)
                i += 1
            except ValueError:
                break

        if not cells:
            raise ValueError(
                f"No layers named '{prefix}_0', '{prefix}_1', ... found in model."
            )
        self.n_layers = len(cells)
        self.decoder  = DecCls(cells)

        self.output_layer.load_weights(keras_model.get_layer("output_dense"))

    def load_weights_manual(
        self,
        embedding_W:       np.ndarray,
        image_proj_W:      np.ndarray,
        image_proj_b:      np.ndarray,
        recurrent_weights: list,       # list of (W_x, W_h, b) tuples for RNN
                                       # or (W_x, W_h, b) for LSTM
        output_W:          np.ndarray,
        output_b:          np.ndarray,
        word2idx:          dict,
        idx2word:          dict,
    ) -> None:
        #buat testing
        self.word2idx = word2idx
        self.idx2word = idx2word

        self.embedding.set_weights(embedding_W)
        self.image_projection.set_weights(image_proj_W, image_proj_b)

        if self.decoder_type == "rnn":
            cells = []
            for W_x, W_h, b in recurrent_weights:
                cell = SimpleRNNCell()
                cell.set_weights(W_x, W_h, b)
                cells.append(cell)
            self.decoder = SimpleRNNDecoder(cells)
        else:
            cells = []
            for W_x, W_h, b in recurrent_weights:
                cell = LSTMCell()
                cell.set_weights(W_x, W_h, b)
                cells.append(cell)
            self.decoder = LSTMDecoder(cells)

        self.n_layers = len(recurrent_weights)
        self.output_layer.set_weights(output_W, output_b)

    def generate_caption(
        self,
        image_feature: np.ndarray,
        max_len: int = 30,
    ) -> str:
        assert self.decoder is not None, "Call load_from_keras() first."

        start_idx = self.word2idx["<start>"]
        end_idx   = self.word2idx["<end>"]
        pad_idx   = self.word2idx["<pad>"]

        hidden_size = self.decoder.cells[0].hidden_size

        if self.decoder_type == "rnn":
            h = [np.zeros(hidden_size, dtype=np.float32) for _ in range(self.n_layers)]

            x_img = self.image_projection.forward(image_feature.astype(np.float32))
            _, h = self.decoder.step(x_img, h)

            generated = []
            token = start_idx
            for _ in range(max_len):
                x = self.embedding.forward(token)
                out, h = self.decoder.step(x, h)
                logits = self.output_layer.forward(out)
                token  = int(np.argmax(logits))
                if token == end_idx:
                    break
                if token != pad_idx:
                    generated.append(self.idx2word.get(token, "<unk>"))

        else:  # lstm
            h = [np.zeros(hidden_size, dtype=np.float32) for _ in range(self.n_layers)]
            c = [np.zeros(hidden_size, dtype=np.float32) for _ in range(self.n_layers)]

            x_img = self.image_projection.forward(image_feature.astype(np.float32))
            _, h, c = self.decoder.step(x_img, h, c)

            generated = []
            token = start_idx
            for _ in range(max_len):
                x = self.embedding.forward(token)
                out, h, c = self.decoder.step(x, h, c)
                logits = self.output_layer.forward(out)
                token  = int(np.argmax(logits))
                if token == end_idx:
                    break
                if token != pad_idx:
                    generated.append(self.idx2word.get(token, "<unk>"))

        return " ".join(generated)

    def forward_sequence(
        self,
        image_feature:  np.ndarray,
        caption_tokens: np.ndarray,
    ) -> np.ndarray:

        assert self.decoder is not None

        x_img = self.image_projection.forward(image_feature.astype(np.float32))
        x_img = x_img[np.newaxis]                                    # (1, embed_dim)

        cap_emb = self.embedding.forward(np.array(caption_tokens))   # (seq_len, embed_dim)

        full_seq = np.concatenate([x_img, cap_emb], axis=0)          # (seq_len+1, embed_dim)

        if self.decoder_type == "rnn":
            outputs, _ = self.decoder.forward(full_seq)
        else:
            outputs, _, _ = self.decoder.forward(full_seq)

        logits = np.stack(
            [self.output_layer.forward(outputs[t]) for t in range(len(outputs))],
            axis=0,
        )   # (seq_len+1, vocab_size)

        return logits

    def generate_captions_batch(
        self,
        image_features: np.ndarray,
        max_len: int = 30,
        batch_size: int = 32,
    ) -> list:
        captions = []
        for i in range(0, len(image_features), batch_size):
            batch = image_features[i : i + batch_size]
            for feat in batch:
                captions.append(self.generate_caption(feat, max_len=max_len))
        return captions
